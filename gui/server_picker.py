"""Jupyter widgets that browse the kernel's filesystem, not the browser's PC."""
from __future__ import annotations

import html
from pathlib import Path

from .player import SyncPlayer

AUDIO_SUFFIXES = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.aif', '.aiff', '.opus'}


def _widgets():
    try:
        import ipywidgets
    except ImportError as exc:
        raise ImportError(
            'サーバー側のファイル選択にはipywidgetsが必要です。'
            'Notebookのセルで %pip install "ipywidgets>=8,<9" を実行し、'
            'カーネルを再起動してください。'
        ) from exc
    return ipywidgets


class ServerFilePicker:
    """Single-file selector with directory navigation and extension filtering."""

    def __init__(self, label, start_dir, suffixes, on_select):
        w = _widgets()
        self.suffixes = suffixes
        self.on_select = on_select
        self.selected_path = None
        self.directory = Path(start_dir).expanduser().resolve()
        self.location = w.Text(description='フォルダ', layout=w.Layout(width='100%'))
        self.go = w.Button(description='移動')
        self.up = w.Button(description='上のフォルダ')
        self.refresh = w.Button(description='更新')
        self.entries = w.Select(rows=9, layout=w.Layout(width='100%'))
        self.open = w.Button(description='フォルダを開く / 選択', layout=w.Layout(width='240px'))
        self.message = w.HTML()
        self.selection = w.HTML(value=f'{html.escape(label)}: 未選択')
        self.widget = w.VBox([
            self.location, w.HBox([self.go, self.up, self.refresh]),
            self.entries, self.open, self.message, self.selection,
        ])
        self.go.on_click(lambda _: self.navigate(self.location.value))
        self.up.on_click(lambda _: self.navigate(self.directory.parent))
        self.refresh.on_click(lambda _: self.navigate(self.directory))
        self.open.on_click(self._open)
        self.entries.observe(lambda _: self._update_button(), names='value')
        self.navigate(self.directory)

    def _update_button(self):
        self.open.disabled = self.entries.value is None

    def navigate(self, path):
        try:
            target = Path(path).expanduser()
            if not target.is_absolute():
                target = self.directory / target
            target = target.resolve(strict=True)
            if not target.is_dir():
                raise NotADirectoryError(f'フォルダではありません: {target}')
            folders, files = [], []
            for entry in target.iterdir():
                try:
                    if entry.is_dir():
                        folders.append((f'[フォルダ] {entry.name}', str(entry)))
                    elif entry.is_file() and entry.suffix.lower() in self.suffixes:
                        files.append((entry.name, str(entry)))
                except OSError:
                    continue
            # List only one level: never recursively traverse a large dataset.
            options = sorted(folders, key=lambda v: v[0].casefold())
            options += sorted(files, key=lambda v: v[0].casefold())
            self.directory = target
            self.location.value = str(target)
            self.entries.options = options
            self.entries.value = None
            self.message.value = '' if options else '対応するファイル・フォルダがありません。'
            self.up.disabled = target == target.parent
            self._update_button()
            return True
        except (OSError, ValueError) as exc:
            self.message.value = f'<span style="color:#c33">{html.escape(str(exc))}</span>'
            return False

    def _open(self, _=None):
        if self.entries.value is None:
            return
        path = Path(self.entries.value)
        if path.is_dir():
            self.navigate(path)
        else:
            try:
                self.select_file(path)
            except (OSError, ValueError) as exc:
                self.message.value = f'<span style="color:#c33">{html.escape(str(exc))}</span>'

    def select_file(self, path):
        path = Path(path).expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.lower() not in self.suffixes:
            raise ValueError(f'対応するファイルを選択してください: {path}')
        self.selected_path = path
        self.selection.value = f'選択済み: <code>{html.escape(str(path))}</code>'
        self.message.value = ''
        self.on_select()


class ServerFilePlayer:
    """Server-side file controls above the existing synchronized HTML player.

    Requires a live notebook kernel. No HTTP file server, extra port, or local
    filesystem mount is used. The selected data travels through Jupyter output.
    """

    def __init__(self, audio_path=None, trajectory_path=None, *, start_dir=None, height=900):
        w = _widgets()
        self.height = SyncPlayer(height=height).height
        self.player = None
        start = Path.cwd() if start_dir is None else Path(start_dir).expanduser()
        if not start.is_dir():
            raise NotADirectoryError(f'開始フォルダが存在しません: {start}')
        self.load = w.Button(description='選択した2ファイルを読み込む', disabled=True,
                             button_style='primary', layout=w.Layout(width='260px'))
        self.message = w.HTML()
        self.output = w.Output()
        self.audio = ServerFilePicker('音声', start, AUDIO_SUFFIXES, self._selection_changed)
        self.trajectory = ServerFilePicker('軌跡CSV', start, {'.csv'}, self._selection_changed)
        self.choosers = w.Accordion(children=[self.audio.widget, self.trajectory.widget])
        self.choosers.set_title(0, '1. サーバーの音声ファイルを選択')
        self.choosers.set_title(1, '2. サーバーの軌跡CSVを選択')
        self.choosers.selected_index = 0
        self.widget = w.VBox([
            w.HTML('<b>サーバー側のファイル選択</b>：フォルダやファイルをクリックし、'
                   '「フォルダを開く / 選択」を押してください。'),
            self.choosers, self.load, self.message, self.output,
        ])
        self.load.on_click(self._load)
        if audio_path is not None:
            path = Path(audio_path).expanduser().resolve()
            self.audio.navigate(path.parent)
            self.audio.select_file(path)
        if trajectory_path is not None:
            path = Path(trajectory_path).expanduser().resolve()
            self.trajectory.navigate(path.parent)
            self.trajectory.select_file(path)
        if audio_path is not None and trajectory_path is not None:
            self._load()

    def _selection_changed(self):
        # Pickers can call this while the second picker is still being built.
        audio = getattr(self, 'audio', None)
        trajectory = getattr(self, 'trajectory', None)
        ready = audio and trajectory and audio.selected_path and trajectory.selected_path
        self.load.disabled = not bool(ready)
        if ready:
            self.message.value = '2ファイルを選択済みです。「読み込む」を押すと反映します。'
        elif audio and audio.selected_path and hasattr(self, 'choosers'):
            self.choosers.selected_index = 1

    def _load(self, _=None):
        if self.load.disabled:
            return
        self.load.disabled = True
        self.message.value = 'サーバーから読み込み中…'
        try:
            player = SyncPlayer(self.audio.selected_path, self.trajectory.selected_path,
                                height=self.height)
            player._payload['serverFiles'] = True
            self.output.outputs = ()
            self.output.append_display_data(player)
            self.player = player
            self.message.value = '読み込み完了。下のプレイヤーで再生できます。'
            self.choosers.selected_index = None
        except (OSError, UnicodeError, ValueError) as exc:
            self.message.value = f'<span style="color:#c33">読み込めません: {html.escape(str(exc))}</span>'
        finally:
            self.load.disabled = not bool(self.audio.selected_path and self.trajectory.selected_path)

    def display(self):
        from IPython.display import display

        display(self.widget)
        return self

    def _ipython_display_(self):
        self.display()
