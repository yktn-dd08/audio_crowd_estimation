"""Self-contained HTML player; no Python audio or plotting dependencies required."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path


class SyncPlayer:
    """Display synchronized audio/CSV plots in Jupyter, or export a standalone HTML.

    Paths are read by Python, so this also works with a remote notebook kernel.
    Files can instead be selected inside the displayed player. CSV column mappings
    and the trajectory time offset can be changed interactively.
    """

    def __init__(self, audio_path=None, trajectory_path=None, *, height=900):
        self.height = int(height)
        if self.height < 400:
            raise ValueError("height must be at least 400 pixels")
        self._payload = {}
        if audio_path is not None:
            path = Path(audio_path).expanduser()
            self._payload["audio"] = base64.b64encode(path.read_bytes()).decode("ascii")
            self._payload["audioName"] = path.name
        if trajectory_path is not None:
            path = Path(trajectory_path).expanduser()
            self._payload["csv"] = path.read_text(encoding="utf-8-sig")
            self._payload["csvName"] = path.name

    def to_html(self):
        """Return a portable HTML document, including the input data."""
        root = Path(__file__).parent
        payload = base64.b64encode(
            json.dumps(self._payload, ensure_ascii=False).encode("utf-8")
        ).decode("ascii")
        return (root / "player.html").read_text(encoding="utf-8").replace(
            "/* PLAYER_SCRIPT */", (root / "player.js").read_text(encoding="utf-8")
        ).replace("__PLAYER_PAYLOAD__", payload)

    def save(self, path):
        """Write a standalone HTML file that can be opened in a browser."""
        path = Path(path).expanduser().resolve()
        path.write_text(self.to_html(), encoding="utf-8")
        return path

    def _repr_html_(self):
        # Isolate each viewer's DOM, styles and JavaScript from the notebook.
        return (
            '<iframe title="Audio trajectory sync player" '
            'allow="autoplay; fullscreen" allowfullscreen '
            f'style="width:100%;height:{self.height}px;border:0" '
            f'srcdoc="{html.escape(self.to_html(), quote=True)}"></iframe>'
        )

    def display(self):
        """Display in Jupyter Notebook / JupyterLab and return this player."""
        from IPython.display import HTML, display

        display(HTML(self._repr_html_()))
        return self


def show_sync_player(
    audio_path=None, trajectory_path=None, *, height=900,
    server_files=False, start_dir=None,
):
    """Display a player, optionally with a kernel-side file browser.

    Set server_files=True in remote notebooks to browse the server filesystem.
    This mode requires ipywidgets; standalone HTML keeps browser-local inputs.
    """
    if server_files:
        from .server_picker import ServerFilePlayer

        return ServerFilePlayer(
            audio_path, trajectory_path, height=height, start_dir=start_dir
        ).display()
    if start_dir is not None:
        raise ValueError("start_dir requires server_files=True")
    return SyncPlayer(audio_path, trajectory_path, height=height).display()
