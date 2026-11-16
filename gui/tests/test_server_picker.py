"""Kernel-side selection tests. Install gui/requirements-notebook.txt to run."""
import base64
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

try:
    import ipywidgets
except ImportError:
    ipywidgets = None

from gui.server_picker import ServerFilePicker, ServerFilePlayer


@unittest.skipIf(ipywidgets is None, 'ipywidgets is an optional notebook dependency')
class ServerPickerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.folder = self.root / '音声 <test>'
        self.folder.mkdir()
        self.audio = self.folder / 'sample.WAV'
        self.audio.write_bytes(b'RIFF\x00WAVE')
        self.csv = self.root / 'trajectory.csv'
        self.csv.write_text('time,x,y\n0,1,2\n', encoding='utf-8')
        (self.root / 'unrelated.txt').write_text('hidden')
        self.player = ServerFilePlayer(start_dir=self.root)

    def test_gui_navigation_selection_and_output(self):
        p = self.player
        self.assertTrue(p.load.disabled)
        options = dict(p.audio.entries.options)
        self.assertIn('[フォルダ] 音声 <test>', options)
        self.assertNotIn('trajectory.csv', options)
        p.audio.entries.value = str(self.folder)
        p.audio.open.click()
        self.assertEqual(p.audio.directory, self.folder.resolve())
        p.audio.entries.value = str(self.audio)
        p.audio.open.click()
        self.assertEqual(p.audio.selected_path, self.audio.resolve())
        self.assertIn('&lt;test&gt;', p.audio.selection.value)
        self.assertEqual(p.choosers.selected_index, 1)
        p.trajectory.entries.value = str(self.csv)
        p.trajectory.open.click()
        self.assertFalse(p.load.disabled)
        p.load.click()
        self.assertEqual(base64.b64decode(p.player._payload['audio']), self.audio.read_bytes())
        self.assertEqual(p.player._payload['csv'], self.csv.read_text())
        self.assertTrue(p.player._payload['serverFiles'])
        self.assertIsNone(p.choosers.selected_index)
        self.assertEqual(len(p.output.outputs), 1)
        self.assertIn('srcdoc=', p.output.outputs[0]['data']['text/html'])
        self.assertIn('読み込み完了', p.message.value)
        p.load.click()
        self.assertEqual(len(p.output.outputs), 1)
        p.audio.up.click()
        self.assertEqual(p.audio.directory, self.root.resolve())
        self.assertEqual(p.audio.selected_path, self.audio.resolve())

    def test_navigation_and_read_failures_preserve_loaded_player(self):
        p = self.player
        p.audio.select_file(self.audio)
        p.trajectory.select_file(self.csv)
        p.load.click()
        previous, previous_output = p.player, p.output.outputs
        self.assertFalse(p.audio.navigate(self.root / 'missing'))
        self.assertEqual(p.audio.directory, self.root.resolve())
        self.csv.unlink()
        p.load.click()
        self.assertIs(p.player, previous)
        self.assertEqual(p.output.outputs, previous_output)
        self.assertIn('読み込めません', p.message.value)
        self.assertFalse(p.load.disabled)
        with self.assertRaises(ValueError):
            p.audio.select_file(self.root / 'unrelated.txt')

    def test_public_entry_point_and_prefilled_files(self):
        from gui import show_sync_player
        with patch.object(ServerFilePlayer, 'display', lambda self: self):
            p = show_sync_player(self.audio, self.csv, server_files=True, start_dir=self.root)
        self.assertIsNotNone(p.player)
        self.assertEqual(p.audio.directory, self.folder.resolve())
        with self.assertRaises(ValueError):
            show_sync_player(start_dir=self.root)
        with self.assertRaises(NotADirectoryError):
            ServerFilePlayer(start_dir=self.root / 'missing')
