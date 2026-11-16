"""Run with: python -m unittest discover -s gui/tests"""
import base64
import json
from pathlib import Path
import re
import tempfile
import unittest

from gui import SyncPlayer


class PlayerTests(unittest.TestCase):
    def test_export_preserves_bytes_and_unsafe_csv_as_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / 'audio.wav'
            audio.write_bytes(b'RIFF\x00\xffWAVE')
            csv = root / 'trajectory.csv'
            content = 'time,x,y,id\n0,1,2,"</script>日本語"\n'
            csv.write_text('\ufeff' + content, encoding='utf-8')
            viewer = SyncPlayer(audio, csv)
            target = viewer.save(root / 'viewer.html')
            document = target.read_text(encoding='utf-8')
            encoded = re.search(r'id="payload"[^>]*>(.*?)</script>', document).group(1)
            payload = json.loads(base64.b64decode(encoded))
            self.assertEqual(base64.b64decode(payload['audio']), audio.read_bytes())
            self.assertEqual(payload['csv'], content)
            self.assertNotIn('</script>日本語', document)
            self.assertNotIn('/* PLAYER_SCRIPT */', document)
            self.assertIn('srcdoc="&lt;!doctype html&gt;', viewer._repr_html_())

    def test_empty_viewer_and_validation(self):
        self.assertIn('id="audioFile"', SyncPlayer().to_html())
        with self.assertRaises(ValueError):
            SyncPlayer(height=200)
        with self.assertRaises(FileNotFoundError):
            SyncPlayer('/does/not/exist.wav')


if __name__ == '__main__':
    unittest.main()
