import json
from pathlib import Path
import tempfile
import unittest

from pipeline.cache_store import pack, restore


class CacheStoreTests(unittest.TestCase):
    def test_verified_pack_prune_restore_and_changed_file_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / 'cache'
            cache.mkdir()
            original = b'{"failed":true,"value":0.123}\n'
            (cache / 'point.json').write_bytes(original)
            archive = root / 'store/1.tar.gz'
            result = pack(root, [cache], archive, prune=True)
            self.assertEqual(result['files'], 1)
            self.assertFalse(cache.exists())
            restore(root, archive)
            self.assertEqual((cache / 'point.json').read_bytes(), original)
            (cache / 'point.json').write_bytes(b'new result')
            with self.assertRaises(FileExistsError):
                restore(root, archive)

    def test_corrupt_archive_cannot_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / 'cache'
            cache.mkdir()
            (cache / 'point.json').write_text('{}')
            archive = root / 'store/1.tar.gz'
            pack(root, [cache], archive, prune=True)
            archive.write_bytes(archive.read_bytes() + b'corruption')
            with self.assertRaises(ValueError):
                restore(root, archive)
            self.assertFalse(cache.exists())


if __name__ == '__main__':
    unittest.main()
