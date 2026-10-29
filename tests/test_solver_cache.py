from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline import solver_cache as cache


class SolverCacheTests(unittest.TestCase):
    def test_sqlite_many_points_single_file_and_parallel_writers(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, JK9_CACHE_MODE='sqlite'):
            folder = Path(tmp) / 'point_cache'
            values = [{'resolved': i % 2 == 0, 'B': i / 7, 'error': 'unresolved'} for i in range(40)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda i: cache.put(folder / f'{i}.json', values[i]), range(40)))
            self.assertEqual(list(folder.iterdir()), [folder / 'cache.sqlite3'])
            self.assertEqual([cache.get(folder / f'{i}.json') for i in range(40)], values)

    def test_memory_and_off_do_not_create_files(self):
        for selected in ('memory', 'off'):
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, JK9_CACHE_MODE=selected):
                path = Path(tmp) / 'point_cache/a.json'
                cache.clear_memory()
                cache.put(path, {'resolved': False})
                self.assertEqual(cache.get(path), {'resolved': False} if selected == 'memory' else None)
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_legacy_json_precedes_sqlite_and_is_never_modified(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, JK9_CACHE_MODE='sqlite'):
            path = Path(tmp) / 'a.json'
            path.write_text('{"resolved": false}')
            cache.put(path, {'resolved': True})
            self.assertEqual(cache.get(path), {'resolved': False})
            self.assertEqual(path.read_text(), '{"resolved": false}')


if __name__ == '__main__':
    unittest.main()
