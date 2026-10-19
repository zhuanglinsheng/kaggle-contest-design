"""Publication checks must catch stale and missing artifacts without mutations."""
from pathlib import Path
import tempfile
import unittest

from pipeline.publication import assert_assets, check_files, digest, local, tex_dependencies


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_nested_appendix_graphic_is_checked_and_comments_are_ignored(self):
        (self.root / 'main.tex').write_text('% \\input{missing}\n\\input{appendix}')
        (self.root / 'appendix.tex').write_text('\\includegraphics[width=1cm]{figure}')
        (self.root / 'figure.pdf').write_bytes(b'figure')
        dependencies = tex_dependencies(self.root / 'main.tex', self.root)
        self.assertEqual({p.name for p in dependencies}, {'main.tex', 'appendix.tex', 'figure.pdf'})
        (self.root / 'figure.pdf').unlink()
        with self.assertRaises(FileNotFoundError):
            tex_dependencies(self.root / 'main.tex', self.root)

    def test_release_paths_cannot_escape_checkout(self):
        for name in ('../outside', '/tmp/outside'):
            with self.assertRaises(ValueError):
                local(self.root, name)

    def test_stale_asset_and_changed_file_are_rejected_without_repair(self):
        source, destination = self.root / 'source.pdf', self.root / 'paper.pdf'
        source.write_bytes(b'current')
        destination.write_bytes(b'historical')
        with self.assertRaises(ValueError):
            assert_assets(self.root, {'assets': {'paper.pdf': 'source.pdf'}})
        self.assertEqual(destination.read_bytes(), b'historical')
        locked = {'source.pdf': digest(source)}
        source.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            check_files(self.root, locked)
        self.assertEqual(source.read_bytes(), b'changed')

    def test_nested_tex_path_escape_is_rejected(self):
        (self.root / 'main.tex').write_text('\\input{../outside}')
        with self.assertRaises(ValueError):
            tex_dependencies(self.root / 'main.tex', self.root)


if __name__ == '__main__':
    unittest.main()
