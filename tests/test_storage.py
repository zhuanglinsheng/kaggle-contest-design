import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pipeline import storage


class StorageTests(unittest.TestCase):
    def test_historical_absolute_record_resolves_after_checkout_and_work_move(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();history=root/'tmp/history'
            target=history/'docs/review/case/chain.csv';target.parent.mkdir(parents=True);target.write_bytes(b'original chain')
            with patch.object(storage,'ROOT',root),patch.object(storage,'HISTORY',history):
                resolved=storage.historical_path('/old/check-out/docs/review/case/chain.csv')
            self.assertEqual(resolved,target)
            self.assertEqual(resolved.read_bytes(),b'original chain')

    def test_specific_relocation_finds_working_inputs_without_rewriting_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();target=root/'tmp/inputs/main/contest_1.json'
            target.parent.mkdir(parents=True);target.write_text('{"frozen": true}')
            inventory=root/'docs/storage_layout.json';inventory.parent.mkdir()
            inventory.write_text(json.dumps(dict(moves=[dict(source='metakaggle/__jsondata__',destination='tmp/inputs/main')])) )
            with patch.object(storage,'ROOT',root),patch.object(storage,'HISTORY',root/'tmp/history'):
                self.assertEqual(storage.historical_path(root/'metakaggle/__jsondata__/contest_1.json'),target)

    def test_stan_staging_puts_build_next_to_tmp_copy_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();source=root/'pipeline/models/model.stan'
            source.parent.mkdir(parents=True);source.write_text('parameters {real x;} model {x ~ normal(0,1);}')
            original=source.read_bytes()
            with patch.object(storage,'ROOT',root),patch.object(storage,'BUILD',root/'tmp/build'):
                staged=storage.stage_stan(source)
                self.assertEqual(staged,root/'tmp/build/models/pipeline/models/model.stan')
                self.assertEqual(staged.read_bytes(),original)
                self.assertEqual(source.read_bytes(),original)
                self.assertFalse(source.with_suffix('').exists())

    def test_archived_theoretical_text_is_retained_and_chain_stdout_is_work(self):
        from pipeline import organize
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();release=root/'results/releases/fixture'
            release.mkdir(parents=True);(release/'manifest.json').write_text('{"files": {}}')
            note=root/'archive/research/precision_conditions/PROOF.txt'
            note.parent.mkdir(parents=True);note.write_text('An analytical derivation, not a solver log.')
            log=root/'docs/review/fixture/chain_1-stdout.txt'
            log.parent.mkdir(parents=True);log.write_text('Sampling progress')
            with patch.object(organize,'ROOT',root),patch.object(organize,'RELEASE',release):
                retained=organize.protected_files()
            self.assertIn(note,retained)
            self.assertNotIn(log,retained)
