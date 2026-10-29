import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pipeline import jk9_design as d
from pipeline.publication import digest,write_json


class DesignAdapterTests(unittest.TestCase):
    def test_fresh_relative_chains_relocated_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);case=root/'fit';case.mkdir()
            paths=[]
            for i in range(4):
                p=case/f'chain{i}.csv';p.write_text('fixture'+str(i));paths.append(p)
            accepted=case/'accepted.json'
            record=dict(diagnostics=dict(passed=True),configuration=dict(raw_input_sha256='input',training_cutoff=100,
                        source_model_sha256=digest(d.ROOT/'pipeline/models/leading_order.stan')),
                        chain_paths=[p.name for p in paths],chain_sha256=[digest(p) for p in paths])
            write_json(accepted,record);original=accepted.read_bytes()
            with patch.object(d,'plan',return_value=[dict(input_sha256='input')]),patch.object(d,'raw_input_N',return_value=100):
                runtime=d.runtime_accepted(2445,root/'out',accepted)
                result=json.loads(runtime.read_text())
                self.assertEqual(result['chain_paths'],[str(p.resolve()) for p in paths])
                self.assertEqual(accepted.read_bytes(),original)
                paths[0].write_text('changed')
                with self.assertRaises(ValueError):d.runtime_accepted(2445,root/'out',accepted)

    def test_full_window_required_for_design(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);paths=[]
            for i in range(4):
                p=root/f'c{i}';p.write_text(str(i));paths.append(p)
            p=root/'accepted.json';write_json(p,dict(diagnostics=dict(passed=True),
                configuration=dict(raw_input_sha256='input',training_cutoff=75,
                    source_model_sha256=digest(d.ROOT/'pipeline/models/leading_order.stan')),
                chain_paths=[x.name for x in paths],chain_sha256=[digest(x) for x in paths]))
            with patch.object(d,'plan',return_value=[dict(input_sha256='input')]),patch.object(d,'raw_input_N',return_value=100):
                with self.assertRaises(ValueError):d.runtime_accepted(2445,root/'out',p)
