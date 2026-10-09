"""Publishing requires matched, converged full and early fits."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline.empirical import finalize
from pipeline.empirical.estimate import stan_data
from pipeline.config import resolve_manifest_path


class EmpiricalProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.fit = root / "main"
        self.fit.mkdir()
        self.data = root / "raw"
        self.data.mkdir()
        self.raw = dict(N_Delta=124, Delta2f=1 / 24, theta=100,
                        hat_t_i=[25, 90, 110], hat_t_j=[30, 120],
                        hat_y=[0.0] * 124, real_y=[1.0] * 124)
        (self.data / "contest_1.json").write_text(json.dumps(self.raw))
        self.write_csv(self.fit / "posterior_summary_final.csv",
                       [dict(contest_id=1, mode="full")])
        self.run_dirs = {}
        for mode, source in (("full", "full_current_data"),
                             ("early", "early_deadline_corrected")):
            run = root / source / mode / "contest_1"
            run.mkdir(parents=True)
            (run / "data.json").write_text(json.dumps(stan_data(self.raw, mode)))
            row = dict(contest_id=1, mode=mode, divergences=0)
            row.update({f"{name}_rhat": 1.0 for name in
                        ("c_i", "c_j", "sigma", "lambda", "mu_0", "r")})
            self.write_csv(root / source / "posterior_summary.csv", [row])
            self.run_dirs[mode] = run
        self.addCleanup(patch.stopall)
        patch.object(finalize, "FIT_DIR", self.fit).start()
        patch.object(finalize, "MAIN_DATA_DIR", self.data).start()

    @staticmethod
    def write_csv(path, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_matching_fits_publish_checksum_and_chain_location(self):
        finalize.main()
        rows = finalize.read_rows(self.fit / "accepted_chain_sources.csv")
        manifest = self.fit / "accepted_chain_sources.csv"
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(len(row["input_sha256"]) == 64 for row in rows))
        self.assertTrue(all(not Path(row["run_dir"]).is_absolute() and
                            not Path(row["raw_input_path"]).is_absolute()
                            for row in rows))
        self.assertTrue(all(resolve_manifest_path(row["run_dir"], manifest).is_dir()
                            for row in rows))
        self.assertTrue(all(resolve_manifest_path(row["raw_input_path"], manifest).read_bytes() ==
                            (self.data / "contest_1.json").read_bytes()
                            for row in rows))

    def test_changed_public_inputs_block_publication(self):
        data = dict(self.raw, N_Delta=125, hat_y=[0.0] * 125)
        (self.data / "contest_1.json").write_text(json.dumps(data))
        with self.assertRaisesRegex(RuntimeError, "do not match"):
            finalize.main()
        self.assertEqual(len(finalize.read_rows(
            self.fit / "posterior_summary_final.csv")), 1)

    def test_bad_fit_blocks_publication(self):
        path = self.run_dirs["full"].parents[1] / "posterior_summary.csv"
        rows = finalize.read_rows(path)
        rows[0]["divergences"] = 1
        self.write_csv(path, rows)
        with self.assertRaisesRegex(RuntimeError, "requires refinement"):
            finalize.main()

    def test_missing_convergence_diagnostic_blocks_publication(self):
        path = self.run_dirs["full"].parents[1] / "posterior_summary.csv"
        rows = finalize.read_rows(path)
        rows[0]["lambda_rhat"] = "nan"
        self.write_csv(path, rows)
        with self.assertRaisesRegex(RuntimeError, "requires refinement"):
            finalize.main()


if __name__ == "__main__":
    unittest.main()
