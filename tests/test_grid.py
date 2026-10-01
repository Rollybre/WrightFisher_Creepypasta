"""Tests de la production de grilles par blocs (exécutée en sous-processus : `python run_grid.py`, avec parallélisme)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_grid import (_chunk_metric, chunk_path, list_chunks, load_counts, load_results, merge_csv,
                             upgrade_chunks)
from simulation import Simulation, compute_metrics

ROOT = Path(__file__).resolve().parents[1]


def run_grid(out_dir, *extra, n_sims=1000, chunk_size=250, n_jobs=2):
    cmd = [sys.executable, "run_grid.py", "--n_sims", str(n_sims), "--chunk_size", str(chunk_size),
           "--n_jobs", str(n_jobs), "--out_dir", str(out_dir), *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


class TestGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name) / "g"
        r = run_grid(cls.dir)
        assert r.returncode == 0, r.stderr

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_blocks_and_results(self):
        self.assertEqual(list_chunks(self.dir), [0, 1, 2, 3])
        res = load_results(self.dir)
        self.assertEqual(len(res["seed"]), 1000)
        self.assertTrue((res["seed"] == np.arange(1, 1001)).all())
        self.assertTrue((res["n_f"] == 10000).all())

    def test_distribution_matches_indices_and_replay(self):
        res = load_results(self.dir)
        for i in (0, 333, 999):
            counts = load_counts(self.dir, i, 250)
            self.assertEqual(int(counts.sum()), 10000)
            m = compute_metrics(counts)
            for k in ("gini_obs", "hill_1", "chao1", "kl_emp"):
                self.assertAlmostEqual(m[k], float(res[k][i]), places=9)
            # rejeu exact (même machine, mêmes versions) à partir des paramètres exacts
            sim = Simulation(int(res["n_i"][i]), int(res["n_f"][i]), int(res["C"][i]), int(res["T"][i]),
                             float(res["alpha"][i]), float(res["mu"][i]), float(res["q"][i]), int(res["seed"][i])).run()
            self.assertTrue(np.array_equal(np.sort(sim.archive)[::-1], counts))

    def test_resume_skips_existing_blocks(self):
        r = run_grid(self.dir)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("4 déjà faits, 0 à faire", r.stdout)

    def test_config_guard_refuses_other_bounds(self):
        r = run_grid(self.dir, "--range", "mu", "0", "0.02")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("autre configuration", r.stdout + r.stderr)

    def test_override_n_f_and_range(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "g2"
            r = run_grid(out, "--fix", "n_f", "30901", "--range", "mu", "0", "0.02", n_sims=500, chunk_size=250)
            self.assertEqual(r.returncode, 0, r.stderr)
            res = load_results(out)
            self.assertTrue((res["n_f"] == 30901).all())
            self.assertLessEqual(res["mu"].max(), 0.02)
            with open(out / "grid_config.json") as f:
                cfg = json.load(f)
            self.assertEqual(cfg["fixed"]["n_f"], 30901)

    def test_missing_metrics_recomputed_and_upgrade(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "g3"
            self.assertEqual(run_grid(out, n_sims=500, chunk_size=250).returncode, 0)
            ref = load_results(out)
            for c in list_chunks(out):                       # simule d'anciens blocs sans kl_emp / gini_obs
                p = chunk_path(str(out), c)
                with np.load(p) as z:
                    payload = {k: z[k] for k in z.files if k not in ("m_kl_emp", "m_gini_obs")}
                np.savez_compressed(p, **payload)
            again = load_results(out)                        # recalculées à la volée
            for k in ("kl_emp", "gini_obs"):
                self.assertTrue(np.allclose(again[k], ref[k], rtol=1e-9, atol=1e-12))
            self.assertEqual(upgrade_chunks(str(out), 1), 2)  # écrites définitivement
            with np.load(chunk_path(str(out), 0)) as z:
                self.assertIn("m_kl_emp", z.files)
            self.assertEqual(upgrade_chunks(str(out), 1), 0)

    def test_empirical_growth_grid(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "g4"
            self.assertEqual(run_grid(out, "--growth", "empirical", n_sims=500, chunk_size=250).returncode, 0)
            with open(out / "grid_config.json") as f:
                self.assertEqual(json.load(f)["growth"], "empirical")
            res = load_results(out)
            i = 123
            counts = load_counts(out, i, 250)
            self.assertEqual(int(counts.sum()), 10000)
            sim = Simulation(int(res["n_i"][i]), int(res["n_f"][i]), int(res["C"][i]), int(res["T"][i]), float(res["alpha"][i]),
                             float(res["mu"][i]), float(res["q"][i]), int(res["seed"][i]), growth="empirical").run()
            self.assertTrue(np.array_equal(np.sort(sim.archive)[::-1], counts))          # rejeu avec le profil de croissance
            self.assertEqual(run_grid(out, "--growth", "empirical", n_sims=500, chunk_size=250).returncode, 0)   # reprise
            r = run_grid(out, n_sims=500, chunk_size=250)                                  # autre croissance : refusé
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("autre configuration", r.stdout + r.stderr)

    def test_old_config_without_growth_key_is_linear(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "g5"
            self.assertEqual(run_grid(out, n_sims=250, chunk_size=250).returncode, 0)
            cfg_path = out / "grid_config.json"
            with open(cfg_path) as f:
                cfg = json.load(f)
            cfg.pop("growth")                                                # configuration écrite avant l'ajout de growth
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)
            self.assertEqual(run_grid(out, n_sims=250, chunk_size=250).returncode, 0)      # reprise en linéaire : accepté
            self.assertNotEqual(run_grid(out, "--growth", "empirical", n_sims=250, chunk_size=250).returncode, 0)

    def test_merge_csv_exact_parameters(self):
        out_csv = Path(self.tmp.name) / "r.csv"
        merge_csv(str(self.dir), str(out_csv))
        import pandas as pd
        df = pd.read_csv(out_csv, float_precision="round_trip")
        res = load_results(self.dir)
        self.assertEqual(len(df), 1000)
        self.assertTrue(np.array_equal(df["q"].to_numpy(), res["q"]))     # 17 chiffres : relecture exacte


if __name__ == "__main__":
    unittest.main()
