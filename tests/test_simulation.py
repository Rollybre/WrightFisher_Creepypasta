"""Tests du modèle et des indices (stdlib unittest : `python -m unittest discover -s tests -t .`, ou pytest)."""
import unittest

import numpy as np

from simulation import (EMPIRICAL_COUNTS, GROWTHS, Simulation, chao1, compute_metrics, draw_params,
                        generate_param_table, gini, growth_sizes, hill_number, kl_divergence)

P = dict(n_i=50, n_f=5000, C=40, T=100, alpha=1.0, mu=0.01, q=0.8)


class TestSimulation(unittest.TestCase):
    def test_archive_final_size_is_n_f(self):
        for n_i, n_f, T in [(1, 3000, 300), (100, 10000, 777), (37, 20000, 450)]:
            sim = Simulation(n_i, n_f, 30, T, 1.0, 0.02, 1.0, seed=3).run()
            self.assertEqual(int(sim.archive.sum()), n_f)

    def test_reproducible_with_same_seed(self):
        a = Simulation(**P, seed=7).run().archive
        b = Simulation(**P, seed=7).run().archive
        c = Simulation(**P, seed=8).run().archive
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(len(a) == len(c) and np.array_equal(a, c))

    def test_no_innovation_keeps_initial_classes(self):
        sim = Simulation(**{**P, "mu": 0.0}, seed=1).run()
        self.assertEqual(len(sim.archive), P["C"])

    def test_innovation_creates_classes(self):
        sim = Simulation(**{**P, "mu": 0.05}, seed=1).run()
        self.assertGreater(len(sim.archive), P["C"])

    def test_n_f_below_n_i_raises(self):
        with self.assertRaises(ValueError):
            Simulation(10, 5, 5, 10, 1.0, 0.1, 1.0).run()

    def test_history_shape(self):
        sim = Simulation(**P, seed=1).run(track_history=True)
        self.assertEqual(len(sim.history["t"]), P["T"] + 1)

    def test_metric_properties(self):
        for seed in range(5):
            m = Simulation(**P, seed=seed).run().metrics
            self.assertLessEqual(m["gini_obs"], m["gini"] + 1e-12)
            self.assertTrue(0 <= m["gini_obs"] <= 1)
            self.assertGreaterEqual(m["hill_0"], m["hill_1"])
            self.assertGreaterEqual(m["hill_1"], m["hill_2"])
            self.assertGreaterEqual(m["hill_2"], m["hill_inf"] - 1e-9)
            self.assertEqual(m["hill_0"], m["n_classes"])
            self.assertGreaterEqual(m["chao1"], m["n_classes"])
            self.assertGreaterEqual(m["kl_emp"], 0)


class TestIndices(unittest.TestCase):
    def test_gini_extremes(self):
        self.assertAlmostEqual(gini(np.full(10, 5.0)), 0.0)
        self.assertGreater(gini(np.array([0, 0, 0, 0, 100.0])), 0.7)
        self.assertTrue(np.isnan(gini(np.zeros(5))))

    def test_hill_numbers(self):
        x = np.full(8, 3.0)                       # 8 classes équiprobables : tous les ordres valent 8
        for order in (0, 1, 2, np.inf):
            self.assertAlmostEqual(hill_number(x, order), 8.0)
        with self.assertRaises(ValueError):
            hill_number(x, -1)

    def test_chao1(self):
        self.assertEqual(chao1(np.array([1, 1, 2, 5])), 4 + 2 ** 2 / (2 * 1))

    def test_kl_divergence(self):
        emp = np.array(EMPIRICAL_COUNTS)
        self.assertAlmostEqual(kl_divergence(emp), 0.0)
        self.assertAlmostEqual(kl_divergence(emp * 3), 0.0)          # invariant d'échelle
        self.assertTrue(np.isnan(kl_divergence(np.zeros(4))))
        self.assertGreater(kl_divergence(np.full(176, 100)), 1.0)    # distribution uniforme : loin
        self.assertGreater(kl_divergence(np.full(400, 25)), kl_divergence(np.full(176, 100)))   # masse hors rang 176 pénalisée
        self.assertTrue(np.isfinite(kl_divergence(np.full(10, 50))))  # rangs non atteints : plancher, pas d'infini

    def test_compute_metrics_keys(self):
        m = compute_metrics(np.array([5, 3, 1, 0]))
        self.assertEqual(set(m), {"gini", "gini_obs", "hill_0", "hill_1", "hill_2", "hill_inf", "chao1", "n_classes", "kl_emp"})
        self.assertEqual(m["n_classes"], 3)


class TestParamTable(unittest.TestCase):
    def test_draw_params_bounds_and_shape(self):
        p = draw_params(np.random.default_rng(0), 1000)
        for name in ("q", "mu", "alpha", "C", "n_i", "T", "n_f"):
            self.assertEqual(len(p[name]), 1000)
        self.assertTrue((p["q"] >= 0.5).all() and (p["q"] <= 1.5).all())
        self.assertTrue((p["C"] >= 2).all() and (p["C"] <= 200).all())
        self.assertTrue((p["n_f"] == 10000).all())

    def test_table_deterministic(self):
        self.assertEqual(generate_param_table(20, rng_seed=3, seed_start=7), generate_param_table(20, rng_seed=3, seed_start=7))
        self.assertEqual([r["seed"] for r in generate_param_table(3, seed_start=7)], [7, 8, 9])



class TestGrowth(unittest.TestCase):
    def test_linear_is_unchanged(self):
        for n_i, n_f, T in [(1, 3000, 300), (76, 30901, 600), (100, 10000, 777)]:
            old = np.rint(np.linspace(n_i, n_f, T + 1)).astype(np.int64)      # ancien calcul (grilles déjà produites)
            self.assertTrue(np.array_equal(growth_sizes(n_i, n_f, T, "linear"), old))

    def test_empirical_profile(self):
        for n_i, n_f, T in [(1, 3000, 300), (76, 30901, 600), (100, 10000, 1000)]:
            sz = growth_sizes(n_i, n_f, T, "empirical")
            self.assertEqual((sz[0], sz[-1], len(sz)), (n_i, n_f, T + 1))
            self.assertTrue((np.diff(sz) >= 0).all())
            lin = growth_sizes(n_i, n_f, T, "linear")
            self.assertGreater(sz[int(0.37 * T)], lin[int(0.37 * T)] + 0.1 * (n_f - n_i))   # croissance rapide au début (2012-2015)

    def test_simulation_final_size_with_empirical_growth(self):
        for T in (300, 600, 1000):
            sim = Simulation(50, 8000, 40, T, 1.0, 0.01, 0.8, seed=2, growth="empirical").run()
            self.assertEqual(int(sim.archive.sum()), 8000)

    def test_growth_changes_the_run_and_is_reproducible(self):
        a = Simulation(**P, seed=4, growth="empirical").run().archive
        b = Simulation(**P, seed=4, growth="empirical").run().archive
        c = Simulation(**P, seed=4, growth="linear").run().archive
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(len(a) == len(c) and np.array_equal(a, c))

    def test_invalid_growth_raises(self):
        with self.assertRaises(ValueError):
            Simulation(**P, growth="exponential")
        with self.assertRaises(ValueError):
            growth_sizes(1, 10, 5, "exponential")
        self.assertEqual(set(GROWTHS), {"linear", "empirical"})


if __name__ == "__main__":
    unittest.main()
