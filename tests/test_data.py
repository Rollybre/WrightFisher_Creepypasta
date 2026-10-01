"""Les comptes empiriques écrits en dur correspondent bien aux données du dépôt."""
import ast
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd

from simulation import EMPIRICAL_COUNTS

DATA = Path(__file__).resolve().parents[1] / "data" / "fandom_data.csv"


class TestEmpiricalCounts(unittest.TestCase):
    def test_hardcoded_counts_match_data(self):
        df = pd.read_csv(DATA)
        cats = df["Category"].apply(ast.literal_eval)
        counts = sorted(Counter(p.strip() for lst in cats for c in lst for p in c.split("|") if p.strip()).values(), reverse=True)
        self.assertEqual(tuple(counts), tuple(EMPIRICAL_COUNTS))
        self.assertEqual(len(counts), 176)
        self.assertEqual(sum(counts), 30901)


if __name__ == "__main__":
    unittest.main()
