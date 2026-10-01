"""Les comptes empiriques écrits en dur correspondent bien aux données du dépôt."""
import ast
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd

from simulation import EMPIRICAL_COUNTS, EMPIRICAL_WEEKLY_ADDITIONS

DATA = Path(__file__).resolve().parents[1] / "data" / "fandom_data.csv"


class TestEmpiricalCounts(unittest.TestCase):
    def test_hardcoded_counts_match_data(self):
        df = pd.read_csv(DATA)
        cats = df["Category"].apply(ast.literal_eval)
        counts = sorted(Counter(p.strip() for lst in cats for c in lst for p in c.split("|") if p.strip()).values(), reverse=True)
        self.assertEqual(tuple(counts), tuple(EMPIRICAL_COUNTS))
        self.assertEqual(len(counts), 176)
        self.assertEqual(sum(counts), 30901)


    def test_hardcoded_weekly_additions_match_data(self):
        df = pd.read_csv(DATA)
        d = df.assign(Category=df["Category"].apply(ast.literal_eval)).explode("Category").dropna(subset=["Category"])
        d = d.assign(Category=d["Category"].str.split("|")).explode("Category")
        d = d[d["Category"].str.strip() != ""]
        weekly = pd.Series(1, index=pd.to_datetime(d["date"])).sort_index().resample("W").sum()
        self.assertEqual(tuple(int(x) for x in weekly.to_numpy()), tuple(EMPIRICAL_WEEKLY_ADDITIONS))
        self.assertEqual(sum(EMPIRICAL_WEEKLY_ADDITIONS), 30901)


if __name__ == "__main__":
    unittest.main()
