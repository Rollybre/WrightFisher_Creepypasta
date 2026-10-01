"""
Recalcule les comptes empiriques par catégorie (corpus fandom) à partir de data/fandom_data.csv, et les
compare à `simulation.EMPIRICAL_COUNTS` (valeurs écrites en dur dans le code).

Chaque ligne du CSV est une histoire dont la colonne `Category` est une chaîne représentant une liste
Python ; les catégories combinées "A|B" sont éclatées en catégories individuelles. Le compte d'une
catégorie est son nombre d'occurrences, toutes histoires confondues.

Usage (depuis la racine du repo) :
    python scripts/empirical_counts.py                       # data/fandom_data.csv
    python scripts/empirical_counts.py --data autre.csv
"""

import argparse
import ast
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation import EMPIRICAL_COUNTS, EMPIRICAL_WEEKLY_ADDITIONS


def empirical_counts(path):
    """Comptes par catégorie, triés par ordre décroissant (liste d'entiers)."""
    df = pd.read_csv(path)
    cats = df["Category"].apply(ast.literal_eval)
    counter = Counter(p.strip() for lst in cats for c in lst for p in c.split("|") if p.strip())
    return sorted(counter.values(), reverse=True)


def weekly_additions(path):
    """Occurrences de catégories déposées par semaine (périodes vides = 0) : profil de croissance empirique."""
    df = pd.read_csv(path)
    d = df.assign(Category=df["Category"].apply(ast.literal_eval)).explode("Category").dropna(subset=["Category"])
    d = d.assign(Category=d["Category"].str.split("|")).explode("Category")
    d = d[d["Category"].str.strip() != ""]
    return [int(x) for x in pd.Series(1, index=pd.to_datetime(d["date"])).sort_index().resample("W").sum().to_numpy()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data" / "fandom_data.csv"))
    args = parser.parse_args()
    counts = empirical_counts(args.data)
    print(f"{len(counts)} catégories, {sum(counts)} occurrences ; plus fréquente : {counts[0]}")
    same = tuple(counts) == tuple(EMPIRICAL_COUNTS)
    print("identique à simulation.EMPIRICAL_COUNTS :", same)
    if not same:
        print("nouvelles valeurs :", tuple(counts))
    weekly = weekly_additions(args.data)
    same_w = tuple(weekly) == tuple(EMPIRICAL_WEEKLY_ADDITIONS)
    print(f"{len(weekly)} semaines, {sum(weekly)} occurrences ; identique à simulation.EMPIRICAL_WEEKLY_ADDITIONS : {same_w}")
    if not same_w:
        print("nouvelles valeurs :", tuple(weekly))
    return 0 if (same and same_w) else 1


if __name__ == "__main__":
    raise SystemExit(main())
