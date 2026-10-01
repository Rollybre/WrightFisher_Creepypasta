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
from simulation import EMPIRICAL_COUNTS


def empirical_counts(path):
    """Comptes par catégorie, triés par ordre décroissant (liste d'entiers)."""
    df = pd.read_csv(path)
    cats = df["Category"].apply(ast.literal_eval)
    counter = Counter(p.strip() for lst in cats for c in lst for p in c.split("|") if p.strip())
    return sorted(counter.values(), reverse=True)


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
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
