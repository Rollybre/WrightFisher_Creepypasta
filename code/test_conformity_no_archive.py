"""
Suite des sections 10-12 (conformity_bias à archive_rate=100%) : ajoute la comparaison avec
`archive=False` (nouveau paramètre de `simulation.run_simulation`, cf. sa docstring) -- c'est-à-
dire AUCUN archivage cumulatif, seulement la population de la DERNIÈRE génération.

Motivation : confusion initiale entre "archive_rate=1.0 (100%)" et "pas d'archivage du tout".
Les deux sont très différents : à archive_rate=1.0, l'archive cumulative contient la somme de
TOUTES les générations (t_max générations x leur taille de population, ici ~6.6M individus, cf.
sections 10-12) ; sans archivage, on n'observe QUE la dernière génération (~8800 individus, la
taille de population finale). Ce script compare explicitement les deux conditions pour objectiver
cette différence, puis vérifie si l'effet de `conformity_bias` (établi sections 10-12 avec
archivage) survit aussi quand on n'archive pas du tout.

Usage :
    $PY code/test_conformity_no_archive.py --distribution uniform
    $PY code/test_conformity_no_archive.py --distribution power_law
    $PY code/test_conformity_no_archive.py --distribution random
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation import (
    run_simulation, compute_diversity_metrics, frequencies_from_archive,
    load_empirical_data, DATA_PATH,
)

ROOT = Path(__file__).parent

# Mêmes paramètres fixes que les sections 10-12, pour rester directement comparable.
N_CLASSES, N_INIT, N_FINAL, T_MAX = 176, 4400, 8800, 1000
CONFORMITY_BIASES = [0.9, 0.95, 1.0, 1.05, 1.1]
N_REPEATS = 8

# archive_rate n'a aucun effet quand archive=False (ignoré par run_simulation), gardé à 1.0
# pour la condition "archive=True" par cohérence avec la section 10 (sweep 3, 100%).
CONDITIONS = [
    ("archive=True (archive_rate=1.0, cumulatif)", True, 1.0),
    ("archive=False (population finale seule)", False, 1.0),
]

METRICS = ("gini", "hill_0", "hill_1", "hill_2", "hill_inf")


def run_sweep(archive, archive_rate, distribution, seed0):
    """Sweep conformity_bias pour une condition (archive=True/False) donnée. Retourne un
    DataFrame long (une ligne par répétition) avec conformity_bias/repeat/taille observée/métriques."""
    rows = []
    for cb in CONFORMITY_BIASES:
        for rep in range(N_REPEATS):
            rng = np.random.default_rng(seed0 + rep)
            result = run_simulation(
                rng, N_CLASSES, N_INIT, N_FINAL, T_MAX, archive_rate,
                conformity_bias=cb, distribution=distribution, archive=archive,
            )
            freq, _ = frequencies_from_archive(result, n_classes=N_CLASSES)
            rows.append({
                "conformity_bias": cb, "repeat": rep, "observed_size": len(result),
                **compute_diversity_metrics(freq),
            })
        sizes = [r["observed_size"] for r in rows if r["conformity_bias"] == cb]
        print(f"  conformity_bias={cb} : taille observée = {np.mean(sizes):.0f} "
              f"(min={min(sizes)}, max={max(sizes)})")
    return pd.DataFrame(rows)


def plot_conditions(summary_df, emp_metrics, distribution, output_path):
    """Une ligne de panneaux par condition (archive=True/False), une colonne par métrique."""
    condition_labels = [c[0] for c in CONDITIONS]
    fig, axes = plt.subplots(len(condition_labels), len(METRICS),
                              figsize=(4 * len(METRICS), 4 * len(condition_labels)), dpi=150)

    for row, label in enumerate(condition_labels):
        sub = summary_df[summary_df.condition == label]
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            ax.errorbar(sub["conformity_bias"], sub[f"{metric}_mean"], yerr=sub[f"{metric}_std"],
                        fmt="o-", color="#41ab5d", label="Simulation", capsize=3)
            if metric in emp_metrics:
                ax.axhline(emp_metrics[metric], color="#d73027", ls="--", label="Empirique")
            ax.set_xlabel("conformity_bias")
            ax.set_title(metric)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
        axes[row, 0].set_ylabel(label, fontsize=10)

    fig.suptitle(f"conformity_bias : archivage cumulatif (100%) vs pas d'archivage du tout "
                 f"-- distribution initiale={distribution}", y=1.01)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure sauvegardée : {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", choices=["power_law", "uniform", "random"], default="uniform",
                         help="Distribution initiale de la population à t=0 (défaut : uniform, "
                              "comme les sections 10-13 du notebook).")
    args = parser.parse_args()
    distribution = args.distribution

    output_png = ROOT.parent / "outputs" / f"conformity_archive_vs_no_archive_{distribution}.png"
    output_csv = ROOT.parent / "outputs" / f"conformity_archive_vs_no_archive_{distribution}.csv"

    print("Chargement des données empiriques (référence)...")
    _, _, emp_freq_all = load_empirical_data(DATA_PATH)
    emp_metrics = compute_diversity_metrics(emp_freq_all)
    print("Métriques empiriques :", {k: round(v, 3) for k, v in emp_metrics.items()})
    print(f"Distribution initiale : {distribution}")

    all_rows = []
    for i, (label, archive, archive_rate) in enumerate(CONDITIONS):
        print("\n" + "=" * 78)
        print(f"Sweep conformity_bias -- {label} -- distribution={distribution}")
        print("=" * 78)
        df = run_sweep(archive, archive_rate, distribution, seed0=8000 + i * 1000)
        df["condition"] = label
        all_rows.append(df)

    long_df = pd.concat(all_rows, ignore_index=True)

    summary_df = (
        long_df.groupby(["condition", "conformity_bias"])[list(METRICS) + ["observed_size"]]
        .agg(["mean", "std"])
    )
    summary_df.columns = [f"{metric}_{stat}" for metric, stat in summary_df.columns]
    summary_df = summary_df.reset_index()
    summary_df.to_csv(output_csv, index=False)
    print(f"\nSauvegardé : {output_csv}")

    for label, _, _ in CONDITIONS:
        print(f"\n--- {label} ---")
        print(summary_df[summary_df.condition == label].round(3).to_string(index=False))

    print("\n--- Taille observée (cumulative vs population finale), preuve que les deux "
          "conditions ne sont PAS équivalentes ---")
    for label, _, _ in CONDITIONS:
        sub = summary_df[summary_df.condition == label]
        print(f"  {label:>45} : taille moyenne = {sub['observed_size_mean'].mean():,.0f}")

    print("\n--- Amplitude (max-min) de chaque métrique par condition ---")
    for label, _, _ in CONDITIONS:
        sub = summary_df[summary_df.condition == label]
        for metric in METRICS:
            amp = sub[f"{metric}_mean"].max() - sub[f"{metric}_mean"].min()
            print(f"  {label:>45} / {metric:>8} : amplitude = {amp:8.3f}")

    plot_conditions(summary_df, emp_metrics, distribution, output_png)


if __name__ == "__main__":
    main()
