"""
Suite de test_archive_vs_conformity_hypothesis.py (section 10 du notebook,
sweep 3 : conformity_bias à archive_rate=1.0, l'archivage à 100%) : reprend
exactement ce sweep mais en le répétant pour les 3 distributions initiales
disponibles (`simulation.initial_distribution`) : power_law, uniform, random.

But : vérifier que l'effet massif de conformity_bias (confirmé sur
distribution uniform dans la section 10) ne dépend pas non plus de la forme
de la distribution initiale à t=0.

Pas de raréfaction ici (contrairement au script parent) : à archive_rate=1.0,
la taille d'archive est déterministe (somme des tailles de population sur
toutes les générations), donc identique pour toute valeur de conformity_bias
ET pour toute distribution -- comparer les métriques brutes est déjà
apples-to-apples.

Usage :
    $PY code/test_conformity_full_archive_by_distribution.py
"""

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
OUTPUT_PNG = ROOT.parent / "outputs" / "conformity_full_archive_by_distribution.png"
OUTPUT_CSV = ROOT.parent / "outputs" / "conformity_full_archive_by_distribution.csv"

# Mêmes paramètres fixes que la section 10 (sweep 3), pour rester directement comparable.
N_CLASSES, N_INIT, N_FINAL, T_MAX = 176, 4400, 8800, 1000
ARCHIVE_RATE = 1.0  # 100% : condition d'observation la plus favorable (aucune perte)
CONFORMITY_BIASES = [0.9, 0.95, 1.0, 1.05, 1.1]
DISTRIBUTIONS = ["power_law", "uniform", "random"]
N_REPEATS = 8

METRICS = ("gini", "hill_0", "hill_1", "hill_2", "hill_inf")


def run_sweep(distribution, seed0):
    """Sweep conformity_bias pour une distribution donnée. Retourne un DataFrame
    long (une ligne par répétition) avec distribution/conformity_bias/repeat/métriques."""
    rows = []
    for cb in CONFORMITY_BIASES:
        for rep in range(N_REPEATS):
            rng = np.random.default_rng(seed0 + rep)
            archive = run_simulation(
                rng, N_CLASSES, N_INIT, N_FINAL, T_MAX, ARCHIVE_RATE,
                conformity_bias=cb, distribution=distribution,
            )
            freq, _ = frequencies_from_archive(archive, n_classes=N_CLASSES)
            rows.append({
                "distribution": distribution, "conformity_bias": cb, "repeat": rep,
                "archive_size": len(archive),
                **compute_diversity_metrics(freq),
            })
        sizes = [r["archive_size"] for r in rows if r["conformity_bias"] == cb and r["distribution"] == distribution]
        print(f"  distribution={distribution} conformity_bias={cb} : "
              f"taille archive = {np.mean(sizes):.0f} (constante attendue)")
    return pd.DataFrame(rows)


def plot_by_distribution(summary_df, emp_metrics, output_path):
    """Une ligne de panneaux par distribution, une colonne par métrique -- barres d'erreur
    (écart-type entre répétitions) + ligne pointillée rouge = référence empirique."""
    fig, axes = plt.subplots(len(DISTRIBUTIONS), len(METRICS),
                              figsize=(4 * len(METRICS), 4 * len(DISTRIBUTIONS)), dpi=150)

    for row, distribution in enumerate(DISTRIBUTIONS):
        sub = summary_df[summary_df.distribution == distribution]
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
        axes[row, 0].set_ylabel(distribution, fontsize=11)

    fig.suptitle("conformity_bias à archive_rate=1.0 (100%), par distribution initiale", y=1.01)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure sauvegardée : {output_path}")


def main():
    print("Chargement des données empiriques (référence)...")
    _, _, emp_freq_all = load_empirical_data(DATA_PATH)
    emp_metrics = compute_diversity_metrics(emp_freq_all)
    print("Métriques empiriques :", {k: round(v, 3) for k, v in emp_metrics.items()})

    all_rows = []
    for i, distribution in enumerate(DISTRIBUTIONS):
        print("\n" + "=" * 78)
        print(f"Sweep conformity_bias (archive_rate=1.0) -- distribution={distribution}")
        print("=" * 78)
        df = run_sweep(distribution, seed0=4000 + i * 1000)
        all_rows.append(df)

    long_df = pd.concat(all_rows, ignore_index=True)

    summary_df = (
        long_df.groupby(["distribution", "conformity_bias"])[list(METRICS)]
        .agg(["mean", "std"])
    )
    summary_df.columns = [f"{metric}_{stat}" for metric, stat in summary_df.columns]
    summary_df = summary_df.reset_index()

    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSauvegardé : {OUTPUT_CSV}")

    for distribution in DISTRIBUTIONS:
        print(f"\n--- distribution={distribution} ---")
        print(summary_df[summary_df.distribution == distribution].round(3).to_string(index=False))

    print("\n--- Amplitude (max-min) de chaque métrique par distribution ---")
    for distribution in DISTRIBUTIONS:
        sub = summary_df[summary_df.distribution == distribution]
        for metric in METRICS:
            amp = sub[f"{metric}_mean"].max() - sub[f"{metric}_mean"].min()
            print(f"  {distribution:>10} / {metric:>8} : amplitude = {amp:8.3f}")

    plot_by_distribution(summary_df, emp_metrics, OUTPUT_PNG)


if __name__ == "__main__":
    main()
