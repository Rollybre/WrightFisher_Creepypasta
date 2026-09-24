"""
Teste l'hypothèse : "l'archivage (archive_rate) joue un rôle similaire au biais de
conformité (conformity_bias)".

Argument mécaniste (avant tout calcul) : dans `run_simulation`, l'archive ne réinjecte
JAMAIS rien dans la dynamique générationnelle -- chaque génération est entièrement
régénérée via `conform_probs` (dépend de `conformity_bias`), et l'archivage se contente
d'enregistrer un sous-échantillon SANS REMISE de la génération courante, sans influencer
la suivante. `archive_rate` devrait donc agir comme une simple PROFONDEUR D'ÉCHANTILLONNAGE
(comme la profondeur de séquençage en écologie), alors que `conformity_bias` agit DANS la
boucle de reproduction (un mécanisme de sélection/dérive).

Test : si `archive_rate` n'est qu'un effet de taille d'échantillon, alors RARÉFIER (sous-
échantillonner) toutes les archives à une taille commune avant de calculer les métriques de
diversité devrait faire DISPARAÎTRE son effet. L'effet de `conformity_bias`, lui, devrait
SURVIVRE à la raréfaction puisqu'il change la distribution sous-jacente elle-même, pas
seulement la profondeur d'observation.

Usage :
    $PY code/test_archive_vs_conformity_hypothesis.py
"""

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation import run_simulation, compute_diversity_metrics

ROOT = Path(__file__).parent
OUTPUT = ROOT.parent / "outputs" / "rarefaction_archive_vs_conformity.png"

# ── Paramètres fixes (résolution empirique, distribution uniforme) ─────────────────────────
N_CLASSES, N_INIT, N_FINAL, T_MAX = 176, 4400, 8800, 1000
DISTRIBUTION = "uniform"
N_REPEATS = 8
N_RAREFACTION_DRAWS = 30   # tirages de raréfaction par run, moyennés (réduit le bruit du sous-échantillonnage)

ARCHIVE_RATES = [0.02, 0.05, 0.1, 0.2, 0.4, 1.0]  # 1.0 = pas d'archivage partiel, toute génération
                                                   # entièrement enregistrée (conformity_bias fixé à 1.0, neutre)
CONFORMITY_BIASES = [0.9, 0.95, 1.0, 1.05, 1.1]   # testé à deux niveaux d'archivage (cf. FIXED_ARCHIVE_RATE / FULL_ARCHIVE_RATE)
FIXED_CONFORMITY_BIAS = 1.0
FIXED_ARCHIVE_RATE = 0.1
FULL_ARCHIVE_RATE = 1.0   # archivage à 100% (aucune perte) -- même sweep conformity_bias, pour voir si
                          # son effet tient toujours dans les conditions d'observation les plus favorables

METRICS = ("gini", "hill_0", "hill_1", "hill_2", "hill_inf")


def rarefy_metrics(archive, target_size, n_draws, rng):
    """Sous-échantillonne `archive` (array d'indices de classe, un par individu archivé) à
    `target_size` individus, `n_draws` fois sans remise, et retourne les métriques de
    diversité moyennées sur ces tirages (réduit le bruit d'un tirage de raréfaction unique)."""
    all_metrics = []
    for _ in range(n_draws):
        sample = rng.choice(archive, size=target_size, replace=False)
        counts = np.bincount(sample, minlength=N_CLASSES).astype(float)
        all_metrics.append(compute_diversity_metrics(counts))
    return {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}


def collect_archives(param_name, values, base_archive_rate, base_conformity_bias, seed0):
    """Lance N_REPEATS runs pour chaque valeur de `param_name` (archive_rate ou
    conformity_bias, l'autre restant fixe), et retourne {value: [archives (arrays), ...]}."""
    archives_by_value = {}
    for value in values:
        archive_rate = value if param_name == "archive_rate" else base_archive_rate
        conformity_bias = value if param_name == "conformity_bias" else base_conformity_bias
        archives = []
        for rep in range(N_REPEATS):
            rng = np.random.default_rng(seed0 + rep)
            archive = run_simulation(
                rng, N_CLASSES, N_INIT, N_FINAL, T_MAX, archive_rate,
                conformity_bias=conformity_bias, distribution=DISTRIBUTION
            )
            archives.append(archive)
        archives_by_value[value] = archives
        sizes = [len(a) for a in archives]
        print(f"  {param_name}={value} : taille archive = {np.mean(sizes):.0f} "
              f"(min={min(sizes)}, max={max(sizes)})")
    return archives_by_value


def raw_and_rarefied_metrics(archives_by_value, param_name, rarefaction_seed0):
    """Calcule les métriques BRUTES (sur l'archive complète) et RARÉFIÉES (toutes les archives
    sous-échantillonnées à la même taille = la plus petite archive rencontrée) pour chaque
    valeur, moyennées sur les répétitions. Retourne (raw_df, raref_df, target_size)."""
    import pandas as pd

    target_size = min(len(a) for archives in archives_by_value.values() for a in archives)
    print(f"  -> taille de raréfaction commune : {target_size}")

    raw_rows, raref_rows = [], []
    for value, archives in archives_by_value.items():
        for rep, archive in enumerate(archives):
            counts = np.bincount(archive, minlength=N_CLASSES).astype(float)
            raw_rows.append({param_name: value, "repeat": rep, **compute_diversity_metrics(counts)})

            rng = np.random.default_rng(rarefaction_seed0 + rep)
            raref_rows.append({
                param_name: value, "repeat": rep,
                **rarefy_metrics(archive, target_size, N_RAREFACTION_DRAWS, rng)
            })

    raw_df = pd.DataFrame(raw_rows).groupby(param_name)[list(METRICS)].mean().reset_index()
    raref_df = pd.DataFrame(raref_rows).groupby(param_name)[list(METRICS)].mean().reset_index()
    return raw_df, raref_df, target_size


def plot_comparison(results, output_path):
    """results : liste de (titre_ligne, param_name, raw_df, raref_df). Une ligne de panneaux
    par entrée, une colonne par métrique -- brut (plein) vs raréfié (pointillé) superposés."""
    fig, axes = plt.subplots(len(results), len(METRICS),
                              figsize=(4 * len(METRICS), 4 * len(results)), dpi=150)
    if len(results) == 1:
        axes = axes[None, :]

    for row, (title, param_name, raw_df, raref_df) in enumerate(results):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            ax.plot(raw_df[param_name], raw_df[metric], "o-", color="#2171b5", label="Brut")
            ax.plot(raref_df[param_name], raref_df[metric], "s--", color="#d73027",
                     label="Raréfié (taille commune)")
            ax.set_xlabel(param_name)
            ax.set_title(metric)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
        axes[row, 0].set_ylabel(title, fontsize=11)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure sauvegardée : {output_path}")


def main():
    print("=" * 78)
    print("Sweep 1 : archive_rate (conformity_bias fixé à 1.0)")
    print("=" * 78)
    archives_ar = collect_archives("archive_rate", ARCHIVE_RATES,
                                    FIXED_ARCHIVE_RATE, FIXED_CONFORMITY_BIAS, seed0=1000)
    raw_ar, raref_ar, target_ar = raw_and_rarefied_metrics(archives_ar, "archive_rate", rarefaction_seed0=5000)

    print("\n" + "=" * 78)
    print("Sweep 2 : conformity_bias (archive_rate fixé à 0.1)")
    print("=" * 78)
    archives_cb = collect_archives("conformity_bias", CONFORMITY_BIASES,
                                    FIXED_ARCHIVE_RATE, FIXED_CONFORMITY_BIAS, seed0=2000)
    raw_cb, raref_cb, target_cb = raw_and_rarefied_metrics(archives_cb, "conformity_bias", rarefaction_seed0=6000)

    print("\n" + "=" * 78)
    print(f"Sweep 3 : conformity_bias (archive_rate fixé à {FULL_ARCHIVE_RATE} = archivage à 100%)")
    print("=" * 78)
    archives_cb_full = collect_archives("conformity_bias", CONFORMITY_BIASES,
                                         FULL_ARCHIVE_RATE, FIXED_CONFORMITY_BIAS, seed0=3000)
    raw_cb_full, raref_cb_full, target_cb_full = raw_and_rarefied_metrics(
        archives_cb_full, "conformity_bias", rarefaction_seed0=7000
    )

    print("\n--- archive_rate : brut ---")
    print(raw_ar.round(3).to_string(index=False))
    print("--- archive_rate : raréfié ---")
    print(raref_ar.round(3).to_string(index=False))

    print("\n--- conformity_bias (archive_rate=0.1) : brut ---")
    print(raw_cb.round(3).to_string(index=False))
    print("--- conformity_bias (archive_rate=0.1) : raréfié ---")
    print(raref_cb.round(3).to_string(index=False))

    print("\n--- conformity_bias (archive_rate=1.0, 100%) : brut ---")
    print(raw_cb_full.round(3).to_string(index=False))
    print("--- conformity_bias (archive_rate=1.0, 100%) : raréfié ---")
    print(raref_cb_full.round(3).to_string(index=False))

    # Amplitude de variation (max-min) sur chaque métrique, brut vs raréfié, pour quantifier
    # si la raréfaction "aplatit" bien l'effet de archive_rate (et pas celui de conformity_bias)
    print("\n--- Amplitude (max-min) des métriques selon le sweep ---")
    for label, raw_df, raref_df, pname in [
        ("archive_rate", raw_ar, raref_ar, "archive_rate"),
        ("conformity_bias (arch=0.1)", raw_cb, raref_cb, "conformity_bias"),
        ("conformity_bias (arch=1.0)", raw_cb_full, raref_cb_full, "conformity_bias"),
    ]:
        for metric in METRICS:
            amp_raw = raw_df[metric].max() - raw_df[metric].min()
            amp_raref = raref_df[metric].max() - raref_df[metric].min()
            reduction = 100 * (1 - amp_raref / amp_raw) if amp_raw > 0 else float("nan")
            print(f"  {label:>28} / {metric:>8} : amplitude brute={amp_raw:8.2f}  "
                  f"raréfiée={amp_raref:8.2f}  (réduction {reduction:5.1f}%)")

    # Comparaison directe conformity_bias à archive_rate=0.1 vs archive_rate=1.0 : si conformity_bias
    # joue vraiment un rôle indépendant de l'archivage, les deux courbes doivent être quasi identiques
    print("\n--- conformity_bias : effet à archive_rate=0.1 vs archive_rate=1.0 (brut) ---")
    comp = raw_cb[["conformity_bias"] + list(METRICS)].merge(
        raw_cb_full[["conformity_bias"] + list(METRICS)], on="conformity_bias",
        suffixes=("_arch0.1", "_arch1.0")
    )
    print(comp.round(3).to_string(index=False))

    plot_comparison(
        [("archive_rate\n(conformity_bias=1.0)", "archive_rate", raw_ar, raref_ar),
         ("conformity_bias\n(archive_rate=0.1)", "conformity_bias", raw_cb, raref_cb),
         ("conformity_bias\n(archive_rate=1.0, 100%)", "conformity_bias", raw_cb_full, raref_cb_full)],
        OUTPUT
    )


if __name__ == "__main__":
    main()
