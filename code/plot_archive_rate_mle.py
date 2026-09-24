"""
Compare plusieurs taux d'archivage (archive_rate) sur UN SEUL graphe rang-fréquence
(log-log), avec ajustement MLE de loi de puissance discrète pour chaque courbe.

Remplace le besoin historique de `powerlaw` (cf. code/legacy/analysis_stats.py, dépendance
volontairement abandonnée dans simulation.py, voir code/todo.txt) : l'estimateur MLE et le choix
de xmin par minimisation KS sont réimplémentés ici à la main (méthode Clauset, Shalizi & Newman
2009), sans dépendance externe autre que scipy (déjà utilisé ailleurs dans le projet).

Usage (depuis code/, avec l'env phd-env) :
    PY=".../phd-env/bin/python"
    $PY plot_archive_rate_mle.py \
        --archive_rates 0.01 0.05 0.1 0.2 0.4 \
        -o ../outputs/archive_rate_mle_comparison.png

    # Résolution empirique (n=176), lissé sur plusieurs répétitions :
    $PY plot_archive_rate_mle.py -N 176 -ni 4400 -nf 8800 --n_repeats 5 \
        --archive_rates 0.01 0.05 0.1 0.2 0.4 0.6
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.special import zeta as hurwitz_zeta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation import (
    run_simulation, frequencies_from_archive, load_empirical_data, DATA_PATH,
)

ROOT = Path(__file__).parent


# ============================================================================
# MLE LOI DE PUISSANCE DISCRÈTE (Clauset, Shalizi & Newman 2009), sans dépendance `powerlaw`
# ============================================================================
def _mle_alpha(x, xmin):
    """Estimateur MLE de l'exposant alpha pour une loi de puissance discrète tronquée à xmin
    (approximation continue standard, cf. eq. 3.7 de Clauset et al. 2009)."""
    n = len(x)
    return 1 + n / np.sum(np.log(x / (xmin - 0.5)))


def _ks_distance(x, alpha, xmin):
    """Distance de Kolmogorov-Smirnov entre la CDF complémentaire empirique (P(X>=x)) et celle
    du modèle ajusté (zeta de Hurwitz, normalisée à xmin)."""
    x_sorted = np.sort(x)
    n = len(x_sorted)
    ecdf_compl = np.arange(n, 0, -1) / n
    model_compl = hurwitz_zeta(alpha, x_sorted) / hurwitz_zeta(alpha, xmin)
    return np.max(np.abs(ecdf_compl - model_compl))


def fit_power_law_mle(freq, xmin_min=1, min_tail_size=5):
    """Ajuste une loi de puissance discrète sur un vecteur de fréquences (valeurs > 0), en
    choisissant xmin par minimisation de la distance KS (comme `powerlaw.Fit`), puis calcule
    l'exposant MLE et son écart-type asymptotique à ce xmin.

    xmin_min : plus petite valeur de xmin candidate (évite d'ajuster sur une queue de 1-2 points,
    non identifiable). min_tail_size : nb minimum de points au-dessus de xmin pour être candidat.

    Retourne un dict {alpha, sigma, xmin, n_tail, ks} ou None si aucun xmin candidat n'est valide
    (distribution trop courte/plate pour un fit fiable)."""
    freq = np.asarray(freq, dtype=float)
    freq = freq[freq > 0]
    candidates = np.unique(freq[freq >= xmin_min])

    best = None
    for xmin in candidates:
        tail = freq[freq >= xmin]
        if len(tail) < min_tail_size:
            continue
        alpha = _mle_alpha(tail, xmin)
        d = _ks_distance(tail, alpha, xmin)
        if best is None or d < best[0]:
            best = (d, alpha, xmin, len(tail))

    if best is None:
        return None
    d, alpha, xmin, n_tail = best
    sigma = (alpha - 1) / np.sqrt(n_tail)
    return {"alpha": alpha, "sigma": sigma, "xmin": xmin, "n_tail": n_tail, "ks": d}


# ============================================================================
# SIMULATION MOYENNÉE PAR TAUX D'ARCHIVAGE
# ============================================================================
def mean_frequencies_for_rate(rng, archive_rate, n_classes, n_init, n_final, t_max,
                               conformity_bias, n_repeats):
    """Lance n_repeats runs pour un archive_rate donné et retourne la distribution de fréquences
    moyenne (triée décroissant, classes absentes incluses), pour lisser le bruit d'un seul tirage
    avant l'ajustement MLE."""
    runs = []
    for _ in range(n_repeats):
        archive = run_simulation(rng, n_classes, n_init, n_final, t_max, archive_rate,
                                  conformity_bias=conformity_bias)
        freq, _ = frequencies_from_archive(archive, n_classes=n_classes)
        runs.append(freq)
    return np.mean(runs, axis=0)


def _add_fit_curve(ax, freq, color, xmin_min, lw=2, zorder=3):
    """Ajuste + trace une droite MLE (pointillé) sur la queue d'une distribution de fréquences
    déjà triée décroissant. Retourne le dict de fit (ou None si non identifiable)."""
    ranks = np.arange(1, len(freq) + 1)
    fit = fit_power_law_mle(freq, xmin_min=xmin_min)
    if fit is not None:
        tail_mask = freq >= fit["xmin"]
        tail_ranks, tail_freq = ranks[tail_mask], freq[tail_mask]
        if len(tail_ranks) >= 2:
            # Loi de puissance sur les fréquences (P(F>=f) ~ f^-(alpha-1)) <=> f(rang) ~ rang^-(1/(alpha-1))
            # d'où la pente -(alpha-1) tracée directement sur l'axe des rangs (cf. analysis_stats.py).
            fit_line = tail_freq[0] * (tail_ranks / tail_ranks[0]) ** -(fit["alpha"] - 1)
            ax.loglog(tail_ranks, fit_line, "--", color=color, lw=lw, zorder=zorder)
    return fit


# ============================================================================
# FIGURE : toutes les courbes (une par archive_rate) sur un seul graphe log-log,
# + courbe empirique de référence (données réelles, hors modèle)
# ============================================================================
def plot_archive_rate_comparison(rng, archive_rates, n_classes, n_init, n_final, t_max,
                                  conformity_bias, n_repeats, xmin_min, output_path,
                                  emp_freq_all=None):
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    # Palette qualitative (tab10) plutôt que viridis continu : des taux voisins (ex. 0.01 vs 0.02)
    # restaient visuellement confondus avec un dégradé séquentiel sur beaucoup de courbes.
    cmap = plt.get_cmap("tab10" if len(archive_rates) <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(len(archive_rates))]

    # ── Référence empirique (données réelles) ────────────────────────────────────────────────
    if emp_freq_all is not None:
        emp_freq = emp_freq_all[:n_classes]
        emp_ranks = np.arange(1, len(emp_freq) + 1)
        emp_fit = _add_fit_curve(ax, emp_freq, "black", xmin_min, lw=2.5, zorder=5)
        emp_label = "Données empiriques"
        if emp_fit is not None:
            emp_label += fr"  ($\hat\alpha$={emp_fit['alpha']:.2f}$\pm${emp_fit['sigma']:.2f})"
        ax.loglog(emp_ranks, emp_freq, "D", ms=6, color="black", mfc="none", mew=1.5,
                   label=emp_label, zorder=5)

    # ── Une courbe par taux d'archivage ───────────────────────────────────────────────────────
    for color, archive_rate in zip(colors, archive_rates):
        mean_freq = mean_frequencies_for_rate(
            rng, archive_rate, n_classes, n_init, n_final, t_max, conformity_bias, n_repeats
        )
        freq = mean_freq[mean_freq > 0]
        ranks = np.arange(1, len(freq) + 1)

        fit = _add_fit_curve(ax, freq, color, xmin_min)
        label = fr"$\alpha_{{arch}}$={archive_rate:g}"
        if fit is not None:
            label += fr"  ($\hat\alpha$={fit['alpha']:.2f}$\pm${fit['sigma']:.2f})"
        else:
            label += "  (fit MLE non identifiable)"
        ax.loglog(ranks, freq, "o", ms=5, alpha=0.8, color=color,
                   markeredgecolor="white", markeredgewidth=0.4, label=label, zorder=3)

    ax.set_xlabel("Rang", fontsize=12)
    ax.set_ylabel("Fréquence", fontsize=12)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
    repeats_note = f", moyenne sur {n_repeats} runs" if n_repeats > 1 else ""
    ax.set_title(
        f"Distributions rang-fréquence simulées selon le taux d'archivage\n"
        f"(ajustements MLE loi de puissance, $n$={n_classes}{repeats_note})",
        fontsize=12
    )
    # Légende à l'extérieur (à droite) : avec l'empirique + plusieurs taux, elle recouvrait
    # sinon la partie basse-droite du nuage de points (queue des distributions).
    ax.legend(fontsize=9, title="Taux d'archivage", loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure sauvegardée : {output_path}")


# ============================================================================
# CLI
# ============================================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Superpose plusieurs taux d'archivage sur un graphe rang-fréquence "
                     "log-log unique, avec ajustement MLE de loi de puissance par courbe."
    )
    parser.add_argument('-N', '--n_classes', type=int, default=40)
    parser.add_argument('-ni', '--initial_pop', type=int, default=1000)
    parser.add_argument('-nf', '--final_pop', type=int, default=2000)
    parser.add_argument('-T', '--generations', type=int, default=1000)
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('-q', '--conformity_bias', type=float, default=1)
    parser.add_argument('--archive_rates', type=float, nargs='+',
                         default=[0.01, 0.05, 0.1, 0.2, 0.4])
    parser.add_argument('--n_repeats', type=int, default=1,
                         help="Nb de runs moyennés par taux d'archivage (lisse le bruit avant le "
                              "fit MLE ; défaut 1 = un seul run par courbe)")
    parser.add_argument('--xmin_min', type=float, default=1,
                         help="Plus petite valeur de xmin candidate pour le fit MLE (évite "
                              "d'ajuster sur une queue de quelques points seulement)")
    parser.add_argument('--no_empirical', action='store_true',
                         help="N'affiche pas la courbe de référence empirique (par défaut, elle "
                              "est superposée aux courbes simulées, Top-n_classes catégories)")
    parser.add_argument('--empirical_path', default=str(DATA_PATH),
                         help="Chemin du CSV empirique (défaut : data/fandom_data.csv)")
    parser.add_argument('-o', '--output',
                         default=str(ROOT.parent / "outputs" / "archive_rate_mle_comparison.png"))
    return parser


def main():
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    emp_freq_all = None
    if not args.no_empirical:
        _, _, emp_freq_all = load_empirical_data(args.empirical_path)

    print(f"Taux d'archivage comparés : {args.archive_rates}")
    plot_archive_rate_comparison(
        rng, args.archive_rates, args.n_classes, args.initial_pop, args.final_pop,
        args.generations, args.conformity_bias, args.n_repeats, args.xmin_min, args.output,
        emp_freq_all=emp_freq_all
    )


if __name__ == '__main__':
    main()
