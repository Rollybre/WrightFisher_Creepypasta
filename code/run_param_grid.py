"""
Exécute localement une grille de paramètres définie dans un CSV — même format que celui généré
par build_param_grid.py et consommé par submit_array.sh sur le HPC. Une ligne = un run
indépendant, optionnellement en parallèle (--n_jobs).

Contrairement à --sweep_param dans simulation.py (qui ne fait varier qu'UN paramètre à la fois et
réutilise un seul rng séquentiel, donc pas parallélisable tel quel) :
- la grille peut mélanger n'importe quelle combinaison de colonnes (ex: plusieurs seeds par
  archive_rate, comme le génère build_param_grid.py --n_seeds) ;
- chaque ligne a son propre rng, seedé indépendamment (rng = np.random.default_rng(seed)) — donc
  chaque run est totalement indépendant des autres, ce qui le rend parallélisable (--n_jobs) et
  fait qu'un run de cette grille en local ou soumis sur le HPC (submit_array.sh, même ligne de
  params.csv) est strictement reproductible à l'identique.

Usage :
    python build_param_grid.py --archive_rates 0.01 0.02 ... --n_seeds 50 -o params.csv
    python run_param_grid.py params.csv -o results.csv --n_jobs 8 --plot sweep_fine.png
    # (la même params.csv peut aussi être soumise sur HPC via : qsub -t 1-N submit_array.sh)
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from simulation import (
    run_simulation, frequencies_from_archive, compute_diversity_metrics,
    load_empirical_data, plot_sweep_metrics, plot_grid_heatmap, DATA_PATH,
)


def _run_one_row(row_dict):
    """Lance UN run à partir d'une ligne de la grille (passée en dict, picklable pour
    multiprocessing). Fonction top-level exprès : ProcessPoolExecutor ne peut pas envoyer une
    closure/méthode à un worker, seulement une fonction importable par son nom de module."""
    rng = np.random.default_rng(int(row_dict["seed"]))

    # .get avec défaut 1.0 : reste compatible avec un params.csv généré avant l'ajout de
    # conformity_bias (colonne absente -> comportement neutre historique, inchangé)
    conformity_bias = float(row_dict.get("conformity_bias", 1.0))

    t0 = time.perf_counter()
    archive = run_simulation(
        rng, int(row_dict["n_classes"]), int(row_dict["initial_pop"]), int(row_dict["final_pop"]),
        int(row_dict["generations"]), float(row_dict["archive_rate"]),
        conformity_bias=conformity_bias
    )
    elapsed = time.perf_counter() - t0

    freq, _ = frequencies_from_archive(archive, n_classes=int(row_dict["n_classes"]))
    metrics = compute_diversity_metrics(freq)
    metrics["runtime_s"] = elapsed
    return {**row_dict, **metrics}


def run_grid(params_df, n_jobs=1):
    """Lance un run indépendant par ligne de params_df et retourne un DataFrame de résultats.
    n_jobs > 1 : répartit les lignes sur n_jobs processus (chaque ligne étant déjà indépendante
    -- son propre rng -- il n'y a rien à coordonner entre workers, contrairement au sweep local)."""
    row_dicts = [row.to_dict() for _, row in params_df.iterrows()]

    if n_jobs > 1:
        # as_completed (plutôt que executor.map) : la barre avance à chaque run terminé, dans
        # l'ordre où les workers finissent (pas l'ordre de soumission) -> progression en temps réel.
        results = [None] * len(row_dicts)
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_run_one_row, d): i for i, d in enumerate(row_dicts)}
            with tqdm(total=len(futures), desc="Runs", unit="run") as pbar:
                for future in as_completed(futures):
                    i = futures[future]
                    results[i] = future.result()
                    pbar.set_postfix(gini=f"{results[i]['gini']:.3f}")
                    pbar.update(1)
    else:
        results = []
        with tqdm(row_dicts, desc="Runs", unit="run") as pbar:
            for d in pbar:
                r = _run_one_row(d)
                results.append(r)
                pbar.set_postfix(gini=f"{r['gini']:.3f}")

    return pd.DataFrame(results)


def aggregate_for_plot(results_df, vary_param,
                        metric_cols=("gini", "hill_0", "hill_1", "hill_2", "hill_inf", "runtime_s")):
    """Regroupe les résultats bruts (plusieurs seeds par valeur de vary_param) en moyenne +
    écart-type, au même format que sweep_parameter() dans simulation.py, pour pouvoir réutiliser
    plot_sweep_metrics telle quelle."""
    grouped = results_df.groupby(vary_param)[list(metric_cols)]
    avg = grouped.mean().reset_index()
    std = grouped.std()
    for col in metric_cols:
        avg[f"{col}_std"] = std[col].values
    return avg


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Lance localement (optionnellement en parallèle) chaque ligne d'une grille "
                     "de paramètres (même format params.csv que build_param_grid.py / submit_array.sh)."
    )
    parser.add_argument("params_file", help="CSV généré par build_param_grid.py")
    parser.add_argument('-o', '--output', default="results.csv",
                         help="CSV de sortie, un run par ligne (défaut: results.csv)")
    parser.add_argument('--n_jobs', type=int, default=1,
                         help="Nb de processus en parallèle (défaut 1 = séquentiel)")
    parser.add_argument('--plot', default=None,
                         help="Chemin d'un .png : métriques moyennées (+ écart-type) vs "
                              "--vary_param, comparées à l'empirique")
    parser.add_argument('--vary_param', default='archive_rate',
                         help="Colonne de params.csv utilisée comme axe des x pour --plot "
                              "(défaut: archive_rate)")
    parser.add_argument('--vary_param2', default=None,
                         help="Deuxième colonne balayée (ex: conformity_bias) : si fournie, "
                              "--plot trace une heatmap 2D (--vary_param x --vary_param2) au lieu "
                              "du plot 1D standard -- pour un sweep sur deux paramètres croisés.")
    parser.add_argument('--data', default=str(DATA_PATH),
                         help="CSV empirique pour la comparaison du plot")
    return parser


def main():
    args = build_arg_parser().parse_args()
    params_df = pd.read_csv(args.params_file)
    print(f"{len(params_df)} runs à lancer depuis {args.params_file} (n_jobs={args.n_jobs})")

    t0 = time.perf_counter()
    results_df = run_grid(params_df, n_jobs=args.n_jobs)
    total_s = time.perf_counter() - t0

    results_df.to_csv(args.output, index=False)
    print(f"\n{len(results_df)} résultats sauvegardés dans {args.output} "
          f"({total_s:.1f}s au total, {total_s / len(results_df):.3f}s/run en moyenne, "
          f"n_jobs={args.n_jobs})")

    if args.plot:
        if args.vary_param2:
            plot_grid_heatmap(results_df, args.vary_param, args.vary_param2, args.plot)
        else:
            _, _, emp_freq_all = load_empirical_data(args.data)
            emp_metrics = compute_diversity_metrics(emp_freq_all)
            sweep_like_df = aggregate_for_plot(results_df, args.vary_param)
            plot_sweep_metrics(sweep_like_df, args.vary_param, emp_metrics, args.plot)


if __name__ == '__main__':
    main()
