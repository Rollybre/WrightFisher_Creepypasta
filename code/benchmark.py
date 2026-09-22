"""
Benchmark à lancer sur une nouvelle machine (ex. serveur les-lillas) avant de lancer une grosse
grille pour de vrai : mesure (1) le temps d'un run seul à l'échelle papier, et (2) le gain réel de
--n_jobs sur run_param_grid.py, POUR CETTE MACHINE (nb de cœurs et charge peuvent différer de la
machine de dev).

Usage :
    python benchmark.py
    python benchmark.py --runs_per_job 20 --max_jobs 16
"""

import argparse
import os
import platform
import time

import numpy as np
import pandas as pd

from simulation import run_simulation, DATA_PATH
from run_param_grid import run_grid


# ============================================================================
# 1. TEMPS D'UN RUN SEUL
# ============================================================================
def benchmark_single_run(n_classes=176, initial_pop=4400, final_pop=8800, generations=1000,
                          archive_rate=0.2, n_repeats=5):
    """Chronomètre n_repeats runs identiques (mêmes paramètres, rng réutilisé) et retourne
    (temps moyen, écart-type) en secondes."""
    rng = np.random.default_rng(0)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        run_simulation(rng, n_classes, initial_pop, final_pop, generations, archive_rate)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


# ============================================================================
# 2. GAIN DE --n_jobs (charge de travail fixe, n_jobs variable)
# ============================================================================
def benchmark_parallel_scaling(runs_per_job=20, max_jobs=None, n_classes=176, initial_pop=4400,
                                final_pop=8800, generations=1000):
    """Construit une grille de taille max_jobs x runs_per_job (charge FIXE, donc comparable
    d'un n_jobs à l'autre) et mesure le temps total pour n_jobs = 1, 2, 4, ... jusqu'à max_jobs.
    archive_rate varie sur la grille (0.05 à 0.5) pour représenter un coût de run réaliste, pas
    un seul point trop optimiste/pessimiste."""
    max_jobs = max_jobs or os.cpu_count()
    total_runs = max_jobs * runs_per_job

    params_df = pd.DataFrame({
        "n_classes": n_classes, "initial_pop": initial_pop, "final_pop": final_pop,
        "generations": generations,
        "archive_rate": np.round(np.linspace(0.05, 0.5, total_runs), 4),
        "seed": np.arange(1, total_runs + 1),
    })

    # Points testés : 1, puis puissances de 2 et paliers usuels jusqu'à max_jobs (inclus)
    candidate_jobs = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    job_counts = sorted(set(j for j in candidate_jobs if j <= max_jobs) | {max_jobs})

    print(f"\nGrille de test : {total_runs} runs (fixe), n_jobs testés : {job_counts}")
    rows = []
    baseline = None
    for n_jobs in job_counts:
        t0 = time.perf_counter()
        run_grid(params_df, n_jobs=n_jobs)
        elapsed = time.perf_counter() - t0
        baseline = elapsed if baseline is None else baseline
        speedup = baseline / elapsed
        rows.append({
            "n_jobs": n_jobs, "total_s": elapsed, "s_per_run": elapsed / total_runs,
            "runs_per_s": total_runs / elapsed, "speedup_vs_n_jobs_1": speedup,
        })
        print(f"  n_jobs={n_jobs:>3} : {elapsed:7.1f}s  ({elapsed / total_runs:.3f}s/run, "
              f"{total_runs / elapsed:5.1f} runs/s, x{speedup:.2f} vs séquentiel)")
    return pd.DataFrame(rows)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Benchmark local : coût d'un run + gain de --n_jobs.")
    parser.add_argument('--runs_per_job', type=int, default=20,
                         help="Nb de runs par worker testé (charge totale = max_jobs x ce nombre)")
    parser.add_argument('--max_jobs', type=int, default=None,
                         help="Nb max de processus testé (défaut : nb de cœurs de la machine)")
    parser.add_argument('-N', '--n_classes', type=int, default=176)
    parser.add_argument('-ni', '--initial_pop', type=int, default=4400)
    parser.add_argument('-nf', '--final_pop', type=int, default=8800)
    parser.add_argument('-T', '--generations', type=int, default=1000)
    parser.add_argument('-o', '--output', default=None,
                         help="CSV où sauvegarder le tableau de scaling (optionnel)")
    return parser


def main():
    args = build_arg_parser().parse_args()

    print(f"Machine : {platform.node()} ({platform.system()} {platform.machine()})")
    print(f"Cœurs détectés : {os.cpu_count()}")
    print(f"Données empiriques : {DATA_PATH}")

    print("\n" + "=" * 60)
    print("1. TEMPS D'UN RUN SEUL (échelle papier, n=176, T=1000)")
    print("=" * 60)
    mean_s, std_s = benchmark_single_run(
        n_classes=args.n_classes, initial_pop=args.initial_pop, final_pop=args.final_pop,
        generations=args.generations
    )
    print(f"  {mean_s:.4f}s ± {std_s:.4f}s (moyenne sur 5 runs)")

    print("\n" + "=" * 60)
    print("2. GAIN DE --n_jobs (run_param_grid.py)")
    print("=" * 60)
    scaling_df = benchmark_parallel_scaling(
        runs_per_job=args.runs_per_job, max_jobs=args.max_jobs,
        n_classes=args.n_classes, initial_pop=args.initial_pop, final_pop=args.final_pop,
        generations=args.generations
    )

    best = scaling_df.loc[scaling_df["runs_per_s"].idxmax()]
    print(f"\nMeilleur débit observé : n_jobs={int(best['n_jobs'])} "
          f"({best['runs_per_s']:.1f} runs/s, x{best['speedup_vs_n_jobs_1']:.2f} vs séquentiel)")

    if args.output:
        scaling_df.to_csv(args.output, index=False)
        print(f"Tableau de scaling sauvegardé : {args.output}")


if __name__ == '__main__':
    main()
