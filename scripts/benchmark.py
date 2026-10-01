"""
Benchmark de scaling de simulation.py : mesure le temps d'exécution de la même table de
simulations pour plusieurs nombres de processus (n_jobs), jusqu'à 24 par défaut.

Pour chaque n_jobs : temps total (mur), temps par simulation, speedup et efficacité relatifs à
n_jobs=1, puis extrapolation du temps pour --extrapolate simulations (ex. la grille finale).
La table est générée une seule fois (mêmes paramètres et mêmes graines pour tous les n_jobs) ;
les résultats de chaque configuration sont comparés à ceux de n_jobs=1 pour vérifier que le
parallélisme ne change rien.

Usage :
    python scripts/benchmark.py                               # 1000 sims, n_jobs 1 2 4 8 12 16 24
    python scripts/benchmark.py --n_sims 2000 --repeats 3 --jobs 1 8 24 -o bench.csv
Note : sur une machine avec moins de cœurs que le n_jobs testé, les mesures au-delà de
os.cpu_count() sont sur-souscrites (le speedup plafonne) ; à lancer sur le serveur cible.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # racine du repo
from simulation import generate_param_table, run_param_table


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Benchmark du scaling de simulation.py selon n_jobs.")
    parser.add_argument("--n_sims", type=int, default=1000, help="nombre de simulations par mesure (défaut 1000)")
    parser.add_argument("--jobs", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 24],
                        help="valeurs de n_jobs à tester (défaut 1 2 4 8 12 16 24)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="répétitions par n_jobs ; on retient le meilleur temps (défaut 1)")
    parser.add_argument("--rng_seed", type=int, default=0, help="graine du tirage des paramètres")
    parser.add_argument("--extrapolate", type=int, default=1_000_000,
                        help="nb de simulations pour l'estimation du temps total (défaut 1 000 000)")
    parser.add_argument("-o", "--output", default=None, help="CSV de sortie (optionnel)")
    return parser


def fmt_duration(seconds):
    """Formate une durée en s / min / h / j selon l'ordre de grandeur."""
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} j"


def main():
    args = build_arg_parser().parse_args()
    jobs = sorted(set(args.jobs))
    if jobs[0] < 1:
        raise SystemExit("--jobs : valeurs >= 1 uniquement")
    if 1 not in jobs:
        jobs = [1] + jobs  # référence pour speedup/efficacité
    n_cpu = os.cpu_count() or 1
    print(f"{args.n_sims} simulations, {args.repeats} répétition(s), cœurs détectés : {n_cpu}")
    if jobs[-1] > n_cpu:
        print(f"ATTENTION : n_jobs jusqu'à {jobs[-1]} > {n_cpu} cœurs -> mesures sur-souscrites au-delà de {n_cpu}")

    rows = generate_param_table(args.n_sims, rng_seed=args.rng_seed)
    reference = None
    records = []
    for n_jobs in jobs:
        best = float("inf")
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            results = run_param_table(rows, n_jobs=n_jobs)
            best = min(best, time.perf_counter() - t0)
        if reference is None:
            reference = results
        identical = results == reference
        records.append({"n_jobs": n_jobs, "total_s": best, "ms_per_sim": 1000 * best / args.n_sims,
                        "identical_to_1": identical})
        print(f"  n_jobs={n_jobs:>3} : {best:8.2f} s ({records[-1]['ms_per_sim']:.2f} ms/sim)"
              f"{'' if identical else '  /!\\ résultats différents de n_jobs=1'}", flush=True)

    t1 = records[0]["total_s"]
    for r in records:
        r["speedup"] = t1 / r["total_s"]
        r["efficiency"] = r["speedup"] / r["n_jobs"]
        r["extrapolated_s"] = r["total_s"] * args.extrapolate / args.n_sims

    print(f"\n{'n_jobs':>6} {'total (s)':>10} {'ms/sim':>8} {'speedup':>8} {'effic.':>7} {'CPU-ms/sim':>11}"
          f" {'-> ' + format(args.extrapolate, ',').replace(',', ' ') + ' sims':>22}")
    for r in records:
        print(f"{r['n_jobs']:>6} {r['total_s']:>10.2f} {r['ms_per_sim']:>8.2f} {r['speedup']:>8.2f} "
              f"{r['efficiency']:>6.0%} {r['ms_per_sim'] * r['n_jobs']:>11.1f} {fmt_duration(r['extrapolated_s']):>22}")
    print("(CPU-ms/sim = ms/sim x n_jobs : coût CPU réel par simulation, augmente si les cœurs se gênent)")

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"résultats écrits dans {args.output}")


if __name__ == "__main__":
    main()
