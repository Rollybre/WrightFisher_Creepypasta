"""
Génère une table de paramètres (params.csv) par tirage aléatoire indépendant sur chaque axe,
plutôt que par produit cartésien (cf. build_param_grid.py). Une ligne = une seule combinaison
(archive_rate, conformity_bias, generations) tirée uniformément dans son intervalle, pas un
croisement de toutes les valeurs entre elles -- donc --n_samples 1_000_000 donne exactement
1 000 000 lignes/jobs, pas 1 000 000^3.

Même format de sortie que build_param_grid.py (mêmes colonnes) -> compatible tel quel avec
run_param_grid.py et submit_array.sh, aucun changement côté exécution des jobs.

Usage (depuis code/hpc/) :
    python build_param_grid_random.py --n_samples 1000000 -o params_random.csv
    -> qsub -t 1-1000000 submit_array.sh          # sur un cluster SGE (Myriad)
    -> python run_param_grid.py params_random.csv --n_jobs 24   # sur une machine seule

    # Bornes personnalisées (les défauts ci-dessous sont déjà ceux-là) :
    python build_param_grid_random.py --n_samples 500000 \\
        --conformity_bias_range 0.8 1.2 --archive_rate_range 0.0 1.0 \\
        --generations_range 300 1000
"""

import argparse
import csv

import numpy as np


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Génère params.csv : une ligne par tirage aléatoire indépendant de "
                     "(archive_rate, conformity_bias, generations), pas un produit cartésien."
    )
    parser.add_argument('-N', '--n_classes', type=int, default=176)
    parser.add_argument('-ni', '--initial_pop', type=int, default=4400)
    parser.add_argument('-nf', '--final_pop', type=int, default=8800)
    parser.add_argument('--n_samples', type=int, default=1_000_000,
                         help="Nb de lignes/jobs à générer (= nb de tirages, un par ligne)")

    parser.add_argument('--conformity_bias_range', type=float, nargs=2, default=[0.8, 1.2],
                         metavar=('LOW', 'HIGH'),
                         help="Borne (incluse) du tirage uniforme continu pour conformity_bias "
                              "(défaut : 0.8 1.2)")
    parser.add_argument('--archive_rate_range', type=float, nargs=2, default=[0.0, 1.0],
                         metavar=('LOW', 'HIGH'),
                         help="Borne (incluse) du tirage uniforme continu pour archive_rate "
                              "(défaut : 0.0 1.0 -- cf. avertissement archive_rate=0 dans le "
                              "docstring de run_simulation : un archive_rate trop proche de 0 "
                              "peut laisser l'archive cumulative totalement vide et faire planter "
                              "hill_number en aval ; en tirage continu la probabilité de tomber "
                              "exactement sur 0.0 est nulle, mais les toutes petites valeurs "
                              "restent théoriquement à risque si generations est bas)")
    parser.add_argument('--generations_range', type=int, nargs=2, default=[300, 1000],
                         metavar=('LOW', 'HIGH'),
                         help="Borne (incluse des deux côtés) du tirage uniforme entier pour "
                              "generations (défaut : 300 1000)")

    parser.add_argument('--archive', choices=['true', 'false'], default='true',
                         help="Fixe (pas d'axe) : 'true' = archivage cumulatif historique "
                              "(défaut), 'false' ignore archive_rate")
    parser.add_argument('--distribution', choices=['power_law', 'uniform', 'random', 'custom'],
                         default='uniform',
                         help="Forme de la distribution initiale, fixe pour toutes les lignes "
                              "(défaut : uniform, comme la grille 5M)")
    parser.add_argument('--distribution_path', default='',
                         help="Requis avec --distribution custom (même fichier pour toutes les lignes)")

    parser.add_argument('--seed_start', type=int, default=1,
                         help="Seed de simulation de la 1ere ligne, incrémentée de 1 par ligne "
                              "(seed de simulation.run_simulation, distincte du rng ci-dessous "
                              "qui ne sert qu'à tirer les paramètres eux-mêmes)")
    parser.add_argument('--rng_seed', type=int, default=0,
                         help="Seed du générateur qui tire les paramètres -> reproductibilité de "
                              "la table elle-même (deux appels avec le même --rng_seed et les "
                              "mêmes bornes/n_samples donnent exactement le même params.csv)")

    parser.add_argument('-o', '--output', default='params_random.csv')
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.distribution == "custom" and not args.distribution_path:
        raise SystemExit("--distribution custom nécessite --distribution_path")

    rng = np.random.default_rng(args.rng_seed)
    n = args.n_samples

    # Tirage indépendant par axe (pas de produit cartésien) : chaque ligne i combine
    # conformity_biases[i], archive_rates[i], generations_values[i] tirés séparément -- même
    # principe qu'un échantillonnage Monte Carlo / LHS naïf sur l'espace des paramètres.
    conformity_biases = rng.uniform(*args.conformity_bias_range, size=n)
    archive_rates = rng.uniform(*args.archive_rate_range, size=n)
    # endpoint=True : borne haute incluse (contrairement à np.random.randint), comme les deux
    # autres tirages continus ci-dessus
    generations_values = rng.integers(
        args.generations_range[0], args.generations_range[1], size=n, endpoint=True
    )

    with open(args.output, "w", newline="") as f:
        fieldnames = [
            "n_classes", "initial_pop", "final_pop", "generations", "archive_rate",
            "conformity_bias", "archive", "distribution", "distribution_path", "seed",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            writer.writerow({
                "n_classes": args.n_classes,
                "initial_pop": args.initial_pop,
                "final_pop": args.final_pop,
                "generations": int(generations_values[i]),
                "archive_rate": float(archive_rates[i]),
                "conformity_bias": float(conformity_biases[i]),
                "archive": args.archive,
                "distribution": args.distribution,
                "distribution_path": args.distribution_path,
                "seed": args.seed_start + i,
            })

    print(f"{n} jobs écrits dans {args.output}")
    print(f"  conformity_bias ~ U{tuple(args.conformity_bias_range)}")
    print(f"  archive_rate    ~ U{tuple(args.archive_rate_range)}")
    print(f"  generations     ~ U{{{args.generations_range[0]}..{args.generations_range[1]}}} (entier)")
    print(f"-> soumettre avec : qsub -t 1-{n} submit_array.sh")


if __name__ == '__main__':
    main()
