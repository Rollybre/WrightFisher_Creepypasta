"""
Génère une table de paramètres (params.csv), une ligne = un job.

Pensé pour un job array HPC (SGE, ex. Myriad UCL) : chaque tâche de l'array lit
SA ligne (indexée par $SGE_TASK_ID) et lance un run de simulation.py avec ces
paramètres. Contrairement au sweep local (--sweep_param dans simulation.py, qui
réutilise un seul rng séquentiel en mémoire), ici chaque ligne = un process
indépendant sur le cluster, donc chaque seed est un run complètement séparé
(pas de problème de partage de rng entre workers, cf. discussion parallélisation).

Usage :
    python build_param_grid.py --archive_rates 0.01 0.05 0.1 0.2 0.4 --n_seeds 5
    -> écrit params.csv (5 seeds x 5 valeurs = 25 lignes)
    -> qsub -t 1-25 submit_array.sh

    # Grille sur conformity_bias plutôt que archive_rate (archive_rates reste à sa valeur
    # par défaut, une seule valeur -> le produit cartésien ne porte que sur conformity_bias x seed) :
    python build_param_grid.py --conformity_biases 0.5 0.6 ... 1.0 --n_seeds 10
"""

import argparse
import csv


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Génère params.csv : une ligne par combinaison (archive_rate x "
                     "conformity_bias x seed) à tester."
    )
    parser.add_argument('-N', '--n_classes', type=int, default=176)
    parser.add_argument('-ni', '--initial_pop', type=int, default=4000)
    parser.add_argument('-nf', '--final_pop', type=int, default=10000)
    parser.add_argument('-T', '--generations', type=int, default=1000)
    parser.add_argument('--archive_rates', type=float, nargs='+', default=[0.05],
                         help="Valeurs de archive_rate à tester (défaut : [0.05], une seule "
                              "valeur -> pas de variation sur cet axe)")
    parser.add_argument('--conformity_biases', type=float, nargs='+', default=[1.0],
                         help="Valeurs de conformity_bias à tester (défaut : [1.0] = neutre, "
                              "comportement historique -> pas de variation sur cet axe)")
    parser.add_argument('--n_seeds', type=int, default=1,
                         help="Nb de seeds indépendantes par combinaison (répétitions)")
    parser.add_argument('--seed_start', type=int, default=1,
                         help="Première seed utilisée (les suivantes s'incrémentent de 1)")
    parser.add_argument('-o', '--output', default='params.csv')
    return parser


def main():
    args = build_arg_parser().parse_args()

    # Grille complète : produit cartésien archive_rate x conformity_bias x seed. Avec les valeurs
    # par défaut (une seule archive_rate ou une seule conformity_bias), ça revient à un sweep 1D
    # sur l'axe qu'on a effectivement fait varier -- même mécanisme, généralisé aux deux paramètres.
    rows = []
    for archive_rate in args.archive_rates:
        for conformity_bias in args.conformity_biases:
            for seed in range(args.seed_start, args.seed_start + args.n_seeds):
                rows.append({
                    "n_classes": args.n_classes,
                    "initial_pop": args.initial_pop,
                    "final_pop": args.final_pop,
                    "generations": args.generations,
                    "archive_rate": archive_rate,
                    "conformity_bias": conformity_bias,
                    "seed": seed,
                })

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} jobs écrits dans {args.output}")
    print(f"-> soumettre avec : qsub -t 1-{len(rows)} submit_array.sh")


if __name__ == '__main__':
    main()
