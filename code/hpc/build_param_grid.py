"""
Génère une table de paramètres (params.csv), une ligne = un job.

Pensé pour un job array HPC (SGE, ex. Myriad UCL) : chaque tâche de l'array lit
SA ligne (indexée par $SGE_TASK_ID) et lance un run de simulation.py avec ces
paramètres. Contrairement au sweep local (--sweep_param dans simulation.py, qui
réutilise un seul rng séquentiel en mémoire), ici chaque ligne = un process
indépendant sur le cluster, donc chaque seed est un run complètement séparé
(pas de problème de partage de rng entre workers, cf. discussion parallélisation).

Marche aussi (sans qsub/SGE) pour paralléliser sur une seule machine multi-cœurs
(ex. serveur les-lillas) via run_param_grid.py --n_jobs N -- cf. son usage.

Usage :
    python build_param_grid.py --archive_rates 0.01 0.05 0.1 0.2 0.4 --n_seeds 5
    -> écrit params.csv (5 seeds x 5 valeurs = 25 lignes)
    -> qsub -t 1-25 submit_array.sh                      # sur un cluster SGE (Myriad)
    -> python run_param_grid.py params.csv --n_jobs 24    # sur une machine seule (les-lillas)

    # Grille sur conformity_bias plutôt que archive_rate (archive_rates reste à sa valeur
    # par défaut, une seule valeur -> le produit cartésien ne porte que sur conformity_bias x seed) :
    python build_param_grid.py --conformity_biases 0.5 0.6 ... 1.0 --n_seeds 10

    # Grille 4D complète (archive_rate x conformity_bias x generations x archive) :
    python build_param_grid.py --conformity_biases 0.80 0.81 ... 1.20 \\
        --generations_values 500 600 700 800 900 1000 --archive_values true false --n_seeds 80
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
    parser.add_argument('--generations_values', type=int, nargs='+', default=None,
                         help="Valeurs de generations (t_max) à tester (défaut : None -> une seule "
                              "valeur, -T/--generations, pas de variation sur cet axe). Fournir "
                              "plusieurs valeurs (ex: 500 600 700 800 900 1000) pour en faire un axe "
                              "de la grille comme archive_rate/conformity_bias.")
    parser.add_argument('--archive_values', choices=['true', 'false'], nargs='+', default=['true'],
                         help="Valeurs de archive (cf. simulation.run_simulation : True = archivage "
                              "cumulatif historique, False = population de la dernière génération "
                              "seule) à tester (défaut : ['true'], pas de variation sur cet axe). "
                              "'false' ignore archive_rate pour les lignes correspondantes (gardé "
                              "dans la table pour la traçabilité, mais sans effet).")
    parser.add_argument('--n_seeds', type=int, default=1,
                         help="Nb de seeds indépendantes par combinaison (répétitions)")
    parser.add_argument('--seed_start', type=int, default=1,
                         help="Première seed utilisée (les suivantes s'incrémentent de 1)")
    parser.add_argument('--distribution', choices=['power_law', 'uniform', 'random', 'custom'],
                         default='power_law',
                         help="Forme de la distribution initiale, fixe pour toute la grille "
                              "(défaut : power_law, comportement historique). Cf. "
                              "simulation.initial_distribution pour le détail des 4 choix.")
    parser.add_argument('--distribution_path', default='',
                         help="Chemin d'un CSV de proportions/comptes personnalisés, requis avec "
                              "--distribution custom (même fichier pour toutes les lignes de la grille)")
    parser.add_argument('-o', '--output', default='params.csv')
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.distribution == "custom" and not args.distribution_path:
        raise SystemExit("--distribution custom nécessite --distribution_path")

    # Grille complète : produit cartésien archive_rate x conformity_bias x generations x archive x
    # seed. Avec les valeurs par défaut (une seule valeur sur un axe), ça revient à un sweep sur
    # les seuls axes effectivement fait varier -- même mécanisme, généralisé à 4 paramètres.
    # `distribution`/`distribution_path` ne varient pas dans la grille (répétés sur chaque ligne) :
    # ce sont des colonnes de configuration, pas un axe qu'on balaye ici.
    generations_values = args.generations_values or [args.generations]
    rows = []
    for archive_rate in args.archive_rates:
        for conformity_bias in args.conformity_biases:
            for generations in generations_values:
                for archive in args.archive_values:
                    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
                        rows.append({
                            "n_classes": args.n_classes,
                            "initial_pop": args.initial_pop,
                            "final_pop": args.final_pop,
                            "generations": generations,
                            "archive_rate": archive_rate,
                            "conformity_bias": conformity_bias,
                            "archive": archive,
                            "distribution": args.distribution,
                            "distribution_path": args.distribution_path,
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
