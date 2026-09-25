Code, data, and article source for the paper on fandom data and Wright-Fisher models.

**Point d'entrée** : `code/archive_rate_conformity_exploration.ipynb` — journal de bord narratif (texte + figures, pas de cellules exécutables) qui documente chaque script lancé et son interprétation, section par section, jusqu'à la conclusion actuelle.

## Structure

```
.
├── code/
│   ├── simulation.py                                  # Modèle Wright-Fisher à archivage cumulatif (run_simulation,
│   │                                                   #   compute_diversity_metrics, sweeps, CLI) -- coeur du projet
│   ├── plot_archive_rate_mle.py                        # Fits MLE de loi de puissance (Clauset-Shalizi-Newman) par archive_rate
│   ├── test_archive_vs_conformity_hypothesis.py        # Test par raréfaction : archive_rate = profondeur d'échantillonnage ?
│   ├── test_conformity_full_archive_by_distribution.py # Même test, conformity_bias à archive_rate=100%, x3 distributions initiales
│   ├── test_conformity_no_archive.py                   # archive_rate=100% (cumulatif) vs archive=False (population finale seule)
│   ├── archive_rate_conformity_exploration.ipynb       # Journal de bord (voir ci-dessus)
│   ├── todo.txt                                        # Pistes ouvertes (conformité pondérée, croissance de population, innovation)
│   ├── hpc/                                            # Grilles de paramètres + exécution (local ou cluster distant)
│   │   ├── build_param_grid.py / run_param_grid.py / submit_array.sh
│   │   └── params_*.csv / results_*.csv                # Sorties de sweep (non versionnées, cf. .gitignore)
│   ├── docs/                                           # Notes de référence (métriques de diversité, cheatsheet simulation)
│   ├── notebook_figures/                               # Snapshot figé des figures citées par le notebook (versionné, pour GitHub)
│   └── legacy/                                         # Prototypes/scripts supersedés (wf.ipynb, analysis_stats.py, test.ipynb, ...)
├── data/
│   ├── fandom_data.csv     # Dataset empirique de référence (176 catégories)
│   └── fandom_links.csv    # Liens entre fandoms
├── outputs/                # Sorties reproductibles depuis code/ (non versionnées, cf. .gitignore)
├── _archive/                # Fichiers scratch mis de côté lors du ménage (à vérifier puis supprimer)
└── article/
    ├── main.tex / camera_ready.tex / references.bib / anthology-ch.cls / main.pdf
    └── illustration/       # Figures de l'article
```

## Running the code

```bash
PY="/Users/rolly/Documents/10-19_Université_et_scolarité/PhD/phd-env/bin/python"

# Simulation canonique / sweep (voir code/simulation.py --help pour toutes les options)
$PY code/simulation.py --sweep_param archive_rate --sweep_values 0.05 0.1 0.2 --plot outputs/sweep.png

# Test archive_rate vs conformity_bias (raréfaction)
$PY code/test_archive_vs_conformity_hypothesis.py

# Même test, par distribution initiale (power_law / uniform / random)
$PY code/test_conformity_full_archive_by_distribution.py

# Journal de bord
jupyter notebook code/archive_rate_conformity_exploration.ipynb
```



##  Parameters 


|  Parameters | Description   | Value  |   |   |
|---|---|---|---|---|
|  $N$| Nombre de classe  |  40 |   |   |
|  $n_i$ | Population initiale  |  1000 |   |   |
|  $n_f$ |   Population finale| 2000  |   |   |
|  $T$ | Nombre de génération  | 1000  |   |   |
|  $t$| Fenêtre d'analyse  |  [0;300] |   |   |
|  $\alpha$ | Taux d'archivage  | 0.05  |   |   |
|  (new)$b$ |Conformité (vs proba génération précédente) |[0;1]   |   |   |
|  (new)$c$ | Le contenu (à voir)  |   |   |   |



## Descriptive Stats

|   |   |   |   |
|---|---|---|---|
| KS  |   |   |   |
| MLE |   |   |   |
|Nombre de Hill (Shanon en plus ?)(diversité intra)   ||   |   |
|Gini (richesse intra) |||||
|tx de remplacement (topx)||||
