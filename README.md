# Simulation v3 : usage des tags avec biais de conformité par rang

## Modèle

Chaque individu (une page) porte un ensemble de tags (classes), en nombre tiré d'une Poisson tronquée (`lam`, minimum 1). La population croît linéairement de `Ni` à `Nf` en `T` générations. Chaque nouvel individu tire ses tags rang par rang et sans remise :
- avec probabilité `mu`, il crée une classe nouvelle ;
- sinon il copie une classe avec une probabilité proportionnelle à `f^q`, où `q` dépend du rang du tag.

`f` est la fréquence des classes parmi tous les tags de la population existante, recalculée une fois par génération (mise à jour synchrone, pas de mort). Le vecteur `Q` (un `q` par rang, 6 valeurs, la dernière répétée ensuite) est tiré une fois par simulation : Q ~ U(0, `q_max`), avec `q_max` = 4 dans `run_grid.py`.

Paramètres : `Ni`, `Nf` (population initiale et finale), `C` (classes initiales), `lam`, `mu`, `T`, `seed`, `q_max`. Modération optionnelle (`K` passes, seuil `j` : les classes de compte inférieur à `j` sont fusionnées vers une classe plus fréquente) ; `K = 0` la désactive.

## Fichiers

```
simulation.py           # modèle de référence et métriques
simulation_opti.py      # même modèle, optimisé (résultats identiques bit à bit pour une même graine) ; utilisé par run_grid.py
simulation_ecdf.py      # variante : innovations calées sur l'ecdf empirique des premières apparitions (data/ecdf_innovation.json)
run_grid.py             # grilles de simulations par blocs, parallèles et reprenables
scripts/metriques_distribution.py   # métriques de distribution (KL, Hill, Gini, Chao1) calculées après coup sur une grille
scripts/empirical_metrics.py        # mêmes métriques sur les données réelles -> data/empirical_metrics.json
scripts/ecdf_innovation.py          # -> data/ecdf_innovation.json
data/                   # valeurs empiriques déjà calculées
```

Les trois scripts de `scripts/` sont les seuls à lire des données hors dépôt, et `empirical_metrics.py` et `ecdf_innovation.py` ne servent pas sur le cluster : les JSON de `data/` suffisent.

## Lancer

```bash
pip install numpy tqdm              # + pandas scipy pour scripts/ seulement

# un run
python simulation.py --mu 0.03 --T 100 --seed 1 --verbose

# une grille (parallèle, reprenable)
python run_grid.py --n_sims 100000 --n_jobs 16 --out_dir results/grid_base/chunks --merge

# grande grille légère, sans données brutes (~1 Ko par run)
python run_grid.py --n_sims 1000000 --out_dir results/grid_fandom_1M/chunks --sans_brut --merge \
    --rng_seed 2 --seed_start 200001 --range mu 0 0.02 --range C 2 300 --fix Nf 13108 --fix lam 2 --q_max 4

# métriques de distribution, après coup
python scripts/metriques_distribution.py --grid results/grid_fandom_1M/chunks --n_jobs 8   # -> metriques.csv
```

Espace de tirage par défaut (uniforme, indépendant) : `mu` ∈ [0 ; 0,1], `lam` ∈ [1 ; 4], `C` ∈ [2 ; 200], `Ni` ∈ [10 ; 1000], `T` ∈ [20 ; 500], `Nf` = 10 000. Surcharges : `--range NOM MIN MAX`, `--fix NOM VALEUR`.

### Grilles

`run_grid.py` découpe la grille en blocs de `--chunk_size` runs (500 par défaut) et écrit `chunk_XXXXXX.npz` de façon atomique. Les paramètres d'un bloc sont tirés avec `(rng_seed, numéro de bloc)`, donc le résultat ne dépend pas de `--n_jobs`. Relancer la même commande reprend où l'on s'était arrêté ; `grid_config.json` refuse une reprise avec une autre configuration (une grille `--sans_brut` ne se reprend pas sans `--sans_brut`, et inversement).

- Mode complet : garde les données brutes de chaque run (flux de tags, tags par individu, classes par génération), environ 37 Ko par run.
- Mode léger (`--sans_brut`) : garde paramètres, Q, métriques et distribution finale, environ 1 Ko par run.


## Métriques

Toutes sont calculées par run et stockées dans la grille (`results.csv`, `metriques.csv`). Le **flux** désigne les tags de tous les individus concaténés dans l'ordre d'arrivée (l'équivalent du « texte » de Cattuto et al.). `p_k` est la fréquence finale de la classe k dans le flux. Pour l'empirique, une page est un individu, rangée par date de création.

### Paramètres et tailles

| Colonne | Définition |
|---|---|
| `q_0` … `q_5` | coefficients de conformité tirés pour le run (rang 1 à 6 ; le rang 6 sert aussi aux suivants) |
| `n_tags` | nombre total de tags dans le flux |
| `C_final` | nombre de classes créées pendant le run |
| `n_classes` | nombre de classes portées par au moins une page à la fin (= `hill_0`). L'écart avec `C_final` correspond aux classes initiales qu'aucune page n'a jamais portées, et aux classes vidées par la modération |

### Forme de la distribution des fréquences

| Colonne        | Définition                                                                                                                                                                                                                                      |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `alpha`        | exposant de la queue de la courbe fréquence-rang, P(R) ~ R^-α : pente (signe changé) d'une régression log-log sur les classes de rang ≥ 10. NaN s'il y a moins de 3 points. Estimation grossière.                                               |
| `beta`         | exposant de Heaps, D(n) ~ n^β, où D(n) est le nombre de classes distinctes parmi les n premiers tags du flux (régression log-log). β = 1 : croissance linéaire (attendu à `mu` constant) ; β < 1 : sublinéaire (attendu empiriquement).         |
| `age_effectif` | corrélation de Spearman entre l'ordre de première apparition d'une classe et son effectif final. Proche de −1 : l'effectif est dicté par l'ancienneté (« premier arrivé, premier servi ») ; proche de 0 : l'âge n'explique pas l'hétérogénéité. |

### Dynamique du top-10 (turnover)

Le turnover à l'étape t est le nombre de classes du top-10 de f_t absentes du top-10 de f_{t-1}. Un top stable (turnover faible) indique de la conformité.

| Colonne | Définition |
|---|---|
| `turnover_moy` | turnover moyen sur les T générations du run. T varie d'un run à l'autre : **ne se compare pas à l'empirique**. |
| `turnover_50` | turnover moyen du top-10 sur 50 blocs d'individus de même taille (ordre d'arrivée), f cumulé à la fin de chaque bloc, tri stable pour les ex aequo (à fréquence égale, indice de classe le plus petit en premier). **Comparable à l'empirique.** |

### Mémoire : autocorrélation du flux (Cattuto, Loreto & Pietronero 2007)

C(dt) est la proportion des positions t telles que tag(t + dt) = tag(t). Le niveau de référence est c = Σ p_k², la valeur attendue pour un flux mélangé au hasard. Une décroissance lente vers c signale une mémoire. Au décalage 1, C est tiré sous c : deux tags voisins sont souvent dans le même individu, qui n'a pas de doublon. On regarde donc dt de 10 à 100.

| Colonne | Définition |
|---|---|
| `autocorr_1` | C(1) |
| `autocorr_ref` | c = Σ p_k² |
| `autocorr_100` | C(100) |
| `exces_100` | C(100) / c − 1 |
| `exces_10_100` | moyenne de C(dt) / c − 1 pour dt = 10 … 100 (moins bruité que `exces_100`) |

### Modération (0 si `K` = 0)

| Colonne | Définition |
|---|---|
| `n_passes` | nombre de passes de modération effectuées |
| `n_fusionnees` | nombre de classes fusionnées, total sur toutes les passes |
| `n_doublons` | nombre de tags supprimés parce que la fusion les a dupliqués dans un même individu |

### Distribution finale (`scripts/metriques_distribution.py`)

Calculées après coup à partir des comptes finaux, avec les mêmes fonctions que la v2 pour comparer les grilles.

| Colonne | Définition |
|---|---|
| `kl_emp` | KL(empirique ‖ simulation) par rang, en nats. Plancher de 0,5 occurrence pour les rangs non atteints ; la masse simulée au-delà du rang 176 reste dans la normalisation, ce qui pénalise les simulations à trop de classes. |
| `hill_0`, `hill_1`, `hill_2`, `hill_inf` | nombres de Hill : richesse, exp(Shannon), 1/Simpson, 1/Berger-Parker |
| `gini_obs` | Gini sur les classes observées (compte > 0), comme l'empirique |
| `chao1`, `n_singletons`, `n_doubletons` | estimateur de richesse Chao1 et nombres de classes de compte 1 et 2 |

Référence empirique (fandom, wiki Creepypasta) : 13 108 pages, 30 892 tags, 176 classes, tags dédoublonnés par page. Les autres plateformes (SCP en, fr, ru, jp, cn, es, et Backrooms) sont dans `data/empirical_metrics.json`.

## Reproductibilité

- La graine de la ligne k vaut `seed_start + k`. Les paramètres flottants sont stockés exactement ; `results.csv` les écrit avec 17 chiffres, à relire avec `pandas.read_csv(..., float_precision="round_trip")`.
- Un run se rejoue à l'identique avec la même version de numpy, enregistrée dans `grid_config.json`. En v2, environ 1 % des runs différaient entre numpy 1.26 et 2.x.
