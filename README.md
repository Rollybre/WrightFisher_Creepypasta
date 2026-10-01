# CultureLab : modèle de transmission culturelle avec archive cumulative

Simulation d'une **archive de classes (tags)** qui grossit de `n_i` à `n_f` individus en `T` générations : à chaque génération une fraction `mu` des nouveaux individus crée une nouvelle classe, les autres copient une classe de l'archive avec une probabilité proportionnelle à `effectif^q` (`q` < 1 : avantage aux classes rares, `q` > 1 : conformisme). Le dépôt contient le modèle, un exécuteur de grandes grilles (millions de runs, parallèle, reprenable) et les outils pour comparer les résultats à un corpus empirique (176 catégories, 30 901 occurrences).

Un tour d'horizon illustré (modèle, implémentation, validation, résultats) : [`notebooks/topo_modele.ipynb`](notebooks/topo_modele.ipynb). L'analyse détaillée et les interprétations : [`notebooks/results.ipynb`](notebooks/results.ipynb).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # numpy, pandas, scipy, matplotlib, seaborn (+ jupyter pour les notebooks)
python -m unittest discover -s tests -t .  # 22 tests, ~10 s
```

Pas de package à installer : les modules (`simulation.py`, `run_grid.py`, `analysis.py`) sont à la racine ; les scripts et les notebooks ajoutent eux-mêmes la racine du dépôt au `sys.path`.

## Utilisation

```bash
# Un run : indices de diversité, et figure (rang-fréquence + métriques au fil des générations)
python simulation.py --C 100 --T 600 --mu 0.007 --q 0.73 --n_i 76 --n_f 30901 --alpha 1.8 --seed 1 --plot run.png

# Une table de paramètres tirés au hasard : exécutée en local (--n_jobs processus) ou seulement écrite (--hpc)
python simulation.py --n_sims 1000 --n_jobs -1 -o results.csv
python simulation.py --n_sims 1000 --hpc -o params.csv
```

```python
from simulation import Simulation, compute_metrics, kl_divergence

sim = Simulation(n_i=76, n_f=30_901, C=110, T=600, alpha=1.78, mu=0.0068, q=0.73, seed=1).run()
sim.archive      # effectifs par classe de l'archive finale (somme = n_f)
sim.metrics      # gini, gini_obs, hill_0/1/2/inf, chao1, n_classes, kl_emp
```

### Indices calculés sur l'archive finale

| Indice | Définition |
|---|---|
| `gini_obs` | Gini sur les classes observées (effectif > 0) : à comparer à l'empirique |
| `gini` | Gini incluant les classes initiales jamais échantillonnées (effectif 0) : gonflé par ces classes |
| `hill_0`, `hill_1`, `hill_2`, `hill_inf` | nombres de Hill (richesse, exp. Shannon, inverse de Simpson, inverse de Berger-Parker) |
| `chao1` | richesse estimée (singletons / doubletons) |
| `kl_emp` | KL(empirique ‖ simulation) entre distributions rang-fréquence (comptes empiriques dans `simulation.EMPIRICAL_COUNTS`) |

## Grandes grilles

`run_grid.py` découpe une grille en blocs de `--chunk_size` runs (5 000 par défaut). Chaque bloc tire ses paramètres (générateur seedé par `(rng_seed, numéro de bloc)`), exécute ses simulations et écrit un fichier `chunk_XXXXXX.npz` (paramètres exacts, indices, distribution finale de chaque run) de façon atomique : parallélisable, reprenable (relancer la même commande saute les blocs terminés), résultat indépendant de `--n_jobs`. Un `grid_config.json` mémorise les bornes et refuse une reprise avec une autre configuration.

```bash
# grille « n_f = 10 000 » : bornes par défaut (cf. simulation.py)
python run_grid.py --n_sims 5000000 --n_jobs 16 --out_dir results/grid_n10000/chunks

# grille « n_f = 30 901 » (taille réelle du corpus) : n_f fixé, mu rééchelonné
python run_grid.py --n_sims 5000000 --n_jobs 16 --out_dir results/grid_n30901/chunks \
    --fix n_f 30901 --range mu 0 0.02

# options utiles : --merge (écrit aussi results.csv), --upgrade (ajoute aux blocs existants les métriques manquantes)
```

Ordres de grandeur (CPU du serveur) : 5 M de runs en 60 min (16 processus, `n_f` = 10 000) et 71 min (20 processus, `n_f` = 30 901) ; ≈ 650 Mo et ≈ 850 Mo de blocs.

### Sur un serveur

```bash
rsync -av simulation.py run_grid.py serveur:CultureLab_v2/
ssh serveur
cd CultureLab_v2 && tmux new -s grille
python3 run_grid.py --n_sims 5000000 --n_jobs 16 --out_dir ~/grid_n30901 --fix n_f 30901 --range mu 0 0.02 2>&1 | tee ~/grid_n30901.log
# Ctrl-b puis d pour détacher ; relancer la même commande pour reprendre après une interruption
```

Le gain de parallélisme plafonne (≈ ×8 à 24 processus sur un CPU hybride) : `python scripts/benchmark.py` le mesure sur la machine cible.

## Résultats (non versionnés)

Les blocs sont volumineux et ne sont pas dans git (`results/` est ignoré). Les notebooks les attendent dans :

```
results/grid_n10000/chunks/chunk_*.npz     # n_f = 10 000
results/grid_n30901/chunks/chunk_*.npz     # n_f = 30 901
```

Pour les rapatrier depuis un serveur :

```bash
mkdir -p results/grid_n30901/chunks
rsync -a --include='chunk_*.npz' --include='grid_config.json' --exclude='*' serveur:grid_n30901/ results/grid_n30901/chunks/
```

```python
from run_grid import load_results, load_counts
from analysis import Explorer

df = ...                                                   # pandas.DataFrame(load_results("results/grid_n30901/chunks"))
counts = load_counts("results/grid_n30901/chunks", 12345)  # distribution finale (comptes triés) du run 12345
ex = Explorer("results/grid_n30901/chunks")
ex.pairplot(mode="cross"); ex.index("kl_emp"); ex.closest(10_000).pairplot()
```

## Structure

```
simulation.py     # modèle (Simulation), indices, kl_divergence, tirage de paramètres, CLI
run_grid.py       # grilles par blocs, reprise, garde-fou de configuration, relecture (load_results, load_counts)
analysis.py       # Explorer (pair plot, détail d'un indice, meilleurs runs), ajustement MLE de loi de puissance
requirements.txt
scripts/          # benchmark.py (scaling), empirical_counts.py (recalcule les comptes empiriques), ecdf_empirical.py
notebooks/        # topo_modele.ipynb (présentation illustrée), results.ipynb (analyse et interprétations)
tests/            # 22 tests (modèle, indices, données, grille en sous-processus)
data/             # corpus empirique (fandom_data.csv, fandom_links.csv)
results/          # (ignoré par git) blocs de simulation
```

## Reproductibilité

- Chaque run a sa graine (`seed = seed_start + numéro de ligne`) et les paramètres tirés `alpha`, `mu`, `q` sont stockés en float64 exact : la simulation est très sensible au dernier chiffre. Relire un CSV avec `float_precision="round_trip"` ; `merge_csv` écrit 17 chiffres significatifs.
- Un run se rejoue à l'identique sur une même version de numpy ; entre numpy 1.26 (serveur) et 2.x (local), environ 1 % des runs diffèrent (arrondis de `pow` / `binomial` / `multinomial` amplifiés par la simulation). Les blocs stockés restent cohérents entre eux.
- Les comptes empiriques sont écrits en dur ; `python scripts/empirical_counts.py` les recalcule depuis `data/fandom_data.csv` (vérifié par les tests).

## Données et historique

`data/` contient le corpus empirique utilisé (histoires de la wiki Creepypasta, Fandom) : à ne pas diffuser publiquement sans en avoir vérifié la licence. L'ancienne implémentation (modèle Wright-Fisher à taux d'archivage, sans innovation) est dans le dépôt `Rollybre/WrightFisher_Creepypasta`.
