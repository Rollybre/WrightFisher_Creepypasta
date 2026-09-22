# Cheat-sheet — `simulation.py`

Aide-mémoire opérationnel pour lancer des simulations Wright-Fisher et les
comparer aux données empiriques (`data/fandom_data.csv`). Pour la théorie
derrière les métriques de diversité, voir `diversity_and_distance_metrics.md`.

Environnement : `phd-env` (le `python3` système n'a pas les dépendances).
```bash
PY=".../PhD/phd-env/bin/python"   # à adapter à ton chemin local
```

---

## 1. Les 3 modes

Le script a trois modes, **mutuellement exclusifs** (déterminés par les flags
présents) :

| Mode | Déclenché par | Ce qu'il fait |
|---|---|---|
| **Run** (défaut) | rien de spécial | 1 run canonique (n=`-N`), optionnellement `--validate` (n=176) |
| **Batch** | `--n_runs > 1` | N runs canoniques avec les *mêmes* paramètres (variance) |
| **Sweep** | `--sweep_param` + `--sweep_values` | 1 run par valeur d'**un** paramètre qui varie (le reste fixe) |

En mode sweep, `--n_runs`/`--validate` sont ignorés (le sweep remplace tout le
reste de `main()`).

---

## 2. Table des paramètres CLI

| Flag | Alias | Type | Défaut | Sens |
|---|---|---|---|---|
| `input` (positionnel) | — | str | `data/fandom_data.csv` | CSV des données empiriques |
| `-N` | `--n_classes` | int | 40 | Nb de catégories du modèle |
| `-ni` | `--initial_pop` | int | 1000 | Taille de population initiale |
| `-nf` | `--final_pop` | int | 2000 | Taille de population finale |
| `-alpha` | `--archive_rate` | float | 0.05 | Fraction archivée par génération |
| `-T` | `--generations` | int | 1000 | Nb de générations simulées |
| `-s` | `--seed` | int | 42 | Graine du RNG |
| `-t` | `--window` | int | 300 | *(non utilisé pour l'instant)* |
| — | `--verbose` | flag | off | Détails archive (taille, classes représentées) |
| — | `--n_runs` | int | 1 | Nb de runs canoniques identiques à moyenner (mode batch) |
| — | `--validate` | flag | off | Ajoute un run à résolution empirique (n=176, init sur proportions réelles) |
| — | `--n_classes_valid` | int | 176 | Résolution du run de validation |
| — | `--sweep_param` | choix | — | Paramètre à balayer (mode sweep) : `archive_rate`, `n_classes`, `initial_pop`, `final_pop`, `generations` |
| — | `--sweep_values` | float(s) | — | Valeurs à tester pour `--sweep_param` |
| `-o` | `--output` | str | — | Dossier de sortie (nom de fichier auto-généré) |
| — | `--plot` | str | — | Chemin d'un `.png` de sortie |

---

## 3. Recettes

**Run simple, juste regarder dans le terminal**
```bash
$PY simulation.py
```

**Run avec des paramètres différents + validation à résolution empirique**
```bash
$PY simulation.py -alpha 0.1 -N 40 -s 7 --validate
```

**Run sauvegardé (1 fichier JSON), pour du batch/cluster**
```bash
$PY simulation.py -alpha 0.1 -s 7 -o results/
# -> results/n40_alpha0.100_seed7.json
```

**Run + figure rang-fréquence (empirique vs modèle)**
```bash
$PY simulation.py -alpha 0.1 --plot figures/rankfreq.png
# avec --validate en plus : génère aussi figures/rankfreq_validation.png
```

**Plusieurs runs identiques pour voir la variance**
```bash
$PY simulation.py --n_runs 20
# (pas de plot en mode batch pour l'instant, cf. TODO en tête de fichier)
```

**Sweep local sur `archive_rate`, comparaison à l'empirique**
```bash
$PY simulation.py --sweep_param archive_rate --sweep_values 0.01 0.05 0.1 0.2 \
    -o results/ --plot figures/sweep_alpha.png
# -> results/sweep_archive_rate.csv + figures/sweep_alpha.png
```

**Boucle cluster (grille de paramètres, 1 job = 1 appel)**
```bash
for alpha in 0.01 0.05 0.1 0.2; do
  for seed in 1 2 3 4 5; do
    $PY simulation.py -alpha $alpha -s $seed -o results/ &
  done
done
wait
# chaque job écrit son propre fichier (results/n40_alpha<...>_seed<...>.json),
# jamais de fichier partagé -> pas de collision entre jobs parallèles.
# Agrégation ensuite : lire tous les results/*.json séparément (pas fait par ce script).
```

---

## 4. Fichiers produits

| Mode | `--output` | `--plot` |
|---|---|---|
| Run | `{output}/n{n_classes}_alpha{archive_rate:.3f}_seed{seed}.json` | `{plot}` (+ `{plot_stem}_validation{ext}` si `--validate`) |
| Batch (`--n_runs`) | même nom, contenu = liste de dicts (1 par run) | — (pas encore fait) |
| Sweep | `{output}/sweep_{param}.csv` | `{plot}` (1 panneau par métrique vs paramètre) |

Contenu d'un enregistrement JSON (mode run) :
```json
{
  "n_classes": 40, "initial_pop": 1000, "final_pop": 2000,
  "generations": 1000, "archive_rate": 0.1, "seed": 7,
  "gini": 0.62, "hill_0": 37.0, "hill_1": 18.85, "hill_2": 11.54, "hill_inf": 5.03,
  "validation": { ...mêmes clés, pour le run n=176... }   // si --validate
}
```

---

## 5. Métriques de diversité (`compute_diversity_metrics`)

`runtime_s` est aussi mesuré et inclus automatiquement dans chaque résultat (JSON, CSV, ou
sweep — moyenné + `runtime_s_std` avec `--sweep_repeats`) : utile pour juger si une expérience
vaut la peine de tourner sur HPC plutôt qu'en local (cf. mesures : ~0.09s/run à l'échelle
papier n=176/T=1000 — largement local jusqu'à plusieurs milliers de runs).

| Clé | Nom | Interprétation |
|---|---|---|
| `gini` | Indice de Gini | 0 = parfaitement égal, 1 = totalement inégal |
| `hill_0` | Richesse | Nb de catégories effectivement représentées (fréquence > 0) |
| `hill_1` | exp(Shannon) | Diversité effective, poids égal à toutes les catégories présentes |
| `hill_2` | Inverse de Simpson | Diversité effective, penalise les catégories rares |
| `hill_inf` | Inverse de Berger-Parker | Domination de la catégorie la plus fréquente (proche de 1 = très dominée) |

`hill_number(x, order)` (fait maison, remplace `skbio.hill`) ignore automatiquement
les catégories à fréquence 0 et accepte n'importe quel `order` réel ≥ 0 (pas
seulement des entiers).

---

## 6. Limitations connues

- `-t/--window` est défini mais non utilisé.
- Le sweep ne fait varier qu'**un seul** paramètre à la fois (pas de grille
  combinée en Python) ; pour une grille, utiliser la boucle bash cluster (§3).
- Pas de figure dédiée au mode `--n_runs` (moyenne + percentiles sur plusieurs
  runs, à la manière de la Figure B de `analysis_stats.py`) — pas encore fait.
- Les figures sont toujours sauvegardées en fichier (backend `Agg`, jamais
  `plt.show()`) — safe sur cluster, mais pas d'affichage interactif direct.
