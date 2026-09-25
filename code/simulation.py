"""
Lance des simulations Wright-Fisher (avec archivage cumulatif) et les compare
aux données empiriques (catégories de fandom).

Ce script reprend une partie de test_stats.py (chargement des données
empiriques, métriques de diversité, modèle WF, run de validation n=176,
moyenne sur plusieurs runs) mais se concentre uniquement sur le fait de
LANCER des simulations depuis la ligne de commande, sans générer les
figures/fits de l'article (qui restent dans analysis_stats.py).

TODO :
- Print comparatif par simulation  [fait: print_diversity_metrics par run avec --n_runs]
- Reprendre la métrique nombre de hill et la coder pour éviter de passer par une lib  [fait: hill_number]
- Rendre hill_number robuste aux fréquences nulles et aux q non-entiers (cf. discussion)  [fait]
- Sweep local sur un paramètre (ex: archive_rate sur un intervalle) + comparaison empirique/simulé  [fait: sweep_parameter, --sweep_param/--sweep_values]
- Fonctions de plot (repris de analysis_stats.py, sans la dépendance à `powerlaw`)  [fait: plot_rank_frequency, plot_sweep_metrics, --plot]
- Plot pour le mode --n_runs (moyenne + percentiles sur plusieurs runs, cf. Figure B de analysis_stats.py) — pas encore fait

"""

# ============================================================================
# IMPORTS
# ============================================================================
import argparse
import ast
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # backend sans affichage : on ne fait jamais plt.show(), toujours plt.savefig()
import matplotlib.pyplot as plt   # (safe même sans écran/DISPLAY, ex. sur un nœud de calcul cluster)


# ============================================================================
# CONSTANTES / CHEMINS
# ============================================================================
ROOT = Path(__file__).parent
DATA_PATH = ROOT.parent / "data" / "fandom_data.csv"   # chemin relatif au script (portable, remplace le chemin en dur)
N_TOP = 40   # nb de catégories les plus fréquentes retenues pour comparer aux simulations (résolution n=40)


# ============================================================================
# CHARGEMENT ET NETTOYAGE DES DONNÉES EMPIRIQUES
# ============================================================================
def clean_cats(cats):
    """Éclate les catégories combinées ('A|B') en catégories individuelles et enlève les espaces."""
    return [p.strip() for c in cats for p in c.split("|") if p.strip()]


def load_empirical_data(data_path):
    """Charge le CSV et retourne (df, counter des catégories, fréquences triées décroissant)."""
    print("Loading empirical data...")
    df = pd.read_csv(data_path)
    # La colonne "Category" est une chaîne représentant une liste Python (ex: "['A|B', 'C']") -> on la reparse
    df["Category_list"] = df["Category"].apply(ast.literal_eval)
    df["Category_clean"] = df["Category_list"].apply(clean_cats)

    # Liste à plat de toutes les occurrences de catégories, toutes histoires confondues
    all_cats = [cat for cats in df["Category_clean"] for cat in cats]
    counter = Counter(all_cats)

    # Fréquences empiriques (nb d'occurrences par catégorie), triées par ordre décroissant
    emp_freq_all = np.array(sorted(counter.values(), reverse=True), dtype=float)
    return df, counter, emp_freq_all


def summarize_empirical_data(df, emp_freq_all, n_top):
    """Affiche taille des données + couverture du Top-N, retourne (emp_freq, emp_ranks)."""
    # On ne garde que le Top-N pour comparer au modèle (même résolution que le modèle: n=40)
    emp_freq = emp_freq_all[:n_top]
    emp_ranks = np.arange(1, len(emp_freq) + 1)

    # Part de l'ensemble des occurrences couverte par ce Top-N (contrôle de représentativité)
    coverage = emp_freq.sum() / emp_freq_all.sum() * 100
    print(f"  {len(df)} stories, {len(emp_freq_all)} categories total")
    print(f"  Top {n_top} categories: {int(emp_freq.sum())} occurrences ({coverage:.1f}% of total)")
    return emp_freq, emp_ranks


# ============================================================================
# MÉTRIQUES DE DIVERSITÉ
# ============================================================================
def gini(x):
    """Indice de Gini (0 = parfaitement égal, 1 = totalement inégal) calculé sur un vecteur de fréquences."""
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x) ** 2 * np.mean(x))


def hill_number(x, order):
    """Nombre de Hill d'ordre `order` pour un vecteur de fréquences/abondances x.
    order=0: richesse, 1: exp(Shannon), 2: inverse de Simpson, np.inf: inverse de Berger-Parker.
    Valide pour tout `order` réel >= 0 (pas seulement les entiers, ex. order=1.5 est légitime)."""
    # raise error au cas où : on accepte tout nombre réel positif (int, float, np.inf), pas que des int
    if not isinstance(order, (int, float, np.integer, np.floating)) or order < 0:
        raise ValueError(f"order doit être un nombre réel >= 0 (ou np.inf), reçu {order!r}")

    # on force un array pour être sûr
    x = np.array(x, dtype=float)
    # Les catégories absentes (fréquence 0) ne comptent pas dans le calcul de diversité : on les
    # retire avant toute chose. Sans ça, log(0) casse le cas order=1 (Shannon -> nan), et 0**0 = 1
    # (convention numpy) fausserait la richesse pour order=0 en comptant des catégories absentes.
    x = x[x > 0]
    if x.size == 0:
        raise ValueError("hill_number: aucune fréquence strictement positive dans x")
    x_prop = x/x.sum()

    if order == 1 :
        return np.exp(-((np.log(x_prop)*x_prop).sum()))
    elif order == np.inf:
        return 1/x_prop.max()
    else :
        return (x_prop**order).sum()**(1/(1-order))



def _hill_key(q):
    """Nom de clé stable pour un ordre de Hill donné (utilisé dans les dicts de métriques)."""
    return "hill_inf" if np.isinf(q) else f"hill_{q}"


def compute_diversity_metrics(freqs, orders=(0, 1, 2, np.inf)):
    """Calcule l'indice de Gini et les nombres de Hill (q dans `orders`) pour une distribution de
    fréquences, et les retourne dans un dict {"gini": ..., "hill_0": ..., "hill_1": ..., "hill_inf": ...}.
    Séparée de l'affichage pour pouvoir réutiliser ces valeurs (comparaison entre runs, sauvegarde, plots)."""
    metrics = {"gini": gini(freqs)}
    for q in orders:
        metrics[_hill_key(q)] = hill_number(freqs, order=q)
    return metrics


def print_diversity_metrics(freqs, label="", orders=(0, 1, 2, np.inf), compare_to=None):
    """Calcule (compute_diversity_metrics) puis affiche l'indice de Gini et les nombres de Hill.
    orders=0: richesse, 1: exp(Shannon), 2: inverse de Simpson, inf: inverse de Berger-Parker
    (plus q augmente, moins les catégories rares pèsent dans l'indice). Retourne le dict calculé.

    `compare_to` (optionnel) : dict de métriques empiriques (même format que le retour de
    compute_diversity_metrics, ex. `compute_diversity_metrics(emp_freq_all)`) -- si fourni, chaque
    ligne affiche en plus la valeur empirique et l'écart (Δ = simulé - empirique), pour comparer
    à chaque run sans avoir à se référer au print initial des données empiriques (cf. --compare_empirical)."""
    names = {0: "Richesse", 1: "Shannon", 2: "Inverse de Simpson", np.inf: "Inverse de Berger-Parker"}
    metrics = compute_diversity_metrics(freqs, orders=orders)
    if label:
        print(f"\n-- Diversité : {label} --")

    def _line(key, display_name, value, decimals):
        if compare_to is not None and key in compare_to:
            emp_value = compare_to[key]
            delta = value - emp_value
            print(f'  {display_name} : {value:.{decimals}f}  '
                  f'(empirique: {emp_value:.{decimals}f}, Δ={delta:+.{decimals}f})')
        else:
            print(f'  {display_name} : {value:.{decimals}f}')

    _line("gini", "Indice de Gini", metrics["gini"], decimals=4)
    for q in orders:
        key = _hill_key(q)
        _line(key, f"Nombre de Hill ordre {q} ({names.get(q, f'q={q}')})", metrics[key], decimals=2)
    return metrics


def frequencies_from_archive(archive, n_classes=None):
    """Convertit une archive (array d'indices de catégories) en fréquences triées décroissant.
    Si n_classes est fourni, inclut les classes non représentées (fréquence 0)."""
    counts = Counter(archive)
    if n_classes is not None:
        freq = np.array([counts.get(i, 0) for i in range(n_classes)], dtype=float)
        freq = np.sort(freq)[::-1]
    else:
        freq = np.array(sorted(counts.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freq) + 1)
    return freq, ranks


# ============================================================================
# DISTRIBUTION INITIALE (état de la population à t=0)
# ============================================================================
def initial_distribution(n_classes, distribution, rng, custom_probs=None):
    """Retourne un vecteur de probabilités (longueur n_classes, somme=1) pour tirer l'état
    initial de la population, selon la forme demandée :
      - "power_law" : rang^-1 (catégorie 0 = la plus fréquente) — comportement historique,
        défaut du modèle depuis le début du projet.
      - "uniform"   : toutes les classes équiprobables (aucune structure de départ).
      - "random"    : composition aléatoire tirée d'une loi de Dirichlet(1,...,1) — une
        distribution différente à chaque `rng` (donc reproductible via --seed), sans forme
        imposée a priori (ni plate, ni en loi de puissance).
      - "custom"    : proportions fournies par l'appelant (`custom_probs`, ex: lues depuis un
        CSV via --distribution_path), normalisées si besoin.
    Ne pas confondre avec `init_probs` dans run_simulation : `init_probs` reste le mécanisme
    existant pour imposer directement des proportions empiriques réelles (cf. run_validation) et
    prend toujours le pas sur `distribution` si les deux sont fournis."""
    if distribution == "power_law":
        ranks = np.arange(1, n_classes + 1, dtype=float)
        probs = ranks ** -1.0
    elif distribution == "uniform":
        probs = np.ones(n_classes, dtype=float)
    elif distribution == "random":
        return rng.dirichlet(np.ones(n_classes))   # déjà normalisé
    elif distribution == "custom":
        if custom_probs is None:
            raise ValueError("distribution='custom' nécessite custom_probs (ou --distribution_path)")
        probs = np.asarray(custom_probs, dtype=float)
        if len(probs) != n_classes:
            raise ValueError(
                f"custom_probs a {len(probs)} valeurs, attendu n_classes={n_classes}"
            )
    else:
        raise ValueError(
            f"distribution inconnue: {distribution!r} "
            "(choix : power_law, uniform, random, custom)"
        )
    return probs / probs.sum()


# ============================================================================
# MODÈLE DE SIMULATION (Wright-Fisher avec archivage cumulatif)
# ============================================================================
def run_simulation(rng, n_classes, n_init, n_final, t_max, archive_rate,
                   init_probs=None, conformity_bias=1, distribution="power_law", custom_probs=None,
                   archive=True):
    """Single WF run. Retourne soit l'archive cumulative (comportement historique, `archive=True`
    par défaut), soit -- si `archive=False` -- l'état de la POPULATION FINALE SEULE (génération
    t_max-1, aucune accumulation sur les générations précédentes).

    Ne pas confondre `archive=False` avec `archive_rate` proche de 0 : `archive_rate` ne contrôle
    QUE la fraction de chaque génération ajoutée à l'archive cumulative -- même à `archive_rate=1.0`
    (100%, "aucune perte"), l'archive reste la somme de TOUTES les générations (donc bien plus
    grande et bien plus diverse que n'importe quelle génération seule). `archive=False` est le
    seul moyen d'obtenir la diversité d'UNE SEULE génération (la dernière), sans cet effet cumulatif
    -- ce que `archive_rate=1.0` ne fait PAS, contrairement à une intuition naturelle mais fausse
    (l'archivage à 100% de chaque génération n'équivaut jamais à ne pas archiver du tout)."""
    # --- État initial : n_init individus répartis sur n_classes catégories ---
    if init_probs is None:
        # Pas de proportions imposées explicitement -> on construit la distribution initiale
        # selon la forme demandée (power_law par défaut = comportement historique inchangé)
        init_probs = initial_distribution(n_classes, distribution, rng, custom_probs=custom_probs)
    state = rng.choice(n_classes, size=n_init, p=init_probs)

    # Taille de la population à chaque génération t : interpolation linéaire entre n_init et n_final
    pop_sizes = [
        int(n_init + t * (n_final - n_init) / t_max)
        for t in range(t_max)
    ]

    archive_list = []
    for t in range(t_max):

        # Rééchantillonnage Wright-Fisher : tirage avec remise dans la génération précédente
#        state = rng.choice(state, replace=True, size=pop_sizes[t])
        bins= np.bincount(state,minlength = n_classes)**conformity_bias
        conform_probs = bins /bins.sum()
        state = rng.choice(n_classes, p=conform_probs, replace=True, size = pop_sizes[t])
        #En partant de la seed (rng.choice), on associe chaque classe à une prob donnée plus haut
        #L'échantillon est de taille pop_sizes[t] ie t-ème génération dans la table de taille prédéfinie.
        # Une fraction archive_rate de la génération courante est archivée (sans remise), de façon
        # cumulative -- seulement si archive=True (comportement historique, cf. docstring).
        if archive:
            n_arch = int(len(state) * archive_rate)
            if n_arch > 0:
                archive_list.extend(rng.choice(state, replace=False, size=n_arch))

    # archive=False : pas d'accumulation -> on retourne uniquement la dernière génération (`state`),
    # jamais l'archive cumulative (qui reste vide dans ce cas).
    return np.array(archive_list) if archive else np.array(state)


def run_multiple_simulations(rng, n_runs, n_classes, n_init, n_final, t_max, archive_rate,
                              init_probs=None, conformity_bias=1, distribution="power_law",
                              custom_probs=None, archive=True):
    """Lance n_runs simulations indépendantes. Retourne la liste des distributions de fréquences
    (une par run, triées décroissant, classes absentes incluses avec fréquence 0)."""
    all_freqs = []
    for _ in range(n_runs):
        result = run_simulation(rng, n_classes, n_init, n_final, t_max, archive_rate, init_probs,
                                 conformity_bias, distribution=distribution, custom_probs=custom_probs,
                                 archive=archive)
        freq, _ = frequencies_from_archive(result, n_classes=n_classes)
        all_freqs.append(freq)
    return all_freqs


# ============================================================================
# ORCHESTRATION DES RUNS (canonique n=40 / validation n=176)
# ============================================================================
def run_canonical(rng, n_classes, n_init, n_final, t_max, archive_rate, verbose=False, conformity_bias=1,
                   distribution="power_law", custom_probs=None, archive=True):
    """Run 'canonique' (paramètres papier, n=40 par défaut, distribution initiale paramétrable
    via `distribution` : power_law (défaut historique), uniform, random, ou custom)."""
    print(f"\nRunning canonical simulation (n={n_classes}, distribution={distribution}, archive={archive})...")
    result = run_simulation(rng, n_classes, n_init, n_final, t_max, archive_rate,
                             conformity_bias=conformity_bias, distribution=distribution,
                             custom_probs=custom_probs, archive=archive)
    # Distribution de fréquences du modèle, triée décroissant (comparable à emp_freq)
    freq, ranks = frequencies_from_archive(result)
    if verbose:
        label = "Archive size" if archive else "Population finale (pas d'archivage)"
        print(f"  {label}: {len(result)}, {len(freq)} classes represented")
    return result, freq, ranks


def run_validation(rng, emp_freq_all, n_classes_valid, n_init, n_final, n_classes_paper,
                    t_max, archive_rate, verbose=False, conformity_bias=1, archive=True):
    """Run de 'validation' à la résolution empirique (n=176 par défaut), initialisé sur les
    proportions empiriques réelles, pour comparaison directe avec les données."""
    # Scale n_i / n_f proportionnellement pour garder la même densité par classe
    # (original: 1000 individus / 40 classes = 25 par classe en moyenne)
    n_init_valid = int(n_init * n_classes_valid / n_classes_paper)
    n_final_valid = int(n_final * n_classes_valid / n_classes_paper)
    emp_probs = emp_freq_all / emp_freq_all.sum()

    print(f"\nRunning validation simulation (n={n_classes_valid}, n_i={n_init_valid})...")
    archive_result = run_simulation(
        rng, n_classes_valid, n_init_valid, n_final_valid, t_max, archive_rate, archive=archive,
        init_probs=emp_probs, conformity_bias=conformity_bias
    )
    freq, ranks = frequencies_from_archive(archive_result, n_classes=n_classes_valid)
    if verbose:
        label = "Archive size" if archive else "Population finale (pas d'archivage)"
        print(f"  {label}: {len(archive_result)}, {(freq > 0).sum()} classes represented")
    return archive_result, freq, ranks


# ============================================================================
# SWEEP LOCAL (fait varier UN paramètre sur une liste de valeurs, en mémoire)
# ============================================================================
def sweep_parameter(param_name, values, rng, n_classes, initial_pop, final_pop, generations,
                     archive_rate, init_probs=None, n_repeats=1, distribution="power_law",
                     custom_probs=None, archive=True, conformity_bias=1):
    """Relance une simulation pour chaque valeur de `values`, en ne changeant QUE `param_name`
    (les autres paramètres restent fixes aux valeurs passées). Tout se fait en mémoire, en
    réutilisant les mêmes fonctions que le mode 'un run' (pas de sous-processus, pas de
    rechargement du CSV) : c'est le mode pour explorer vite en local, avant de passer au calcul
    cluster (où chaque valeur deviendrait plutôt un job séparé, via --output).

    n_repeats > 1 : répète chaque valeur n_repeats fois et moyenne les métriques (+ écart-type),
    pour distinguer un vrai effet du paramètre du bruit d'un tirage stochastique unique. Le même
    `rng` est réutilisé d'une répétition à l'autre SANS être reseedé à chaque fois -> chaque
    répétition tire des nombres différents (sinon moyenner n'aurait aucun sens), mais l'ensemble
    du sweep reste reproductible d'un bout à l'autre via le --seed initial qui a créé ce `rng`.

    Retourne un tuple (avg_df, raw_df) :
    - avg_df : une ligne par valeur testée, métriques moyennées sur les répétitions (+ colonnes
      *_std si n_repeats > 1). C'est ce qu'utilise plot_sweep_metrics.
    - raw_df : une ligne par (valeur, répétition) SANS moyenne — chaque run individuel gardé tel
      quel, avec son numéro de répétition. À sauvegarder séparément si on veut la distribution
      complète des runs plutôt que juste leur moyenne."""
    # Les noms ici correspondent exactement aux arguments CLI (n_classes, initial_pop, ...)
    # pour rester cohérent avec `params` dans main() et éviter toute confusion de nommage.
    base_params = {
        "n_classes": n_classes, "initial_pop": initial_pop, "final_pop": final_pop,
        "generations": generations, "archive_rate": archive_rate, "conformity_bias": conformity_bias,
    }
    if param_name not in base_params:
        raise ValueError(f"param_name doit être un de {list(base_params)}, reçu {param_name!r}")

    rows, raw_rows = [], []
    for value in values:
        # On repart des paramètres de base à chaque itération, en écrasant uniquement param_name
        run_params = {**base_params, param_name: value}

        repeat_metrics = []
        for repeat_i in range(1, n_repeats + 1):
            # Chronométré ici (et pas dans run_simulation) pour que la mesure reste dans le
            # DataFrame de résultats comme une métrique de plus (moyennée/écart-type avec le reste)
            t0 = time.perf_counter()
            result = run_simulation(
                rng, run_params["n_classes"], run_params["initial_pop"], run_params["final_pop"],
                run_params["generations"], run_params["archive_rate"], init_probs=init_probs,
                conformity_bias=run_params["conformity_bias"],
                distribution=distribution, custom_probs=custom_probs, archive=archive
            )
            elapsed = time.perf_counter() - t0
            # n_classes peut lui-même être le paramètre balayé -> toujours utile pour compléter
            # les classes non représentées par des zéros (comparaison cohérente d'une valeur à l'autre)
            freq, _ = frequencies_from_archive(result, n_classes=run_params["n_classes"])
            metrics = compute_diversity_metrics(freq)
            metrics["runtime_s"] = elapsed
            repeat_metrics.append(metrics)
            raw_rows.append({**run_params, "repeat": repeat_i, **metrics})

        metrics_df = pd.DataFrame(repeat_metrics)
        row = {**run_params, **metrics_df.mean().to_dict()}
        if n_repeats > 1:
            # Écart-type entre répétitions : un point "meilleur" avec un std élevé peut n'être
            # qu'un tirage chanceux -> à regarder avant de conclure à un effet du paramètre.
            row.update({f"{k}_std": v for k, v in metrics_df.std().to_dict().items()})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(raw_rows)


# ============================================================================
# VISUALISATION
# ============================================================================
def plot_rank_frequency(emp_freq, model_freq, output_path, label_model="Simulation"):
    """Nuage de points rang-fréquence en échelle log-log, empirique vs simulé.
    Repris de la logique de analysis_stats.py, mais sans le fit power-law (donc sans dépendre
    de la librairie `powerlaw`) : uniquement les points observés, pour une comparaison visuelle rapide.
    Toujours sauvegardé dans un fichier (jamais affiché à l'écran, cf. backend Agg en tête de fichier)."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    emp_ranks = np.arange(1, len(emp_freq) + 1)
    model_ranks = np.arange(1, len(model_freq) + 1)
    ax.loglog(emp_ranks, emp_freq, "o", ms=4, alpha=0.7, color="#2171b5", label="Empirique")
    ax.loglog(model_ranks, model_freq, "s", ms=4, alpha=0.7, color="#41ab5d", label=label_model)
    ax.set_xlabel("Rang")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution rang-fréquence : empirique vs simulé")
    ax.legend()
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)   # libère la mémoire de la figure (important si on en génère plusieurs dans un même run)
    print(f"\nFigure sauvegardée : {output_path}")


def plot_rank_frequency_band(emp_freq, all_freqs, output_path, label_model="Simulation"):
    """Distribution rang-fréquence MOYENNÉE sur plusieurs runs (mode --n_runs), avec bande
    5e-95e percentile : reprend la logique de la Figure 6 de analysis_stats.py (moyenne +
    percentiles sur N_RUNS simulations). `all_freqs` est la liste des distributions de fréquences
    (une par run, triées décroissant, classes absentes incluses avec des zéros)."""
    # Les runs peuvent avoir des longueurs différentes (classes non représentées selon le run) ->
    # on complète avec des zéros à la longueur max pour pouvoir empiler en un seul tableau 2D
    max_len = max(len(f) for f in all_freqs)
    padded = np.array([np.pad(f, (0, max_len - len(f))) for f in all_freqs], dtype=float)
    mean_freq = padded.mean(axis=0)
    p05 = np.percentile(padded, 5, axis=0)
    p95 = np.percentile(padded, 95, axis=0)
    ranks = np.arange(1, max_len + 1)

    # Retire la queue où la moyenne est nulle (rangs jamais atteints par aucun run) : pas informatif
    nonzero = mean_freq > 0
    mean_freq, p05, p95, ranks = mean_freq[nonzero], p05[nonzero], p95[nonzero], ranks[nonzero]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    emp_ranks = np.arange(1, len(emp_freq) + 1)
    ax.fill_between(ranks, p05, p95, alpha=0.2, color="#41ab5d", label="5e-95e percentile")
    ax.loglog(ranks, mean_freq, "-", color="#41ab5d", lw=2,
              label=f"{label_model} (moyenne, {len(all_freqs)} runs)")
    ax.loglog(emp_ranks, emp_freq, "o", ms=4, alpha=0.7, color="#2171b5", label="Empirique")
    ax.set_xlabel("Rang")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution rang-fréquence moyennée : empirique vs simulé")
    ax.legend()
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure sauvegardée : {output_path}")


def save_distributions(all_freqs, output_path):
    """Sauvegarde les distributions de fréquences brutes en CSV (une ligne par run, triées
    décroissant, complétées par des zéros) : permet de rejouer une analyse ou un plot plus tard
    sans refaire tourner les simulations (mode --n_runs, réponse au besoin 'garder les
    distributions à la fin')."""
    max_len = max(len(f) for f in all_freqs)
    padded = np.array([np.pad(f, (0, max_len - len(f))) for f in all_freqs], dtype=float)
    df = pd.DataFrame(padded, columns=[f"rank_{i + 1}" for i in range(max_len)])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index_label="run")
    print(f"\nDistributions sauvegardées : {output_path}")


def plot_sweep_metrics(sweep_df, param_name, emp_metrics, output_path,
                        metrics=("gini", "hill_0", "hill_1", "hill_2", "hill_inf")):
    """Un panneau par métrique de diversité, valeur simulée en fonction du paramètre balayé,
    avec une ligne horizontale pointillée = valeur empirique correspondante (référence à atteindre).
    C'est la réponse directe au besoin 'comparer empirique/simulé sur un intervalle de alpha'."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), dpi=150)
    for ax, metric in zip(axes, metrics):
        std_col = f"{metric}_std"
        if std_col in sweep_df.columns:
            # Barres d'erreur = écart-type entre répétitions (--sweep_repeats > 1) : permet de
            # voir si un point qui a l'air "meilleur" l'est vraiment, ou si c'est dans le bruit.
            ax.errorbar(sweep_df[param_name], sweep_df[metric], yerr=sweep_df[std_col],
                        fmt="o-", color="#41ab5d", label="Simulation", capsize=3)
        else:
            ax.plot(sweep_df[param_name], sweep_df[metric], "o-", color="#41ab5d", label="Simulation")
        if metric in emp_metrics:
            ax.axhline(emp_metrics[metric], color="#d73027", ls="--", label="Empirique")
        ax.set_xlabel(param_name)
        ax.set_title(metric)
        ax.legend(fontsize=8)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure sauvegardée : {output_path}")


def plot_grid_heatmap(results_df, param_x, param_y, output_path,
                       metrics=("gini", "hill_0", "hill_1", "hill_2", "hill_inf")):
    """Heatmap pour un sweep 2D (deux paramètres croisés, ex. archive_rate x conformity_bias) :
    une case = moyenne de la métrique sur toutes les seeds pour cette combinaison. Un panneau par
    métrique. Contrairement à plot_sweep_metrics (1 seul paramètre, moyenne sur tout le reste),
    ici les deux axes sont conservés -- pas de moyenne qui mélangerait les deux dimensions."""
    x_values = sorted(results_df[param_x].unique())
    y_values = sorted(results_df[param_y].unique())

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.2), dpi=150)
    for ax, metric in zip(axes, metrics):
        pivot = results_df.pivot_table(index=param_y, columns=param_x, values=metric, aggfunc="mean")
        pivot = pivot.reindex(index=y_values, columns=x_values)
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis",
                        extent=[min(x_values), max(x_values), min(y_values), max(y_values)])
        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        ax.set_title(metric)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure sauvegardée : {output_path}")


# ============================================================================
# ENREGISTREMENT DES RÉSULTATS (pensé pour un usage cluster : 1 run = 1 fichier)
# ============================================================================
def make_result_record(params, freq):
    """Assemble un enregistrement plat {params..., métriques...} pour un run donné.
    C'est l'unité de résultat : un appel CLI avec un jeu de `params` produit un de ces dicts."""
    record = dict(params)
    record.update(compute_diversity_metrics(freq))
    return record


def save_results(results, output_path):
    """Écrit un résultat (ou une liste de résultats) dans un fichier JSON.
    Un fichier par run/appel : sur cluster, chaque job doit écrire le sien (jamais un fichier
    partagé entre jobs parallèles) ; l'agrégation de plusieurs fichiers se fait ensuite, à part."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRésultat(s) sauvegardé(s) : {output_path}")


# ============================================================================
# POINT D'ENTRÉE CLI
# ============================================================================
def build_arg_parser():
    """Définition des arguments en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Lance des simulations Wright-Fisher et les compare aux données empiriques."
    )
    parser.add_argument("input", nargs="?", default=str(DATA_PATH),
                         help="Chemin vers le CSV des données empiriques")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument('-N', '--n_classes', type=int, default=40)             # nb de catégories du modèle
    parser.add_argument('-ni', '--initial_pop', type=int, default=1000)        # taille de population initiale
    parser.add_argument('-nf', '--final_pop', type=int, default=2000)          # taille de population finale
    parser.add_argument('-alpha', '--archive_rate', type=float, default=0.05)  # fraction archivée par génération
    parser.add_argument('-T', '--generations', type=int, default=1000)         # nb de générations simulées
    parser.add_argument('-t', '--window', type=int, default=300)               # (non utilisé pour l'instant)
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('-q','--conformity_bias', type=float, default=1)       #Biais de conformité
    parser.add_argument('--no_archive', action='store_true',
                         help="Désactive l'archivage cumulatif : retourne uniquement la POPULATION "
                              "FINALE (génération t_max-1), sans accumuler sur les générations "
                              "précédentes. Par défaut, l'archivage est activé (comportement "
                              "historique). Attention : --archive_rate=1.0 (100%%) n'est PAS "
                              "équivalent à --no_archive -- même à 100%%, l'archive cumulative "
                              "reste la somme de TOUTES les générations, pas juste la dernière.")
    parser.add_argument('--distribution', choices=['power_law', 'uniform', 'random', 'custom'],
                         default='power_law',
                         help="Forme de la distribution initiale de la population à t=0 (défaut : "
                              "power_law = rang^-1, comportement historique). 'uniform' = toutes "
                              "les classes équiprobables. 'random' = composition aléatoire "
                              "(Dirichlet), reproductible via --seed. 'custom' nécessite "
                              "--distribution_path.")
    parser.add_argument('--distribution_path', default=None,
                         help="Chemin d'un CSV à une colonne (une valeur -- proportion ou compte -- "
                              "par ligne, n_classes lignes) utilisé comme distribution initiale "
                              "avec --distribution custom")
    parser.add_argument('--n_runs', type=int, default=1,
                         help="Nombre de runs canoniques à lancer/moyenner (1 = run unique)")
    parser.add_argument('--validate', action='store_true',
                         help="Lance en plus un run de validation à la résolution empirique (n=176)")
    parser.add_argument('--n_classes_valid', type=int, default=176)
    parser.add_argument('-o', '--output', default=None,
                         help="Chemin d'un dossier où sauvegarder les métriques de ce run "
                              "(fichier nommé n{n_classes}_alpha{archive_rate}_seed{seed}.json ; "
                              "en mode --sweep_param, sauvegarde plutôt sweep_<param>.csv)")
    parser.add_argument('--sweep_param',
                         choices=["archive_rate", "conformity_bias", "n_classes", "initial_pop",
                                  "final_pop", "generations"],
                         default=None,
                         help="Nom du paramètre à faire varier (sweep local, en mémoire). "
                              "Remplace le mode run/validate normal ; à utiliser avec --sweep_values. "
                              "Les autres paramètres restent fixes à leur valeur CLI (ex: --distribution, "
                              "--no_archive, --conformity_bias si ce n'est pas lui le paramètre balayé).")
    parser.add_argument('--sweep_values', type=float, nargs='+', default=None,
                         help="Liste des valeurs à tester pour --sweep_param (ex: 0.01 0.05 0.1 0.2)")
    parser.add_argument('--sweep_repeats', type=int, default=1,
                         help="Répète chaque valeur du sweep ce nombre de fois et moyenne les "
                              "métriques (+ écart-type), pour distinguer un vrai effet du bruit "
                              "d'un tirage unique (défaut 1 = pas de répétition)")
    parser.add_argument('--compare_empirical', action='store_true',
                         help="Affiche, à chaque run (unique, --n_runs, --validate, ou chaque ligne "
                              "d'un --sweep_param), la valeur empirique et l'écart (Δ = simulé - "
                              "empirique) juste à côté de chaque métrique, plutôt que seulement une "
                              "fois en tête de sortie.")
    parser.add_argument('--plot', default=None,
                         help="Chemin d'un fichier image (.png) où sauvegarder une figure de comparaison "
                              "empirique/simulé (rang-fréquence en mode run, métriques vs paramètre en mode sweep)")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # --- Distribution initiale personnalisée (--distribution custom) ---
    custom_probs = None
    if args.distribution == "custom":
        if not args.distribution_path:
            parser.error("--distribution custom nécessite --distribution_path")
        custom_probs = np.loadtxt(args.distribution_path, delimiter=",").reshape(-1)
        print(f"\nDistribution initiale personnalisée chargée : {args.distribution_path} "
              f"({len(custom_probs)} valeurs)")

    # --- Données empiriques : chargement + métriques de diversité ---
    df, counter, emp_freq_all = load_empirical_data(args.input)
    emp_freq, emp_ranks = summarize_empirical_data(df, emp_freq_all, N_TOP)
    print_diversity_metrics(emp_freq_all, label="Données empiriques")
    # Calculé une seule fois, réutilisé partout où --compare_empirical demande une comparaison
    # (run unique, --n_runs, --validate, chaque ligne de --sweep_param).
    emp_metrics = compute_diversity_metrics(emp_freq_all)
    compare_to = emp_metrics if args.compare_empirical else None

    # Paramètres du run courant, réutilisés pour construire l'enregistrement de résultat
    params = {
        "n_classes": args.n_classes,
        "initial_pop": args.initial_pop,
        "final_pop": args.final_pop,
        "generations": args.generations,
        "archive_rate": args.archive_rate,
        "seed": args.seed,
        "conformity_bias" : args.conformity_bias,
        "distribution": args.distribution,
        "archive": not args.no_archive,
    }

    # ── Mode sweep : fait varier UN paramètre, en mémoire, et compare à l'empirique ───────────
    # Remplace entièrement le mode run/validate normal ci-dessous (ne se combine pas avec
    # --n_runs/--validate ; chaque valeur du sweep est un run canonique simple).
    if args.sweep_param:
        if args.validate:
            print("\n[!] --validate est ignoré en mode sweep (--sweep_param) : "
                  "chaque point du sweep est un run canonique simple, pas de run n=176.")
        values = args.sweep_values
        if args.sweep_param not in ("archive_rate", "conformity_bias"):
            values = [int(v) for v in values]  # les autres paramètres balayables sont des entiers
        repeats_note = f" (x{args.sweep_repeats} répétitions, moyennées)" if args.sweep_repeats > 1 else ""
        print(f"\nSweep sur '{args.sweep_param}' : {values}{repeats_note} "
              f"(distribution={args.distribution}, archive={not args.no_archive})")
        sweep_df, sweep_raw_df = sweep_parameter(
            args.sweep_param, values, rng,
            args.n_classes, args.initial_pop, args.final_pop, args.generations, args.archive_rate,
            n_repeats=args.sweep_repeats, distribution=args.distribution, custom_probs=custom_probs,
            archive=not args.no_archive, conformity_bias=args.conformity_bias
        )
        if compare_to is not None:
            # Une colonne d'écart (Δ = simulé - empirique) par métrique, ajoutée juste après la
            # colonne elle-même -- comparer chaque ligne du sweep à l'empirique sans devoir se
            # référer au print initial des données empiriques, ni tracer --plot pour le voir.
            for metric, emp_value in compare_to.items():
                if metric in sweep_df.columns:
                    sweep_df.insert(sweep_df.columns.get_loc(metric) + 1,
                                     f"{metric}_vs_emp", sweep_df[metric] - emp_value)
        print(sweep_df.to_string(index=False))

        if args.output:
            # Un sweep produit un tableau, pas un dict unique -> CSV plutôt que le JSON du mode run
            csv_path = Path(args.output) / f"sweep_{args.sweep_param}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            sweep_df.to_csv(csv_path, index=False)
            print(f"\nRésultat(s) sauvegardé(s) : {csv_path}")

            if args.sweep_repeats > 1:
                # Tous les runs individuels, sans moyenne (sinon identique à sweep_<param>.csv,
                # pas la peine de dupliquer le fichier)
                raw_csv_path = Path(args.output) / f"sweep_{args.sweep_param}_raw.csv"
                sweep_raw_df.to_csv(raw_csv_path, index=False)
                print(f"Runs individuels sauvegardés : {raw_csv_path}")

        if args.plot:
            plot_sweep_metrics(sweep_df, args.sweep_param, emp_metrics, args.plot)
        return   # le mode sweep s'arrête ici, il ne passe pas par le run/validate normal

    # --- Run(s) canonique(s) (paramètres papier, n=N_CLASSES) ---
    if args.n_runs > 1:
        print(f"\nRunning {args.n_runs} canonical simulations (n={args.n_classes})...")
        t0 = time.perf_counter()
        all_freqs = run_multiple_simulations(
            rng, args.n_runs, args.n_classes, args.initial_pop, args.final_pop,
            args.generations, args.archive_rate, conformity_bias=args.conformity_bias,
            distribution=args.distribution, custom_probs=custom_probs, archive=not args.no_archive
        )
        runtime_s = (time.perf_counter() - t0) / args.n_runs   # temps moyen par run
        # Print comparatif par simulation (TODO historique)
        results = []
        for i, freq in enumerate(all_freqs, 1):
            print_diversity_metrics(freq, label=f"Run {i}/{args.n_runs} (n={args.n_classes})",
                                     compare_to=compare_to)
            results.append(make_result_record({**params, "run": i, "runtime_s": runtime_s}, freq))

        # Visuel façon Figure 6 de analysis_stats.py : moyenne + bande 5e-95e percentile sur les
        # n_runs, vs empirique (Top-N, même logique que le plot rang-fréquence en mode run unique)
        if args.plot:
            plot_rank_frequency_band(emp_freq, all_freqs, args.plot,
                                      label_model=f"Simulation (n={args.n_classes})")

        # Distributions brutes des n_runs, à part du JSON de métriques (résolution du besoin
        # "conserver les distributions à la fin" : rejouer une analyse sans relancer les sims)
        if args.output:
            dist_filename = f'n{args.n_classes}_alpha{args.archive_rate:.3f}_seed{args.seed}_distributions.csv'
            save_distributions(all_freqs, Path(args.output) / dist_filename)
    else:
        t0 = time.perf_counter()
        archive, model_freq, model_ranks = run_canonical(
            rng, args.n_classes, args.initial_pop, args.final_pop,
            args.generations, args.archive_rate, verbose=args.verbose,
            conformity_bias=args.conformity_bias, distribution=args.distribution,
            custom_probs=custom_probs, archive=not args.no_archive
        )
        runtime_s = time.perf_counter() - t0
        print_diversity_metrics(model_freq, label=f"Simulation (n={args.n_classes})",
                                 compare_to=compare_to)
        print(f"  Durée du run : {runtime_s:.3f}s")
        results = make_result_record({**params, "runtime_s": runtime_s}, model_freq)

        # Visuel rang-fréquence (empirique Top-N vs modèle canonique) : seulement en mode run
        # unique (--n_runs=1), car en mode --n_runs il n'y a pas UNE distribution modèle à tracer
        # (cf. TODO en tête de fichier : moyenne + percentiles sur plusieurs runs, pas encore fait)
        if args.plot:
            plot_rank_frequency(emp_freq, model_freq, args.plot,
                                 label_model=f"Simulation (n={args.n_classes})")

    # --- Run de validation optionnel (n=176, initialisé sur les proportions empiriques) ---
    if args.validate:
        t0 = time.perf_counter()
        _, valid_freq, _ = run_validation(
            rng, emp_freq_all, args.n_classes_valid, args.initial_pop, args.final_pop,
            args.n_classes, args.generations, args.archive_rate, verbose=args.verbose,
            conformity_bias=args.conformity_bias, archive=not args.no_archive
        )
        validation_runtime_s = time.perf_counter() - t0
        valid_freq_nonzero = valid_freq[valid_freq > 0]
        print_diversity_metrics(valid_freq_nonzero, label=f"Validation (n={args.n_classes_valid})",
                                 compare_to=compare_to)
        print(f"  Durée du run : {validation_runtime_s:.3f}s")
        validation_record = make_result_record(
            {**params, "n_classes_valid": args.n_classes_valid, "runtime_s": validation_runtime_s},
            valid_freq_nonzero
        )
        # Rattache le résultat de validation à chaque enregistrement du run canonique
        targets = results if isinstance(results, list) else [results]
        for r in targets:
            r["validation"] = validation_record

        # Deuxième figure dédiée à la validation (résolution n=176, comparable à l'empirique complet) :
        # suffixée "_validation" pour ne pas écraser le plot du run canonique au même --plot
        if args.plot:
            plot_path = Path(args.plot)
            validation_plot_path = plot_path.with_name(f"{plot_path.stem}_validation{plot_path.suffix}")
            plot_rank_frequency(emp_freq_all, valid_freq_nonzero, validation_plot_path,
                                 label_model=f"Validation (n={args.n_classes_valid})")

    # --- Sauvegarde optionnelle (1 fichier JSON pour ce run) ---
    if args.output:
        filename = f'n{args.n_classes}_alpha{args.archive_rate:.3f}_seed{args.seed}.json'
        save_results(results, Path(args.output) / filename)


if __name__ == '__main__':
    main()
