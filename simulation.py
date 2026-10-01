"""
Nouvelle implémentation du fichier simulation.py, pour l'adapter à un nouveau modèle. 
Simplifie l'implémentation précédente, en gardant uniquement les éléments nécessaires (modification du sweep pour le faire uniquement aléatoirement...)
On retire le chargement des données empiriques brutes. 
On garde la possibilité de charger un csv de paramètres (ou le génère depuis ici directement)

Description du modèle : 

On commence avec population initiale (ie. le dump originel) $p_i$, où chaque individu de la population se voit attribuer un tag parmis les $C$ disponibles au départ.
(On envisage a (défaut , 0), de tel sorte à ce que la distribution de tag soit plus ou moins proche d'une loi de puissance)
À t=-1, on ajoute la population initiale à un ensemble ARCHIVE. 
À chaque pas de temps t : 
- On ajoute n_t individus, de sorte que la taille de l'ARCHIVE passe linéairement de $p_i$ à $p_f$ (par défaut 10000) en T générations : n_t ≈ ($p_f$ - $p_i$)/T par génération, et l'archive contient exactement $p_f$ individus à la fin
- Une proportion aléatoire µ de la nouvelle génération n se voit attribué une nouvelle classe. 
- Les individus n(1-µ) se voit attribué une classe c parmis C en suivant la distribution de l'ensemble ARCHIVE, à la puissance q (biais de conformité)
- On ajoute la nouvelle distribution entièrement à ARCHIVE

Après les T générations, on calcule : 
- coefficient de Gini (gini : toutes les classes, y compris de compte 0 ; gini_obs : classes observées seulement)
- Nombre de Hill (à tous les ordres) (en évitant le 0, surement moins de classes)
- Chao1 (mesure des classes les pus rares)
- Divergence de Kullback-Leibler KL(empirique || simulation) entre distributions rang-fréquence



TODO : 
- [fait] Classe python exportable (Simulation) 
- [fait] Créer une fonction qui génère une table de simulation (comme le fichier generate grid)
    - Prend deux paramètres : un flag HPC, un nombre de simulation
    - Pas un sweep comme la simulation précédente => Table générée directement, sans prendre d'intervalle en arguments, juste le nombre de simulation
- [fait] Ajouter une option --plot : 
    - Si flag présent => conserver l'évolution de l'archive et plot 



"""


import argparse
import csv
import functools
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor


# ============================================================================
# MÉTRIQUES DE DIVERSITÉ
# ============================================================================
def gini(x):
    """
    Indice de Gini d'un vecteur de fréquences/abondances par classe.

    Mesure l'inégalité de répartition : 0 = toutes les classes ont la même fréquence,
    valeur proche de 1 = quasiment tous les individus dans une seule classe.
    Calculé via la formule sur vecteur trié, sum((2i - n - 1) * x_i) / (n * sum(x)), avec i le rang
    (1..n) : équivalent à la moyenne des différences absolues par paires, mais en O(n log n)
    au lieu de O(n²).

    Paramètres :
    - x : vecteur (array-like) de fréquences ou de comptes par classe, >= 0. Les classes de
          fréquence 0 sont conservées et comptent dans n (elles augmentent le Gini).

    Retour :
    - float dans [0, 1[ ; np.nan si x est entièrement nul (Gini non défini, ex. archive jamais
      peuplée) plutôt que de laisser numpy lever un RuntimeWarning sur une division 0/0.
    """
    if np.mean(x) == 0:
        return np.nan
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * x.sum())

def hill_number(x, order):
    """
    Nombre de Hill d'ordre `order` : nombre "effectif" de classes équiprobables qui donnerait
    la même diversité que la distribution observée.

    Avec p_i = x_i / sum(x) : D_q = (sum p_i^q)^(1/(1-q)). Cas particuliers :
    - order=0      : richesse (nombre de classes présentes)
    - order=1      : exp(entropie de Shannon) (limite de la formule, traitée à part)
    - order=2      : inverse de l'indice de Simpson
    - order=np.inf : inverse de Berger-Parker (1 / proportion de la classe dominante)
    Plus l'ordre est grand, plus la mesure pondère les classes fréquentes (et ignore les rares).
    Toujours D_0 >= D_1 >= D_2 >= ... >= D_inf.

    Paramètres :
    - x     : vecteur (array-like) de fréquences ou d'abondances par classe, >= 0 ; les classes
              de fréquence 0 sont ignorées.
    - order : nombre réel >= 0 (int, float ou np.inf), pas seulement entier (ex. 1.5 est légitime).

    Retour :
    - float ; np.nan si x ne contient aucune classe présente (cas limite légitime, pas d'erreur).

    Lève :
    - ValueError si `order` n'est pas un réel >= 0.
    """
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
        # Cas limite légitime (ex. archive_rate proche de 0 -> archive cumulative jamais
        # peuplée) plutôt qu'une erreur d'appel -> diversité non définie, nan (pas de raise :
        # ça ferait planter tout un run_param_grid.py sur une seule ligne de grille dégénérée).
        return np.nan
    x_prop = x/x.sum()

    if order == 1 :
        return np.exp(-((np.log(x_prop)*x_prop).sum()))
    elif order == np.inf:
        return 1/x_prop.max()
    else :
        return (x_prop**order).sum()**(1/(1-order))



def chao1(x):
    """
    Estimateur Chao1 : borne inférieure de la richesse réelle (nombre de classes, y compris
    celles non observées), fondée sur les classes les plus rares.

    Avec S_obs le nombre de classes observées, f1 le nombre de singletons (classes d'abondance 1)
    et f2 le nombre de doubletons (abondance 2) :
    - f2 > 0  : S_obs + f1² / (2 * f2)
    - f2 == 0 : S_obs + f1 * (f1 - 1) / 2  (version corrigée du biais)
    Attention : l'estimation explose quand f2 est très petit devant f1 (beaucoup de singletons,
    quasi aucun doubleton), typiquement avec un fort biais de conformité.

    Paramètres :
    - x : vecteur (array-like) d'abondances (comptes entiers) par classe. Ce sont des comptes
          et non des labels individuels ; les classes d'abondance 0 sont ignorées.

    Retour :
    - float : richesse estimée (>= nombre de classes observées).
    """
    counts = np.asarray(x)
    counts = counts[counts > 0]
    s_obs = counts.size
    f1 = np.count_nonzero(counts == 1)
    f2 = np.count_nonzero(counts == 2)
    if f2 > 0:
        return s_obs + f1**2 / (2 * f2)
    return s_obs + f1 * (f1 - 1) / 2



# ============================================================================
# DIVERGENCE À L'EMPIRIQUE
# ============================================================================
# Comptes empiriques par catégorie (data/fandom_data.csv : 13 128 histoires, catégories combinées
# "A|B" éclatées ; 176 catégories, 30 901 occurrences), triés par ordre décroissant.
EMPIRICAL_COUNTS = (
    3158, 2698, 1688, 1269, 1263, 1244, 1198, 1090, 1051, 1043, 975, 932, 764, 724, 621, 587, 557, 542, 540, 488, 462, 401,
    313, 309, 290, 281, 272, 264, 243, 237, 216, 199, 190, 181, 178, 171, 161, 155, 144, 141, 134, 133, 131, 114,
    113, 112, 108, 100, 100, 98, 96, 86, 79, 69, 68, 66, 58, 53, 52, 49, 48, 47, 46, 45, 43, 41,
    40, 36, 35, 34, 33, 33, 33, 33, 32, 31, 29, 29, 29, 28, 28, 28, 28, 26, 26, 25, 24, 24,
    23, 23, 23, 21, 21, 20, 20, 20, 20, 20, 19, 19, 18, 18, 17, 17, 17, 16, 16, 16, 16, 16,
    15, 15, 15, 15, 15, 15, 15, 15, 14, 14, 13, 13, 13, 12, 12, 12, 12, 11, 11, 11, 11, 11,
    11, 11, 10, 10, 10, 9, 9, 8, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 5, 4, 4,
    4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
)


def kl_divergence(counts, reference=None, floor=0.5):
    """
    Divergence de Kullback-Leibler KL(P || Q), en nats, entre la distribution rang-fréquence de
    référence P (empirique par défaut) et celle de la simulation Q.

    Les classes n'ont pas d'identité commune entre simulation et données : on compare les
    distributions PAR RANG. P(r) = count_ref(r) / total_ref pour les rangs r = 1..R de la référence
    (R = 176 pour l'empirique) ; Q(r) = count_sim(r) / Z pour les mêmes rangs, les comptes simulés étant
    triés par ordre décroissant. Comme des fréquences relatives sont comparées, les tailles
    d'échantillon différentes (10 000 simulés contre 30 901 empiriques) ne biaisent pas directement.

    Deux précautions rendent Q propre (somme 1) et la divergence finie :
    - un rang que la simulation n'atteint pas (moins de R classes) reçoit un plancher `floor`
      (0.5 occurrence) au lieu de 0 : sinon KL = +inf ;
    - la masse simulée au-delà du rang R (classes que la référence n'a pas) reste dans la
      normalisation Z : la simulation qui répartit des occurrences sur trop de classes est pénalisée.

    Paramètres :
    - counts    : comptes par classe de la simulation (array-like, ordre quelconque, zéros admis)
    - reference : comptes de référence (défaut EMPIRICAL_COUNTS)
    - floor     : plancher (en occurrences) des rangs non atteints

    Retour : float >= 0 (0 = distributions identiques par rang) ; np.nan si `counts` est vide/nul.
    """
    ref = np.asarray(EMPIRICAL_COUNTS if reference is None else reference, dtype=float)
    ref = np.sort(ref)[::-1]
    p = ref / ref.sum()
    c = np.sort(np.asarray(counts, dtype=float))[::-1]
    if c.sum() == 0:
        return np.nan
    r = len(ref)
    head = np.zeros(r)
    head[:min(r, len(c))] = c[:r]
    tail_mass = c[r:].sum()
    q = np.maximum(head, floor) / (np.maximum(head, floor).sum() + tail_mass)
    return float(np.sum(p * np.log(p / q)))


def compute_metrics(counts):
    """
    Calcule toutes les métriques de diversité sur un vecteur de comptes par classe.

    Paramètre :
    - counts : vecteur (array-like) d'abondances par classe.

    Retour : dict {"gini", "gini_obs", "hill_0", "hill_1", "hill_2", "hill_inf", "chao1", "n_classes", "kl_emp"}
    (n_classes = nombre de classes de compte > 0, en int ; kl_emp = KL(empirique || simulation) par
    rang, cf. `kl_divergence`). "gini" compte les classes de compte 0 (classes initiales jamais
    échantillonnées) ; "gini_obs" ne porte que sur les classes observées (compte > 0), comme
    l'empirique où toutes les catégories sont observées : c'est lui qu'il faut comparer aux données.
    """
    counts = np.asarray(counts)
    return {
        "gini": gini(counts),
        "gini_obs": gini(counts[counts > 0]),
        "hill_0": hill_number(counts, 0),
        "hill_1": hill_number(counts, 1),
        "hill_2": hill_number(counts, 2),
        "hill_inf": hill_number(counts, np.inf),
        "chao1": chao1(counts),
        "n_classes": int((counts > 0).sum()),
        "kl_emp": kl_divergence(counts),
    }


# ============================================================================
# CROISSANCE DE L'ARCHIVE
# ============================================================================
GROWTHS = ("linear", "empirical")

# Nombre d'occurrences de catégories déposées chaque semaine dans le corpus (708 semaines, du 2010-08-08 au
# 2024-02-25 ; somme = 30 901). Recalculé depuis data/fandom_data.csv par scripts/empirical_counts.py.
EMPIRICAL_WEEKLY_ADDITIONS = (
    73, 91, 344, 17, 16, 9, 4, 26, 40, 11, 4, 0, 3, 275, 25, 20, 13, 39, 34, 56, 9, 49,
    29, 23, 42, 22, 18, 16, 52, 28, 61, 33, 129, 238, 71, 8, 31, 189, 70, 31, 12, 16, 30, 16,
    29, 40, 38, 94, 40, 54, 32, 35, 41, 37, 31, 60, 60, 59, 20, 28, 35, 18, 43, 48, 49, 59,
    42, 63, 39, 40, 84, 41, 40, 94, 63, 35, 67, 82, 63, 28, 64, 74, 96, 90, 86, 116, 199, 121,
    189, 124, 72, 186, 94, 149, 74, 89, 65, 91, 92, 70, 87, 91, 51, 67, 40, 75, 81, 59, 40, 40,
    50, 65, 66, 92, 65, 76, 87, 78, 101, 67, 104, 78, 115, 83, 67, 107, 165, 82, 94, 77, 80, 106,
    72, 127, 83, 107, 98, 60, 77, 132, 115, 64, 118, 82, 97, 111, 79, 84, 104, 68, 85, 75, 56, 100,
    114, 80, 118, 82, 123, 108, 122, 81, 76, 76, 96, 139, 132, 133, 209, 137, 95, 104, 99, 102, 153, 73,
    56, 50, 82, 67, 89, 58, 33, 45, 75, 65, 80, 99, 69, 72, 69, 57, 77, 65, 62, 64, 93, 76,
    104, 120, 120, 108, 115, 120, 106, 78, 90, 96, 107, 120, 110, 107, 142, 82, 41, 89, 106, 96, 46, 47,
    58, 66, 47, 34, 61, 66, 26, 20, 39, 58, 62, 45, 81, 86, 56, 112, 59, 106, 68, 73, 56, 64,
    73, 93, 83, 84, 59, 74, 66, 65, 69, 95, 72, 63, 108, 61, 132, 119, 62, 48, 53, 55, 60, 68,
    54, 40, 34, 23, 30, 17, 49, 31, 24, 33, 32, 10, 22, 11, 6, 18, 41, 19, 25, 20, 38, 34,
    20, 20, 27, 39, 8, 11, 16, 32, 19, 22, 30, 16, 20, 26, 28, 25, 20, 18, 25, 21, 14, 20,
    18, 18, 28, 20, 9, 15, 18, 11, 17, 15, 14, 32, 33, 21, 8, 15, 28, 19, 24, 24, 19, 18,
    28, 29, 40, 51, 20, 24, 19, 21, 24, 40, 52, 34, 20, 28, 35, 46, 32, 12, 39, 17, 22, 9,
    24, 18, 14, 19, 45, 22, 12, 37, 21, 28, 18, 12, 21, 58, 25, 69, 27, 55, 36, 57, 75, 95,
    99, 38, 34, 69, 20, 29, 20, 21, 41, 26, 44, 38, 12, 76, 39, 43, 11, 15, 31, 70, 36, 38,
    29, 31, 19, 41, 40, 45, 61, 76, 62, 55, 38, 54, 49, 50, 67, 112, 54, 60, 68, 47, 31, 65,
    70, 95, 26, 77, 43, 40, 36, 31, 48, 45, 42, 29, 13, 41, 17, 16, 17, 22, 11, 42, 28, 15,
    2, 12, 16, 6, 22, 20, 16, 15, 10, 6, 6, 11, 8, 15, 7, 14, 21, 27, 16, 31, 63, 28,
    26, 19, 32, 21, 8, 13, 16, 22, 21, 21, 11, 10, 25, 11, 37, 25, 16, 6, 19, 20, 30, 4,
    25, 25, 20, 27, 20, 21, 19, 5, 5, 23, 26, 23, 21, 43, 11, 10, 22, 10, 19, 16, 25, 38,
    47, 33, 52, 59, 29, 57, 26, 36, 33, 26, 30, 23, 18, 24, 16, 22, 30, 43, 32, 15, 21, 24,
    17, 33, 19, 23, 32, 22, 34, 36, 34, 25, 24, 20, 23, 26, 11, 26, 16, 15, 12, 27, 21, 19,
    17, 30, 25, 35, 26, 26, 19, 12, 43, 34, 21, 47, 32, 4, 34, 8, 20, 12, 11, 6, 27, 5,
    7, 15, 12, 11, 28, 10, 18, 6, 30, 25, 15, 37, 44, 17, 58, 14, 10, 29, 15, 35, 39, 36,
    15, 20, 12, 25, 18, 16, 23, 13, 33, 18, 26, 11, 20, 10, 8, 17, 20, 22, 37, 11, 11, 15,
    15, 10, 20, 4, 12, 0, 7, 4, 2, 7, 5, 6, 8, 10, 46, 30, 13, 19, 15, 15, 12, 9,
    18, 27, 18, 64, 35, 17, 12, 8, 15, 3, 16, 6, 11, 7, 9, 9, 11, 12, 13, 14, 20, 17,
    26, 25, 15, 34, 24, 0, 20, 9, 17, 19, 15, 12, 13, 14, 19, 17, 15, 26, 19, 9, 23, 34,
    77, 9, 6, 7, 13, 25, 10, 9, 73, 9, 13, 16, 15, 30, 14, 23, 8, 8, 10, 5, 5, 0,
    17, 12, 3, 7,
)
_weekly = np.asarray(EMPIRICAL_WEEKLY_ADDITIONS, dtype=float)
_EMP_U = np.arange(len(_weekly) + 1) / len(_weekly)                      # temps normalisé [0, 1]
_EMP_F = np.r_[0.0, np.cumsum(_weekly)] / _weekly.sum()                  # part cumulée des occurrences déposées


def growth_sizes(n_i, n_f, T, growth="linear"):
    """
    Taille de l'archive après chaque génération t = 0..T (tableau d'entiers de longueur T + 1 ; sizes[0] = n_i,
    sizes[T] = n_f). Les ajouts par génération sont n_t = np.diff(sizes).

    - "linear"    : croissance linéaire, n_t constant (à l'arrondi près).
    - "empirical" : croissance suivant la répartition empirique des dépôts dans le temps : la part cumulée
                    d'occurrences F(u) du corpus (résolution hebdomadaire, interpolée linéairement) à l'instant
                    normalisé u = t / T, soit sizes[t] = n_i + (n_f - n_i) * F(t / T). T reste libre (nombre de
                    générations) : la forme de la croissance est celle des données (rapide en 2012-2015, lente
                    après 2016), pas son échelle de temps.

    Paramètres :
    - n_i, n_f : taille initiale et finale de l'archive (n_f >= n_i)
    - T        : nombre de générations
    - growth   : "linear" ou "empirical"
    """
    if growth == "linear":
        return np.rint(np.linspace(n_i, n_f, T + 1)).astype(np.int64)
    if growth == "empirical":
        F = np.interp(np.linspace(0.0, 1.0, T + 1), _EMP_U, _EMP_F)
        return np.rint(n_i + (n_f - n_i) * F).astype(np.int64)
    raise ValueError(f"growth doit valoir {GROWTHS}, reçu {growth!r}")


# ============================================================================
# SIMULATION
# ============================================================================
class Simulation:
    """
    Modèle de transmission culturelle avec archive cumulative (voir docstring du module).

    Usage :
        sim = Simulation(n_i=1000, n_f=10000, C=50, T=100, alpha=0, mu=0.01, q=1, seed=0).run()
        sim.archive   # comptes cumulés par classe (archive finale)
        sim.metrics   # dict de métriques (cf. `compute_metrics`)

    Déroulé de `run` :
    1. Population initiale : n_i individus répartis sur C classes selon une loi de puissance
       (probabilité ∝ rang^-alpha), tirage multinomial. Elle constitue l'ARCHIVE de départ.
    2. À chaque génération t (T au total), n_t individus sont ajoutés à l'archive. Avec growth="linear",
       n_t ≈ (n_f - n_i) / T (arrondi par cumul : la taille de l'archive après la génération t est
       round(n_i + (n_f - n_i) * t / T)) ; avec growth="empirical", les ajouts suivent la répartition
       empirique des dépôts dans le temps (cf. `growth_sizes`). Dans les deux cas l'archive contient
       exactement n_f individus à la fin :
       - k ~ Binomial(n_t, mu) innovateurs : chacun crée sa propre nouvelle classe (compte 1) ;
       - les n_t - k autres copient une classe existante, tirée avec une probabilité ∝ archive[c]**q
         (tirage multinomial sur les classes déjà présentes) ;
       - toute la génération est ajoutée à l'archive. Les classes créées à la génération t ne
         sont copiables qu'à partir de la génération t+1.

    Paramètres :
    - n_i   : taille de la population initiale (dump originel)
    - n_f   : taille FINALE de l'archive (nombre total d'individus après T générations ; >= n_i)
    - C     : nombre de classes initiales (tags)
    - T     : nombre de générations
    - alpha : exposant de la loi de puissance de la distribution initiale (0 = uniforme sur les C
              classes ; plus alpha est grand, plus la distribution initiale est inégale)
    - mu    : probabilité d'innovation, dans [0, 1] ; nb d'innovateurs ~ Binomial(n_t, mu)
    - q     : biais de conformité ; P(classe c) ∝ archive[c]**q (1 = proportionnel à la fréquence,
              >1 conformisme : les classes fréquentes sont favorisées, <1 anti-conformisme,
              0 = uniforme sur les classes existantes)
    - seed  : graine du générateur aléatoire (np.random.default_rng) pour la reproductibilité ;
              None = non reproductible
    - growth : profil de croissance de l'archive : "linear" (défaut, n_t constant) ou "empirical" (ajouts
               suivant la répartition empirique des dépôts, cf. `growth_sizes`)

    Attributs (None avant `run`) :
    - archive : np.ndarray d'entiers, comptes cumulés par classe dans l'archive finale
                (taille C + nombre total d'innovateurs ; somme = n_f)
    - metrics : dict de métriques calculées sur `archive`
    - history : None, ou (si run(track_history=True)) dict de np.ndarray de longueur T + 1
                {"t", "gini", "hill_1", "hill_2", "n_classes"} ; l'indice 0 = archive initiale,
                l'indice t = archive après la génération t
    """

    def __init__(self, n_i, n_f, C, T, alpha, mu, q, seed=None, growth="linear"):
        if growth not in GROWTHS:
            raise ValueError(f"growth doit valoir {GROWTHS}, reçu {growth!r}")
        self.n_i, self.n_f, self.C, self.T = n_i, n_f, C, T
        self.alpha, self.mu, self.q, self.seed, self.growth = alpha, mu, q, seed, growth
        self.archive = None
        self.metrics = None
        self.history = None

    def run(self, track_history=False):
        """
        Exécute la simulation ; remplit `archive`, `metrics` (et `history` si demandé).

        Paramètre :
        - track_history : si True, enregistre gini / hill_1 / hill_2 / n_classes de l'archive à
                          chaque génération (coût : un tri de l'archive par génération, à réserver
                          aux runs individuels, pas aux sweeps massifs).

        Retour : self (permet `Simulation(...).run()`).
        """
        rng = np.random.default_rng(self.seed)
        C, T = self.C, self.T

        # On fixe la répartition initiale des C classes :
        ranks = np.arange(1, C + 1, dtype=float)
        probs_class = ranks ** -self.alpha
        probs_class /= probs_class.sum()

        # Tailles de génération : l'archive passe de n_i à n_f (n_f exactement à la fin) selon le profil `growth`
        if self.n_f < self.n_i:
            raise ValueError(f"n_f ({self.n_f}) doit être >= n_i ({self.n_i})")
        n_t = np.diff(growth_sizes(self.n_i, self.n_f, T, self.growth))
        n_inno = rng.binomial(n_t, self.mu)

        # ARCHIVE : vecteur de comptes par classe (pré-alloué : C classes initiales + 1 par innovateur)
        archive = np.zeros(C + n_inno.sum(), dtype=np.int64)
        archive[:C] = rng.multinomial(self.n_i, probs_class)
        n_classes = C

        hist = {k: np.zeros(T + 1) for k in ("t", "gini", "hill_1", "hill_2", "n_classes")} if track_history else None

        def record(t):
            counts = archive[:n_classes]
            hist["t"][t] = t
            hist["gini"][t] = gini(counts)
            hist["hill_1"][t] = hill_number(counts, 1)
            hist["hill_2"][t] = hill_number(counts, 2)
            hist["n_classes"][t] = (counts > 0).sum()

        if track_history:
            record(0)

        for t in range(T):
            k = n_inno[t]
            n_copy = n_t[t] - k

            # Copieurs : tirage selon la distribution de l'archive, à la puissance q (classes présentes uniquement)
            counts = archive[:n_classes]
            if n_copy > 0 and counts.sum() > 0:
                w = np.where(counts > 0, counts.astype(float) ** self.q, 0.0)
                new = rng.multinomial(n_copy, w / w.sum())
            else:
                new = np.zeros(n_classes, dtype=np.int64)

            # Toute la génération est ajoutée à l'archive (innovateurs : nouvelles classes de compte 1)
            # (les nouvelles classes sont ajoutées après le tirage : elles ne sont pas copiables dans la même génération)
            archive[:n_classes] += new
            archive[n_classes:n_classes + k] = 1
            n_classes += k

            if track_history:
                record(t + 1)

        self.archive = archive[:n_classes]
        self.metrics = compute_metrics(self.archive)
        self.history = hist
        return self


def simulation(n_i, n_f, C, T, alpha, mu, q, seed=None, growth="linear"):
    """
    Raccourci fonctionnel autour de `Simulation` : construit, exécute, et renvoie (archive, metrics).
    Voir la docstring de `Simulation` pour la signification des paramètres.

    Retour : tuple (final, metrics)
    - final   : np.ndarray d'entiers, comptes cumulés par classe dans l'archive finale
    - metrics : dict {"gini", "gini_obs", "hill_0", "hill_1", "hill_2", "hill_inf", "chao1", "n_classes", "kl_emp"}
    """
    sim = Simulation(n_i, n_f, C, T, alpha, mu, q, seed, growth).run()
    return sim.archive, sim.metrics


# ============================================================================
# TABLE DE SIMULATIONS
# ============================================================================
# Tirages uniformes indépendants par paramètre (une ligne = un tirage, pas de produit cartésien).
# Les bornes sont codées ici et non passées en arguments : à ajuster dans ces constantes.
FLOAT_RANGES = {"mu": (0.0, 0.05), "q": (0.5, 1.5), "alpha": (0.0, 2.0)}   # uniforme continu
INT_RANGES = {"T": (300, 1000), "C": (2, 200), "n_i": (1, 100)}             # uniforme entier (bornes incluses)
FIXED_PARAMS = {"n_f": 10000}                                               # constants pour toutes les lignes
PARAM_COLUMNS = ["C", "n_i", "n_f", "T", "alpha", "mu", "q", "seed"]


def draw_params(rng, n, float_ranges=None, int_ranges=None, fixed=None):
    """
    Tire n jeux de paramètres (hors graine de simulation) : uniforme continu dans `float_ranges`,
    uniforme entier (bornes incluses) dans `int_ranges`, constantes de `fixed`. Par défaut
    FLOAT_RANGES, INT_RANGES et FIXED_PARAMS.

    Paramètres :
    - rng          : np.random.Generator utilisé pour les tirages
    - n            : nombre de jeux de paramètres
    - float_ranges : {nom: (min, max)} (défaut FLOAT_RANGES)
    - int_ranges   : {nom: (min, max)} (défaut INT_RANGES)
    - fixed        : {nom: valeur} (défaut FIXED_PARAMS)

    Retour : dict {nom: np.ndarray de taille n} pour tous les paramètres de PARAM_COLUMNS sauf "seed".
    """
    float_ranges = FLOAT_RANGES if float_ranges is None else float_ranges
    int_ranges = INT_RANGES if int_ranges is None else int_ranges
    fixed = FIXED_PARAMS if fixed is None else fixed
    columns = {name: rng.uniform(min(b), max(b), size=n) for name, b in float_ranges.items()}  # min/max : tolère des bornes inversées
    columns.update({name: rng.integers(low, high, size=n, endpoint=True) for name, (low, high) in int_ranges.items()})
    columns.update({name: np.full(n, value) for name, value in fixed.items()})
    return columns


def generate_param_table(n_sims, rng_seed=0, seed_start=1):
    """
    Génère une table de paramètres : n_sims lignes tirées uniformément et indépendamment dans
    FLOAT_RANGES / INT_RANGES, les autres paramètres étant fixés par FIXED_PARAMS (cf. `draw_params`).

    Paramètres :
    - n_sims     : nombre de lignes (= nombre de simulations)
    - rng_seed   : graine du tirage des paramètres (même valeur -> même table)
    - seed_start : graine de simulation de la 1ère ligne, incrémentée de 1 par ligne

    Retour : liste de n_sims dicts, une clé par colonne de PARAM_COLUMNS.
    """
    columns = draw_params(np.random.default_rng(rng_seed), n_sims)

    rows = []
    for i in range(n_sims):
        row = {name: col[i].item() for name, col in columns.items()}
        row["seed"] = seed_start + i
        rows.append({c: row[c] for c in PARAM_COLUMNS})
    return rows


def _run_row(row, growth="linear"):
    """Exécute une ligne de paramètres et renvoie {paramètres + métriques} (fonction de module : picklable pour les workers)."""
    _, metrics = simulation(**row, growth=growth)
    return {**row, **metrics}


def run_param_table(rows, n_jobs=1, growth="linear"):
    """
    Exécute une simulation par ligne de `rows` (cf. `generate_param_table`).

    Paramètres :
    - rows   : liste de dicts de paramètres
    - n_jobs : nombre de workers en parallèle. 1 = séquentiel (pas de sous-processus) ;
               -1 = tous les cœurs (os.cpu_count()). Ce sont des processus et non des threads :
               la simulation est du calcul CPU numpy/Python, les threads seraient bridés par le GIL.
               Chaque ligne porte sa propre graine, donc le résultat ne dépend pas de n_jobs.
    - growth : profil de croissance de l'archive pour toutes les lignes ("linear" ou "empirical")

    Retour : liste de dicts {paramètres de la ligne + métriques}, dans l'ordre de `rows`.
    """
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if n_jobs < 1:
        raise ValueError(f"n_jobs doit être >= 1 ou -1, reçu {n_jobs}")
    if n_jobs == 1 or len(rows) <= 1:
        return [_run_row(row, growth) for row in rows]

    with ProcessPoolExecutor(max_workers=min(n_jobs, len(rows))) as pool:
        # map conserve l'ordre ; chunksize limite le coût de communication sur les grosses tables
        return list(pool.map(functools.partial(_run_row, growth=growth), rows, chunksize=max(1, len(rows) // (n_jobs * 4))))


def write_csv(rows, path):
    """Écrit une liste de dicts (mêmes clés) dans un CSV avec en-tête."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================================
# PLOT
# ============================================================================
def plot_simulation(sim, path):
    """
    Trace et sauvegarde deux panneaux pour une simulation exécutée avec track_history=True :
    - gauche : distribution rang-fréquence de l'archive finale (log-log)
    - droite : évolution de gini, hill_1, hill_2 et n_classes en fonction de la génération
      (gini sur un axe de droite, les autres en échelle log)

    Paramètres :
    - sim  : Simulation déjà exécutée avec run(track_history=True)
    - path : chemin du fichier image (ex. "out.png")
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if sim.history is None:
        raise ValueError("sim.history est vide : exécuter run(track_history=True)")
    h = sim.history

    fig, (ax_rank, ax_time) = plt.subplots(1, 2, figsize=(11, 4))

    freqs = np.sort(sim.archive[sim.archive > 0])[::-1] / sim.archive.sum()
    ax_rank.loglog(np.arange(1, freqs.size + 1), freqs, ".", markersize=3)
    ax_rank.set_xlabel("rang")
    ax_rank.set_ylabel("fréquence dans l'archive finale")
    ax_rank.set_title("Rang-fréquence")

    for key in ("hill_1", "hill_2", "n_classes"):
        ax_time.plot(h["t"], h[key], label=key)
    ax_time.set_yscale("log")
    ax_time.set_xlabel("génération")
    ax_time.set_ylabel("hill_1, hill_2, n_classes")
    ax_gini = ax_time.twinx()
    ax_gini.plot(h["t"], h["gini"], color="k", linestyle="--", label="gini")
    ax_gini.set_ylabel("gini")
    lines = ax_time.get_lines() + ax_gini.get_lines()
    ax_time.legend(lines, [l.get_label() for l in lines], loc="center right")
    ax_time.set_title("Métriques de l'archive au cours du temps")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================
def main():
    """
    Point d'entrée en ligne de commande, deux modes :

    1. Simulation unique (défaut) : affiche les métriques (une ligne `métrique: valeur`).
       Requiert --C, --T, --mu, --q ; optionnels : --n_i, --n_f, --alpha, --seed, --growth.
       --plot CHEMIN.png : conserve l'évolution de l'archive et sauvegarde la figure
       (`plot_simulation`).
           python simulation.py --C 50 --T 100 --mu 0.01 --q 1 --seed 0 --plot run.png

    2. Table de simulations (--n_sims N) : tire N combinaisons de paramètres (bornes codées dans
       FLOAT_RANGES / INT_RANGES / FIXED_PARAMS, cf. `generate_param_table`).
       - sans --hpc : exécute les N simulations en local (--n_jobs processus en parallèle,
         -1 = tous les cœurs) et écrit params + métriques dans -o (défaut results.csv)
       - avec --hpc : écrit uniquement la table de paramètres dans -o (défaut params.csv), sans
         rien exécuter (pour distribuer les lignes sur un cluster)
       --rng_seed fixe la graine du tirage des paramètres, --seed la graine de la 1ère ligne.
           python simulation.py --n_sims 1000 --hpc -o params.csv
           python simulation.py --n_sims 1000 --n_jobs -1 -o results.csv
    """
    parser = argparse.ArgumentParser(description="Simulation de transmission culturelle avec archive cumulative.")
    parser.add_argument("--n_i", type=int, default=1000, help="taille de la population initiale")
    parser.add_argument("--n_f", type=int, default=10000, help="taille finale de l'archive (nb d'individus à la fin)")
    parser.add_argument("--C", type=int, help="nombre de classes initiales")
    parser.add_argument("--T", type=int, help="nombre de générations")
    parser.add_argument("--alpha", default= 0, type=float, help="exposant de la loi de puissance initiale")
    parser.add_argument("--mu", type=float, help="probabilité d'innovation")
    parser.add_argument("--q", type=float, help="biais de conformité")
    parser.add_argument("--seed", type=int, default=None,
                        help="graine aléatoire (mode table : graine de la 1ère ligne, défaut 1)")
    parser.add_argument("--growth", choices=GROWTHS, default="linear",
                        help="profil de croissance de l'archive : linear (n_t constant, défaut) ou empirical "
                             "(ajouts suivant la répartition empirique des dépôts dans le temps)")
    parser.add_argument("--plot", metavar="CHEMIN", default=None,
                        help="sauvegarde la figure (rang-fréquence + métriques vs génération), simulation unique")
    parser.add_argument("--n_sims", type=int, default=None, help="mode table : nombre de simulations à générer")
    parser.add_argument("--hpc", action="store_true", help="mode table : écrit seulement la table de paramètres")
    parser.add_argument("--rng_seed", type=int, default=0, help="mode table : graine du tirage des paramètres")
    parser.add_argument("-o", "--output", default=None, help="mode table : CSV de sortie")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="mode table (sans --hpc) : nb de processus en parallèle, -1 = tous les cœurs (défaut 1)")
    args = parser.parse_args()

    # Mode table
    if args.n_sims is not None:
        if args.plot:
            parser.error("--plot n'est disponible qu'en simulation unique")
        rows = generate_param_table(args.n_sims, rng_seed=args.rng_seed,
                                    seed_start=args.seed if args.seed is not None else 1)
        if args.hpc:
            out = args.output or "params.csv"
            write_csv(rows, out)
            print(f"{len(rows)} lignes de paramètres écrites dans {out}")
        else:
            out = args.output or "results.csv"
            write_csv(run_param_table(rows, n_jobs=args.n_jobs, growth=args.growth), out)
            print(f"{len(rows)} simulations exécutées, résultats dans {out}")
        return

    # Simulation unique
    missing = [f"--{n}" for n in ("C", "T", "mu", "q") if getattr(args, n) is None]
    if missing:
        parser.error(f"arguments requis en simulation unique : {', '.join(missing)}")
    if args.hpc:
        parser.error("--hpc nécessite --n_sims")

    sim = Simulation(args.n_i, args.n_f, args.C, args.T, args.alpha, args.mu, args.q, seed=args.seed, growth=args.growth)
    sim.run(track_history=bool(args.plot))
    for k, v in sim.metrics.items():
        print(f"{k}: {v}")
    if args.plot:
        plot_simulation(sim, args.plot)
        print(f"figure sauvegardée dans {args.plot}")


if __name__ == "__main__":
    main()
