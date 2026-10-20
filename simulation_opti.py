## %
# =============================================================================
# simulation_opti.py — version optimisée de simulation.py (2026-10-19)
# Résultats IDENTIQUES bit à bit à simulation.py (même graine -> même d, C,
# historique) ; seules tirer_tags(), generation() et simulation() changent :
#   1. poids f**q (et leur cdf) précalculés une fois par génération
#   2. tirage par cdf.searchsorted(rng.random()) = ce que fait
#      Generator.choice(p=...) en interne, sans ses vérifications
#   3. comptes des classes tenus à jour (plus de reconcaténation de toute la
#      population à chaque génération) ; recalculés après une modération
# Ajout 2026-10-19 : fitness des classes (sigma, défaut 0). Avec sigma = 0, les
# résultats restent identiques à simulation.py ; avec sigma > 0, pas d'équivalent.
# =============================================================================
# =============================================================================
# TODO — Modèle d'usage des tags sur une plateforme de production en ligne
# =============================================================================
#
# Choix de modèle actés
# ---------------------
# - Chaque individu porte un ensemble de tags (classes), nb_tags ~ Poisson tronquée.
# - f = fréquence relative de chaque classe PARMI LES TAGS (bincount sur tous
#   les tags concaténés, minlength = C).
# - Mise à jour SYNCHRONE : f est calculé une fois par génération sur TOUS les
#   individus existants (cumulatif, pas de mort -> mémoire uniforme, tau = inf).
# - Q : un q par RANG de tag (1er tag -> Q[0], 2e -> Q[1], ...), tiré une fois
#   par simulation, commun à toute la population. Rang > len(Q) -> Q[-1].
# - Tirage rang par rang, sans remise :
#     1. avec proba mu : création d'une classe nouvelle (C += 1)
#     2. sinon copie conformiste : w_k = f_k ** q_r, classes déjà choisies -> 0,
#        renormalisation, tirage d'UNE classe
#   Si tous les poids restants sont nuls (épuisement) : on n'ajoute plus de tag.
#   (Exclure puis élever à la puissance == l'inverse : on part toujours du f d'origine.)
#
# Implémentation
# --------------
# [x] frequences() : f sur tous les tags, longueur C (C croît avec l'innovation)
# [x] tirer_tags() : tirage rang par rang décrit ci-dessus
# [x] generation() : f figé, puis tirage pour chacun des nouveaux individus
# [x] boucle principale : enchaîner les générations, stocker f à chaque génération
#     (compléter par des zéros les anciens vecteurs quand C augmente)
# [x] calculer f une seule fois par génération (hors de la boucle sur les individus)
# [x] mettre à jour `classes` / d avec les nouveaux individus à la fin de chaque génération
# [ ] calibrer mu (Cattuto et al. trouvent p ~ 0.03-0.06)
# [x] vérifier l'init : np.arange(1, N) donne N-1 individus
# [x] remplacer les +50 individus/génération par une vraie interpolation
# [x] rng : un seul générateur passé partout (reproductibilité via une graine)
#
# Mesures / validation
# --------------------
# [x] courbe fréquence-rang (log-log) : pente de la queue, aplatissement des premiers rangs
# [x] loi de Heaps D(n) : nb de classes distinctes vs nb total de tags
#     (mu constant -> croissance linéaire, attendu empiriquement : sublinéaire)
# [x] renouvellement (turnover) du top-k par génération : distingue conformité
#     (top stable) et mémoire courte (modes)
# [x] autocorrélation temporelle des tags (signature de la mémoire, cf. Cattuto)
# [x] comparer aux données empiriques de la plateforme
#
# À envisager
# -----------
# [ ] Ajouter un cycle de modération dans les simulations (E.g. : Toutes les n générations, on
#      agr)
# [ ] exploration nu : avec proba nu, tirage uniforme dans tout le vocabulaire
#     existant (classes éteintes incluses) -> modèle hybride copie/exploration/création
# [ ] mu décroissant (type Pitman-Yor / restaurant chinois) si D(n) trop linéaire
# [ ] mémoire tau : pondérer chaque individu par 1/(âge en générations + tau) dans f
#     (tau -> inf = modèle actuel). Attention : confusion possible q / tau sur la
#     courbe fréquence-rang -> les séparer via turnover et autocorrélation.
#     N'ajouter tau que si les données montrent des modes que q n'explique pas.
# [ ] f mesuré en part d'individus plutôt qu'en part de tags (comparer)
# [ ] q propre à chaque individu (hétérogénéité, éventuellement héritée)
# [ ] mettre à l'épreuve l'hypothèse de Poisson : version sans Poisson, voir si
#     elle émerge
#
# Référence : Cattuto, Loreto & Pietronero (2007, PNAS), Semiotic dynamics and
# collaborative tagging (Yule-Simon avec mémoire longue) ; Tria et al. (2014) ;
# Bentley et al. (turnover).
# =============================================================================
import argparse
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from tqdm import tqdm

# valeurs empiriques des métriques (scripts/empirical_metrics.py)
EMPIRIQUE = Path(__file__).resolve().parent / "data" / "empirical_metrics.json"

Tags = npt.NDArray[np.int_]      # classes d'une histoire, ou vecteur d'indices
Population = dict[int, Tags]     # d : {histoire: tags}


def poisson_tronquee(lam, n, xmin, rng):
    """Tire n valeurs d'une loi de Poisson(lam) tronquée à gauche (valeurs >= xmin).

    Méthode par rejet : on tire par paquets de 2n et on ne garde que les
    valeurs >= xmin jusqu'à en avoir n.
    Retourne un int si n == 1, sinon un array de taille n.
    Sert à tirer le nombre de tags de chaque individu (xmin=1 : au moins un tag).
    """
    out = np.empty(0, dtype=int)
    while out.size < n:
        t = rng.poisson(lam, size=2 * n)
        out = np.concatenate([out, t[t >= xmin]])
    if n == 1:
        return int(out[0])
    return out[:n]


def frequences(d, C):
    """Fréquence relative de chaque classe parmi tous les tags de la population.

    d : dict {individu: array des classes}, tous les individus existants
        (population cumulative, pas de mort)
    C : nombre de classes existantes (innovations comprises)
    Retourne un array de taille C qui somme à 1 ; une classe absente vaut 0.

    Attention : minlength est une longueur minimale. Si une classe >= C est
    présente dans d (C pas mis à jour après une innovation), le vecteur est
    plus long que C sans erreur. Donne des nan si d ne contient aucun tag
    (n'arrive pas : xmin=1 à l'initialisation).
    """
    cl = np.concatenate(list(d.values()))
    return np.bincount(cl, minlength=C) / len(cl)

    # raise NotImplementedError


def cdf_poids(w):
    """cdf normalisée des poids w, calculée comme Generator.choice(p=w/w.sum()) :
    p = w / s ; cdf = p.cumsum() ; cdf /= cdf[-1]. None si la somme est nulle."""
    s = w.sum()
    if s == 0:
        return None
    cdf = (w / s).cumsum()
    cdf /= cdf[-1]
    return cdf


def tirer_tags(W, CDF, nb_tags, mu, C, rng):
    """Tire les tags d'un individu, rang par rang, sans remise.

    W       : W[r] = poids f**Q[r] (0 pour les classes absentes) de la
              génération, figés (voir generation())
    CDF     : CDF[r] = cdf_poids(W[r]), utilisée quand rien n'est à exclure
    nb_tags : nombre de tags de l'individu (Poisson tronquée)
    mu      : probabilité d'innovation à chaque rang
    C       : nombre de classes existantes avant cet individu
    rng     : générateur unique de la simulation

    Même modèle et même consommation de rng que simulation.tirer_tags() :
      1. on tire u ~ U(0, 1)
      2. si u < mu (innovation) : le tag est la nouvelle classe d'indice C,
         puis C += 1.
      3. sinon (copie conformiste) : poids W[min(r, len(W)-1)], classes déjà
         portées par l'individu à 0 (sans remise). Somme nulle (épuisement) :
         on s'arrête. Sinon on tire une classe : cdf.searchsorted(rng.random(),
         side="right"), exactement ce que fait rng.choice(len(p), p=p).

    W a la longueur C du début de génération : une classe créée pendant la
    génération n'est pas copiable avant la génération suivante (synchrone).
    Retourne (array des classes de l'individu, C mis à jour).
    """
    tags = []
    for r in range(nb_tags):
        u = rng.random()
        if u < mu:
            tags.append(C)
            C += 1
            continue
        k = min(r, len(W) - 1)
        exclus = [tag for tag in tags if tag < len(W[k])]  # hors innovations
        if exclus:
            w = W[k].copy()
            w[exclus] = 0
            cdf = cdf_poids(w)
        else:
            cdf = CDF[k]
        if cdf is None:
            return np.array(tags, dtype=int), C
        tags.append(int(cdf.searchsorted(rng.random(), side="right")))

    return np.array(tags, dtype=int), C


def comptes(d: Population, C: int) -> Tags:
    """
    d -> Dictionnaire de suivi
    C -> Nombre de classes
    Renvoie un array de fréquence pour chaque classe, à partir du dictionaire de suivi global ( indiv : classes)
    Les fréquences sont les index de l'array. Facilite la manipulation directement.
    """
    #assert len(count) == C
    return np.bincount(np.concatenate(list(d.values())), minlength=C)

def compte_rare(count: Tags, j: int, C_derniere_passe: int) -> tuple[Tags, Tags]: 
    """ Renvoie les indices des classes rares (ie. dont la fréquences est sous le seuil) et les classes candidates à la fusion

    Args:
        count (Tags): Sorti de la fonction compte (ie. array de fréquence des classes indexées)
        j (int):  seuil 'rare' à partir duquel une classe est fusionnée
        C_derniere_passe (int):Nombre de classe à la dernière passe. Permet de filtrer les apparitions lors de la dernière classe

    Returns:
        tuple[Tags, Tags] :  (l'indices des classes rares, l'indices des classes candidates à la fusion)
    """    
  

    
    rare_indx = (count > 0) & (count < j) & (np.arange(len(count)) < C_derniere_passe)
    rare_indx = np.flatnonzero(rare_indx)

    target = np.flatnonzero(count>=j)
    return rare_indx, target

def table_fusion(indx_rare: Tags, indx_cible: Tags, C: int, rng: np.random.Generator) -> Tags:
    """Fonction qui établi la fusion des classes entre les rares ( à supprimer) et les cibles (avec qui la fusion va s'opérer).
    Args:
        indx_rare (Tags): Array d'indice des classes rares à fusionner
        indx_cible (Tags): Array d'indices des classes sur qui la fusion peut s'opérer
        C (int): Le nombre de classe
        rng (np.random.Generator): un generator initialisé en amont 

    Returns:
        Tags : table de remplacement, où les indices à fusionner sont remplacés par leur fusion cible.
    """    
    c = np.arange(C)
    if (len(indx_rare) == 0) | (len(indx_cible) ==0): 
        return c

    c[indx_rare]  = rng.choice(indx_cible, len(indx_rare))
    return c
    
def fusion(table: Tags, 
           d: Population) -> tuple[Population, int]:
    """Fusionne les classes rare / cible à partir de la table de fusion de la fonction "fusion_table()".

    Args:
        table (Tags): table de fusion
        d (Population): dictionnaire de suivi des individus / classes

    Returns:
        tuple[Population, int] : dictionnaire de suivi des individus mis à jour, nombre de doublons 
    """   
    e = {}
    n_doublons = 0
    touche = table != np.arange(len(table))   # True pour les classes redirigées
    for story, tags in d.items():
        if not touche[tags].any():            # aucune classe rare dans l'histoire
            e[story] = tags
            continue
        new_tags = table[tags]
        ind = np.sort(np.unique(new_tags, return_index=True)[1])
        new_tags = new_tags[ind]
        e[story] = new_tags
        n_doublons += len(tags) - len(new_tags)
    
    return e, n_doublons


def moderer(d: Population, C: int, j: int, C_derniere_passe: int, rng: np.random.Generator) -> tuple[Population, int, int]:
    """
    Fonction de modération : Fusionne les classes rares non récentes avec d'autres classes, de façon uniforme. 

    Args:
        d (Population): dictionnaire de population
        C (int): Nombre de classes
        j (int): Seuil de rareté
        C_derniere_passe (int): Nombre de classes à la dernière passe
        rng (np.random.Generator): rng

    Returns:
        tuple[Population, int, int]: Dictionnaire de population mis à jour, nombre de doublons, nombre de fusions
    """    

    count = comptes(d=d, C= C)
    rare_idx, target_idx = compte_rare(count= count, j =j ,C_derniere_passe=C_derniere_passe)
    #Vérifier si besoin de modération (rien de rare ou aucune cible) :

    if (len(rare_idx) == 0) or (len(target_idx) == 0) : 
        return d , 0 , 0
    else:
        table = table_fusion(indx_rare=rare_idx, indx_cible=target_idx,C=C, rng=rng )
        d_fusion, n_doublons = fusion(table=table, d = d )
        # Combien de classes fusionnés : 
        n_fusionnees= int((table != np.arange(C)).sum())

        return  d_fusion, n_doublons, n_fusionnees
   


def generation(d, nouveaux, Q, mu, C, lam, rng, count, eta=None):
    """Ajoute une génération d'individus à d (mise à jour synchrone).

    d        : dict {individu: array des classes}, population existante
    nouveaux : identifiants des nouveaux individus
    Q, mu    : voir simulation()
    C        : nombre de classes existantes
    lam      : paramètre de la Poisson tronquée (xmin=1)
    rng      : générateur unique de la simulation
    count    : comptes des classes dans d (longueur C), tenus à jour par
               simulation()
    eta      : fitness des classes (longueur >= C), ou None (pas de fitness)

    1. f = count / count.sum() (= frequences(d, C)), puis poids eta * f**q et
       leur cdf pour chaque valeur de Q, calculés UNE seule fois.
    2. Pour chaque nouvel individu : nb_tags ~ Poisson tronquée, puis
       tirer_tags() avec ces poids figés (C est mis à jour au fil des
       innovations).
    3. Les nouveaux individus sont ajoutés à d à la fin de la génération :
       ils ne servent de modèle qu'à la génération suivante.
    Retourne (d, C, f utilisé pour cette génération, count mis à jour).
    """
    f = count / count.sum()
    W = [np.where(f > 0, f**q, 0) for q in Q]  # permet de ne pas faire revivre des classes
    if eta is not None:
        W = [w * eta[: len(f)] for w in W]
    CDF = [cdf_poids(w) for w in W]

    nb_tag_nouveaux = np.atleast_1d(poisson_tronquee(lam, len(nouveaux), 1, rng))
    for ind, nv in enumerate(nouveaux):
        d[nv], C = tirer_tags(W, CDF, nb_tag_nouveaux[ind], mu, C, rng)

    # comptes : on ajoute les tags des nouveaux individus (classes créées comprises)
    count = np.concatenate([count, np.zeros(C - len(count), dtype=count.dtype)])
    if len(nouveaux):
        count += np.bincount(np.concatenate([d[nv] for nv in nouveaux]), minlength=C)

    return d, C, f, count


def simulation(Ni, Nf, C, lam, mu, T,verbose=False, seed=42, q_max=2.0, K=0, j=2, sigma=0.0):
    """
    Fonction de simulation (version optimisée de simulation.simulation(),
    résultats identiques à graine égale). Description du modèle:
    - Ni individus initiaux (identifiants 0 à Ni-1), chacun avec
      nb_tags ~ Poisson tronquée (xmin=1) classes tirées uniformément SANS
      remise parmi C classes.
    - Q : un coefficient de conformité par rang de tag, tiré une fois pour
      la simulation (6 valeurs ~ U(0, q_max), la dernière répétée ensuite).
      q_max vaut 2 par défaut (grilles jusqu'au 2026-10-14) ; changer q_max
      ne décale pas la suite aléatoire (toujours 6 tirages), seul Q change.
    - Modération : K passes, régulièrement réparties en nombre d'individus
      entre Ni et Nf, la dernière à Nf. Une passe a lieu à la fin de la
      première génération qui franchit un seuil, au plus une par génération
      (n_passes <= K si T < K). Les classes de compte entre 1 et j-1 sont
      fusionnées vers une classe tirée uniformément parmi celles de compte
      >= j ; les classes créées depuis la passe précédente sont exemptées.
      K=0 : aucune modération, aucun tirage aléatoire (simulation inchangée).
    - Fitness (biais de contenu) : chaque classe k reçoit à sa création une
      fitness eta_k = exp(sigma * Z_k), Z_k ~ N(0, 1), fixe ensuite. Le poids de
      copie devient eta_k * f_k ** q (Bianconi & Barabási 2001). Les Z_k
      viennent d'un générateur séparé, default_rng([seed, 1]), dans l'ordre des
      classes : le générateur principal n'est pas touché, et Z_k ne dépend que
      de (seed, k), quels que soient sigma, K, j. sigma=0 : pas de fitness
      (simulation inchangée).
    - La population croît linéairement de Ni à Nf en T générations
      (tailles cumulées arrondies, identifiants Ni à Nf-1). Les nouveaux
      individus copient la population existante (cumulative, pas de mort) :
      voir generation() et tirer_tags().
    - Innovation : à chaque rang, avec proba mu, création d'une classe
      nouvelle à la place d'une copie.
    - Un seul générateur aléatoire pour toute la simulation (reproductibilité).

    C : nombre initial de classes (devient le nombre courant au fil des
        innovations)
    verbose : affiche une barre de progression
    K : nombre de passes de modération (passe finale à Nf comprise)
    j : seuil de rareté (compte de tags) de la modération
    sigma : dispersion (log) des fitness ; 0 = pas de fitness

    Retourne un dict :
      d          : population finale {individu: array des classes}
      Q          : coefficients de conformité utilisés
      C          : nombre final de classes
      historique : array (T, C final), ligne t = f utilisé à la génération t
                   (complété par des zéros pour les classes créées après)
      C_par_gen  : nombre de classes à la fin de chaque génération
      f_final    : fréquences de la population finale (après la passe finale)
      n_passes   : nombre de passes de modération effectuées
      n_fusionnees, n_doublons : totaux sur toutes les passes (voir moderer())
      eta        : fitness des C classes finales (des 1 si sigma = 0)
    """

    ## ------------INITIALISATION------------
    rng = np.random.default_rng(seed=seed)

    # initialisation des coef de conformité
    Q = rng.uniform(0, q_max, size=6)
    Q = np.append(Q, [Q[-1]] * 10)

    d = {}
    for p in range(Ni):
        nb_tags = poisson_tronquee(
            lam, 1, 1, rng
        )  # Penser à intégrer une version sans poisson, jsute full random et voir si la poisson émerge sur les meilleures cas
        # sans remise : au plus C tags distincts
        d[p] = rng.choice(C, min(nb_tags, C), replace=False)

    # tailles cumulées de la population : Ni, ..., Nf (T+1 valeurs, T générations)
    tailles = np.round(np.linspace(Ni, Nf, T + 1)).astype(int)
    nouveaux_par_gen = np.split(np.arange(Ni, Nf), tailles[1:-1] - Ni)

    # passes de modération : pour chaque seuil, première génération dont la
    # taille finale l'atteint ; le set garde au plus une passe par génération
    seuils_moderations = np.linspace(Ni, Nf, K + 1)[1:]
    gen_passes = set(np.searchsorted(tailles[1:], seuils_moderations).tolist())

    C_derniere_passe = C
    n_fusionnees = n_doublons = 0
    count = comptes(d, C)

    # fitness : Z_k tirés à la demande dans un générateur séparé (ordre des classes)
    rng_fit = np.random.default_rng([seed, 1])
    Z = rng_fit.standard_normal(C) if sigma > 0 else None

    ###------------CORE------------
    historique = []
    C_par_gen = []
    for t, nouveaux in enumerate(tqdm(nouveaux_par_gen, disable=not verbose)):
        d, C, f, count = generation(
            d=d,
            nouveaux=nouveaux,
            Q=Q,
            mu=mu,
            C=C,
            lam=lam,
            rng=rng,
            count=count,
            eta=None if Z is None else np.exp(sigma * Z),
        )
        if Z is not None and C > len(Z):  # classes créées pendant la génération
            Z = np.concatenate([Z, rng_fit.standard_normal(C - len(Z))])
        historique.append(f)
        C_par_gen.append(C)
        ## Passes de modération (après historique : f reste celui d'avant la passe)
        if t in gen_passes:
            d, n_doublon, n_fusionnee = moderer(
                d=d,
                C=C,
                j=j,
                C_derniere_passe=C_derniere_passe,
                rng=rng,
            )
            C_derniere_passe = C
            count = comptes(d, C)
            n_fusionnees += n_fusionnee
            n_doublons += n_doublon

    ###------------CONCLUSION------------
    # les f n'ont pas tous la même longueur (C croît) : on complète par des zéros
    H = np.zeros((T, C))
    for t, f in enumerate(historique):
        H[t, : len(f)] = f

    return {
        "d": d,
        "Q": Q,
        "C": C,
        "historique": H,
        "C_par_gen": np.array(C_par_gen),
        "f_final": frequences(d, C),
        "n_passes": len(gen_passes),
        "n_fusionnees": n_fusionnees,
        "n_doublons": n_doublons,
        "eta": np.ones(C) if Z is None else np.exp(sigma * Z),
    }


###------------MÉTRIQUES------------


def flux_tags(d):
    """Flux chronologique de tous les tags : tags des individus concaténés
    dans l'ordre des identifiants (= ordre d'arrivée dans la population).
    Équivalent du « texte » de Cattuto et al.
    """
    return np.concatenate([d[i] for i in sorted(d)])


def rang_frequence(f):
    """Courbe fréquence-rang : fréquences non nulles triées par ordre
    décroissant. Retourne (rangs à partir de 1, fréquences).
    À tracer en log-log.
    """
    fr = np.sort(f[f > 0])[::-1]
    return np.arange(1, len(fr) + 1), fr


def pente_queue(f, rang_min=10):
    """Exposant alpha de la queue de la courbe fréquence-rang
    (P(R) ~ R^-alpha), par régression linéaire en log-log sur les rangs
    >= rang_min. Retourne nan s'il y a moins de 3 points.
    Estimation grossière : à compléter par une inspection visuelle.
    """
    rangs, fr = rang_frequence(f)
    m = rangs >= rang_min
    if m.sum() < 3:
        return np.nan
    pente, _ = np.polyfit(np.log(rangs[m]), np.log(fr[m]), 1)
    return -pente


def heaps(flux):
    """Loi de Heaps : D(n) = nombre de classes distinctes parmi les n premiers
    tags du flux, et exposant beta de D(n) ~ n^beta (régression log-log).
    beta ~ 1 : croissance linéaire (attendu avec mu constant) ;
    beta < 1 : croissance sublinéaire (attendu empiriquement).
    Retourne (D, beta).
    """
    _, premieres = np.unique(flux, return_index=True)
    nouveau = np.zeros(len(flux), dtype=int)
    nouveau[premieres] = 1
    D = np.cumsum(nouveau)
    n = np.arange(1, len(flux) + 1)
    beta, _ = np.polyfit(np.log(n), np.log(D), 1)
    return D, beta


def turnover(H, k=10):
    """Renouvellement du top-k (Bentley et al.) : pour chaque génération t >= 1,
    nombre de classes du top-k de f_t absentes du top-k de f_{t-1}.
    H : historique (T, C) renvoyé par simulation().
    Seules les classes de fréquence non nulle comptent dans le top-k.
    Top stable (turnover faible) : conformité ; modes : mémoire courte.
    Retourne un array de taille T-1.
    """

    def top(f):
        ordre = np.argsort(f)[::-1][:k]
        return set(ordre[f[ordre] > 0].tolist())

    tops = [top(f) for f in H]
    return np.array([len(tops[t] - tops[t - 1]) for t in range(1, len(tops))])


def autocorrelation(flux, max_lag=100):
    """Autocorrélation temporelle du flux de tags (Cattuto et al.) :
    C(lag) = proportion des positions t telles que tag(t + lag) == tag(t),
    pour lag = 1..max_lag, et niveau de référence c = somme des p_k^2
    (corrélation attendue dans un flux mélangé au hasard).
    Une décroissance lente vers c signale un effet mémoire.
    Attention : aux petits lags, deux positions appartiennent souvent au même
    individu, qui n'a pas de doublon (sans remise) -> C(lag) tiré vers le bas.
    Retourne (lags, C, c).
    """
    lags = np.arange(1, max_lag + 1)
    Cl = np.array([np.mean(flux[l:] == flux[:-l]) for l in lags])
    p = np.bincount(flux) / len(flux)
    return lags, Cl, np.sum(p**2)


def turnover_stable(H, k=10):
    """turnover(), mais avec un tri stable : à fréquence égale, la classe
    d'indice le plus petit passe devant. Avec le tri par défaut (non stable),
    l'ordre des ex aequo dépend de la machine et des ex aequo à la frontière
    du top-k comptent comme du renouvellement.
    """
    tops = []
    for f in H:
        ordre = np.argsort(-f, kind="stable")[:k]
        tops.append(set(ordre[f[ordre] > 0].tolist()))
    return np.array([len(tops[t] - tops[t - 1]) for t in range(1, len(tops))])


def turnover_blocs(ntags, flux, T=50, k=10):
    """Turnover moyen du top-k sur T blocs d'individus de même taille (ordre
    d'arrivée) : f cumulé à la fin de chaque bloc, comme metrics() dans
    scripts/empirical_metrics.py, avec un tri stable (turnover_stable).
    Comparable à l'empirique, contrairement à turnover() sur les T générations.
    ntags : nombre de tags de chaque individu, dans l'ordre d'arrivée.
    """
    C = int(flux.max()) + 1
    bornes = np.round(np.linspace(0, len(ntags), T + 1)).astype(int)[1:]
    fin = np.cumsum(ntags)[bornes - 1]                        # nb de tags à la fin de chaque bloc
    bloc = np.repeat(np.arange(T), np.diff(np.r_[0, fin]))     # bloc de chaque tag du flux
    H = np.bincount(bloc * C + flux, minlength=T * C).reshape(T, C).cumsum(0).astype(float)
    H /= H.sum(1, keepdims=True)
    return float(turnover_stable(H, k=k).mean())


def autocorr_exces(flux, lags=np.arange(10, 101)):
    """Autocorrélation du flux aux décalages moyens, rapportée au plateau
    c = somme des p² (signature de la mémoire, Cattuto et al. 2007).
    Au décalage 1, C est tiré sous c (pas de doublon dans un individu) :
    on regarde donc dt de 10 à 100.
    Retourne autocorr_100 = C(100), exces_100 = C(100)/c - 1 et
    exces_10_100 = moyenne de C(dt)/c - 1 sur les décalages (moins bruité).
    """
    C = np.array([np.mean(flux[l:] == flux[:-l]) for l in lags])
    p = np.bincount(flux) / len(flux)
    c = np.sum(p ** 2)
    return {"autocorr_100": float(C[-1]), "exces_100": float(C[-1] / c - 1), "exces_10_100": float(C.mean() / c - 1)}


def age_effectif(flux):
    """Corrélation de Spearman entre l'ordre de première apparition d'une classe dans le flux et son
    effectif final (classes présentes). Proche de -1 : l'effectif est dicté par l'âge (premier arrivé,
    premier servi) ; plus proche de 0 : hétérogénéité que l'âge n'explique pas (biais de contenu,
    fitness). Rangs moyens pour les ex aequo d'effectif. NaN si moins de 3 classes."""
    _, premier, effectif = np.unique(flux, return_index=True, return_counts=True)
    if len(premier) < 3:
        return np.nan

    def rangs(x):
        ordre = np.argsort(x, kind="stable")
        xs = x[ordre]
        debut = np.r_[True, xs[1:] != xs[:-1]]
        groupe = np.cumsum(debut) - 1
        bornes = np.r_[np.flatnonzero(debut), len(x)]
        moyen = (bornes[:-1] + bornes[1:] - 1) / 2
        r = np.empty(len(x))
        r[ordre] = moyen[groupe]
        return r

    a, b = rangs(premier), rangs(effectif)
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulation de l'usage des tags avec biais de conformité"
    )
    parser.add_argument(
        "--Ni", type=int, default=100, help="taille initiale de la population"
    )
    parser.add_argument(
        "--Nf", type=int, default=1000, help="taille finale de la population"
    )
    parser.add_argument("--C", type=int, default=10, help="nombre initial de classes")
    parser.add_argument(
        "--lam",
        type=float,
        default=1.98,
        help="paramètre de la Poisson tronquée (nb de tags)",
    )
    parser.add_argument(
        "--mu", type=float, default=0.05, help="probabilité d'innovation par rang"
    )
    parser.add_argument("--T", type=int, default=50, help="nombre de générations")
    parser.add_argument(
        "--seed", type=int, default=42, help="graine du générateur aléatoire"
    )
    parser.add_argument(
        "--k", type=int, default=10, help="taille du top-k pour le turnover"
    )
    parser.add_argument(
        "--max_lag", type=int, default=100, help="lag maximal de l'autocorrélation"
    )
    parser.add_argument(
        "--sortie",
        type=str,
        default=None,
        help="fichier .npz où enregistrer les résultats",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="affiche la barre de progression"
    )
    parser.add_argument(
        "--plateforme",
        type=str,
        default="fandom",
        help="plateforme empirique de comparaison (fandom, scp_en, scp_fr, ..., backrooms ; aucune = pas de comparaison)",
    )
    args = parser.parse_args()

    res = simulation(
        Ni=args.Ni,
        Nf=args.Nf,
        C=args.C,
        lam=args.lam,
        mu=args.mu,
        T=args.T,
        verbose=args.verbose,
        seed=args.seed,
    )

    flux = flux_tags(res["d"])
    D, beta = heaps(flux)
    to = turnover(res["historique"], k=args.k)
    lags, Cl, c = autocorrelation(flux, max_lag=args.max_lag)

    n_par_indiv = np.array([len(v) for v in res["d"].values()])
    simu = {
        "n_pages": len(res["d"]),
        "n_tags": len(flux),
        "n_classes": int((np.bincount(flux) > 0).sum()),
        "tags_par_page": n_par_indiv.mean(),
        "part_1_tag": (n_par_indiv == 1).mean(),
        "alpha": pente_queue(res["f_final"]),
        "beta": beta,
        "turnover_moy": to.mean(),
        "autocorr_1": Cl[0],
        f"autocorr_{args.max_lag}": Cl[-1],
        "autocorr_ref": c,
    }
    emp = {}
    if args.plateforme != "aucune":
        if EMPIRIQUE.exists():
            toutes = json.loads(EMPIRIQUE.read_text())
            if args.plateforme in toutes:
                emp = toutes[args.plateforme]
            else:
                print(
                    f"plateforme inconnue '{args.plateforme}' (disponibles : {', '.join(toutes)})"
                )
        else:
            print(f"{EMPIRIQUE} absent : lancer scripts/empirical_metrics.py")

    lignes = [
        ("individus (pages)", "n_pages", "d"),
        ("tags", "n_tags", "d"),
        ("classes observées", "n_classes", "d"),
        ("tags / individu", "tags_par_page", ".2f"),
        ("part à 1 tag", "part_1_tag", ".2f"),
        ("alpha (queue)", "alpha", ".3f"),
        ("beta (Heaps)", "beta", ".3f"),
        (f"turnover top-{args.k}", "turnover_moy", ".2f"),
        ("autocorr. lag 1", "autocorr_1", ".4f"),
        (f"autocorr. lag {args.max_lag}", f"autocorr_{args.max_lag}", ".4f"),
        ("autocorr. référence", "autocorr_ref", ".4f"),
    ]

    def fmt(v, f):
        if v is None:
            return "-"
        return format(int(v), f) if f == "d" else format(float(v), f)

    print(f"Q : {np.round(res['Q'][:6], 3)}   classes créées : {res['C']}")
    entete = f"{'métrique':22s} {'simulation':>12s}"
    if emp:
        entete += f" {args.plateforme:>12s}"
    print(entete)
    print("-" * len(entete))
    for nom, cle, f in lignes:
        ligne = f"{nom:22s} {fmt(simu.get(cle), f):>12s}"
        if emp:
            ligne += f" {fmt(emp.get(cle), f):>12s}"
        print(ligne)
    if emp:
        print(
            f"{args.plateforme} : lam estimé {emp['lam_estime']:.2f} (Poisson tronquée à 1), "
            f"turnover sur T = {emp['T']} blocs de pages, k = {emp['k']}"
        )
        if args.T != emp["T"] or args.k != emp["k"]:
            print(
                f"ATTENTION : T ou k différents de la simulation (T = {args.T}, k = {args.k}) : turnover non comparable"
            )

    if args.sortie:
        rangs, fr = rang_frequence(res["f_final"])
        np.savez(
            args.sortie,
            Q=res["Q"],
            historique=res["historique"],
            C_par_gen=res["C_par_gen"],
            f_final=res["f_final"],
            flux=flux,
            heaps_D=D,
            turnover=to,
            lags=lags,
            autocorr=Cl,
            autocorr_ref=c,
            rangs=rangs,
            freq_rang=fr,
        )
        print(f"résultats enregistrés dans {args.sortie}")
