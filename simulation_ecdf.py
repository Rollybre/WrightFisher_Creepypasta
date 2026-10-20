# =============================================================================
# simulation_ecdf.py — variante de simulation_opti.py (2026-10-19), sans fitness
#
# Différence avec le modèle actuel : l'innovation n'est plus aléatoire (pas de
# mu) et le stock initial n'est plus un paramètre (pas de C). Les classes
# apparaissent en suivant la fonction de répartition empirique (ecdf) des
# premières apparitions de tags sur une plateforme (data/ecdf_innovation.json,
# écrit par scripts/ecdf_innovation.py) :
#   D_cible(n) = nombre de classes empiriques apparues avant la fraction n / Nf
#                des pages,
# ramené à Nf pages simulées. Avec Nf = nombre de pages empiriques (13 108 pour
# fandom), D_cible(n) est exactement le nombre de classes apparues dans les n
# premières pages.
#
# - Initialisation : les Ni premiers individus tirent leurs tags uniformément
#   SANS remise parmi D_cible(Ni) classes (la salve fondatrice). Les classes
#   qu'aucun individu initial n'a tirées sont retirées (renumérotées) : elles
#   seront créées plus tard par le rattrapage ci-dessous.
# - Génération t : le nombre d'innovations est D_cible(taille à la fin de t)
#   moins le nombre de classes déjà créées (rattrapage, jamais négatif).
#   Elles occupent des emplacements de tags tirés uniformément sans remise
#   parmi tous les emplacements de la génération (une innovation remplace une
#   copie, comme dans le modèle actuel). Les autres emplacements copient
#   (poids f**q, sans remise, f figé pendant la génération). S'il y a plus
#   d'innovations que d'emplacements, le reste passe à la génération suivante.
# - Épuisement (plus aucune classe copiable pour un individu) : l'emplacement
#   de copie est sauté, les innovations prévues plus loin ont quand même lieu.
# - Modération (K, j) : inchangée (voir simulation_opti.moderer). Elle ne
#   change pas le nombre de classes CRÉÉES, donc pas le calendrier.
#
# Paramètres restants : Ni, Nf, lam, T, Q (tiré, q_max), K, j, plateforme.
# Les métriques sont celles de simulation_opti (réexportées ici).
# =============================================================================
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from simulation_opti import (EMPIRIQUE, Population, Tags, age_effectif, autocorr_exces,  # noqa: F401
                             autocorrelation, cdf_poids, comptes, flux_tags, frequences, heaps, moderer, pente_queue,
                             poisson_tronquee, rang_frequence, turnover, turnover_blocs, turnover_stable)

ECDF = Path(__file__).resolve().parent / "data" / "ecdf_innovation.json"


def charger_ecdf(plateforme="fandom"):
    """Positions relatives (dans [0, 1[) des premières apparitions des classes de la plateforme, triées."""
    if not ECDF.exists():
        raise FileNotFoundError(f"{ECDF} absent : lancer scripts/ecdf_innovation.py")
    toutes = json.loads(ECDF.read_text())
    if plateforme not in toutes:
        raise ValueError(f"plateforme '{plateforme}' absente de {ECDF.name} (disponibles : {', '.join(toutes)})")
    e = toutes[plateforme]
    return np.sort(np.asarray(e["premiere_page"], dtype=float)) / e["n_pages"]


def classes_cibles(n, Nf, positions):
    """D_cible(n) : nombre de classes empiriques apparues avant la fraction n / Nf des pages (n scalaire ou array)."""
    return np.searchsorted(positions, np.asarray(n) / Nf, side="left")


def tirer_tags(W, CDF, innov, C, rng):
    """Tire les tags d'un individu, rang par rang, sans remise.

    W, CDF : poids f**q par rang et leur cdf (voir simulation_opti.generation)
    innov  : booléens, un par emplacement : True = innovation imposée par le
             calendrier (nouvelle classe d'indice C, puis C += 1)
    C      : nombre de classes créées avant cet individu
    Copie (innov False) : comme simulation_opti.tirer_tags, mais si plus
    aucune classe n'est copiable, l'emplacement est sauté (les innovations
    prévues aux rangs suivants ont lieu).
    Retourne (array des classes de l'individu, C mis à jour).
    """
    tags = []
    for r, nouveau in enumerate(innov):
        if nouveau:
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
            continue
        tags.append(int(cdf.searchsorted(rng.random(), side="right")))
    return np.array(tags, dtype=int), C


def generation(d, nouveaux, Q, C, lam, rng, count, n_innov):
    """Ajoute une génération d'individus à d (mise à jour synchrone), avec n_innov innovations imposées.

    1. f = count / count.sum(), poids f**q et leur cdf pour chaque valeur de Q, calculés une fois.
    2. nb_tags ~ Poisson tronquée pour chaque nouvel individu ; min(n_innov, nombre d'emplacements)
       emplacements tirés uniformément sans remise deviennent des innovations.
    3. tirer_tags() pour chaque individu ; les nouveaux individus ne servent de modèle qu'à la
       génération suivante.
    Retourne (d, C, f utilisé, count mis à jour, nombre d'innovations réalisées).
    """
    f = count / count.sum()
    W = [np.where(f > 0, f**q, 0) for q in Q]
    CDF = [cdf_poids(w) for w in W]

    nb = np.atleast_1d(poisson_tronquee(lam, len(nouveaux), 1, rng))
    S = int(nb.sum()) if len(nouveaux) else 0
    k = min(n_innov, S)
    masque = np.zeros(S, dtype=bool)
    if k > 0:
        masque[rng.choice(S, k, replace=False)] = True
    debuts = np.r_[0, np.cumsum(nb)]
    for ind, nv in enumerate(nouveaux):
        d[nv], C = tirer_tags(W, CDF, masque[debuts[ind]:debuts[ind + 1]], C, rng)

    count = np.concatenate([count, np.zeros(C - len(count), dtype=count.dtype)])
    if len(nouveaux):
        count += np.bincount(np.concatenate([d[nv] for nv in nouveaux]), minlength=C)
    return d, C, f, count, k


def simulation(Ni, Nf, lam, T, verbose=False, seed=42, q_max=2.0, K=0, j=2, plateforme="fandom"):
    """
    Simulation avec innovations calées sur l'ecdf empirique de la plateforme (voir l'en-tête du fichier).
    Même modèle que simulation_opti.simulation pour la copie (Q par rang, f figé par génération, sans
    remise), la croissance (linéaire de Ni à Nf en T générations) et la modération (K, j).

    Retourne le même dict que simulation_opti.simulation (d, Q, C, historique, C_par_gen, f_final,
    n_passes, n_fusionnees, n_doublons), plus :
      D_cible  : nombre de classes visé à la fin de chaque génération (ecdf)
      C_init   : nombre de classes après l'initialisation (classes tirées par les Ni premiers individus)
      retard   : D_cible final - classes créées (> 0 seulement si les emplacements ont manqué)
    """
    rng = np.random.default_rng(seed=seed)
    Q = rng.uniform(0, q_max, size=6)
    Q = np.append(Q, [Q[-1]] * 10)

    positions = charger_ecdf(plateforme)
    tailles = np.round(np.linspace(Ni, Nf, T + 1)).astype(int)
    D_cible = classes_cibles(tailles, Nf, positions)

    # salve fondatrice : D_cible(Ni) classes, tirées uniformément par les Ni premiers individus
    C0 = max(int(D_cible[0]), 1)
    d = {}
    for p in range(Ni):
        nb_tags = poisson_tronquee(lam, 1, 1, rng)
        d[p] = rng.choice(C0, min(nb_tags, C0), replace=False)
    # classes non tirées : retirées (renumérotation), elles seront créées par rattrapage
    presentes = np.unique(np.concatenate(list(d.values())))
    if len(presentes) < C0:
        nouvel_indice = np.full(C0, -1)
        nouvel_indice[presentes] = np.arange(len(presentes))
        d = {p: nouvel_indice[v] for p, v in d.items()}
    C = len(presentes)
    C_init = C

    nouveaux_par_gen = np.split(np.arange(Ni, Nf), tailles[1:-1] - Ni)
    seuils_moderations = np.linspace(Ni, Nf, K + 1)[1:]
    gen_passes = set(np.searchsorted(tailles[1:], seuils_moderations).tolist())
    C_derniere_passe = C
    n_fusionnees = n_doublons = 0
    count = comptes(d, C)

    historique, C_par_gen = [], []
    for t, nouveaux in enumerate(tqdm(nouveaux_par_gen, disable=not verbose)):
        n_innov = max(0, int(D_cible[t + 1]) - C)
        d, C, f, count, _ = generation(d, nouveaux, Q, C, lam, rng, count, n_innov)
        historique.append(f)
        C_par_gen.append(C)
        if t in gen_passes:
            d, n_doublon, n_fusionnee = moderer(d=d, C=C, j=j, C_derniere_passe=C_derniere_passe, rng=rng)
            C_derniere_passe = C
            count = comptes(d, C)
            n_fusionnees += n_fusionnee
            n_doublons += n_doublon

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
        "D_cible": D_cible[1:],
        "C_init": C_init,
        "retard": int(D_cible[-1]) - C,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation avec innovations calées sur l'ecdf empirique")
    parser.add_argument("--Ni", type=int, default=220)
    parser.add_argument("--Nf", type=int, default=13108)
    parser.add_argument("--lam", type=float, default=2.0)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q_max", type=float, default=4.0)
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--j", type=int, default=2)
    parser.add_argument("--plateforme", default="fandom")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    res = simulation(Ni=args.Ni, Nf=args.Nf, lam=args.lam, T=args.T, verbose=args.verbose, seed=args.seed,
                     q_max=args.q_max, K=args.K, j=args.j, plateforme=args.plateforme)
    flux = flux_tags(res["d"])
    _, beta = heaps(flux)
    ntags = np.array([len(res["d"][i]) for i in sorted(res["d"])])
    simu = {"n_classes": int((np.bincount(flux) > 0).sum()), "alpha": pente_queue(res["f_final"]), "beta": beta,
            "turnover_50": turnover_blocs(ntags, flux), **autocorr_exces(flux), "age_effectif": age_effectif(flux)}
    emp = json.loads(EMPIRIQUE.read_text()).get(args.plateforme, {}) if EMPIRIQUE.exists() else {}
    print(f"Q : {np.round(res['Q'][:6], 3)} ; classes initiales {res['C_init']}, créées au total {res['C']} "
          f"(cible {res['D_cible'][-1]}, retard {res['retard']})")
    for cle, v in simu.items():
        e = emp.get(cle)
        print(f"{cle:14s} {v:10.3f}" + (f" {e:10.3f}" if e is not None else ""))
