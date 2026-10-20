"""
Métriques de la distribution finale pour une grille v3 déjà calculée, sans relancer les simulations : KL à
l'empirique, nombres de Hill, Gini, Chao1, singletons. Ajoute aussi un turnover comparable à l'empirique.

Les fonctions sont celles de simulations/v2/simulation.py (kl_divergence, hill_number, gini, chao1), pour que les
valeurs se comparent directement à celles des grilles V2 :
- kl_emp : KL(empirique || simulation) par rang, en nats, plancher 0,5 occurrence pour les rangs non atteints,
  masse simulée au-delà du rang 176 gardée dans la normalisation (pénalise les simulations à trop de classes) ;
- hill_0, hill_1, hill_2, hill_inf : richesse, exp(Shannon), 1/Simpson, 1/Berger-Parker ;
- gini_obs : Gini sur les classes observées (compte > 0), comme l'empirique ;
- chao1, n_singletons, n_doubletons.

Référence empirique : corpus fandom lu comme dans scripts/empirical_metrics.py (une page = un individu, tags
dédoublonnés par page ; 13 108 pages, 30 892 tags, 176 classes). Ce n'est pas exactement EMPIRICAL_COUNTS de V2
(30 901 occurrences, sans dédoublonnage), d'où des kl_emp légèrement différents pour une même distribution.

turnover_50 : turnover du top-10 sur 50 blocs d'individus de même taille, dans l'ordre d'arrivée, comme l'empirique
(scripts/empirical_metrics.py), avec un tri stable pour les ex aequo (cf. turnover_stable). La colonne turnover_moy
de la grille est calculée sur les T générations du run (T variable) et ne se compare pas à l'empirique.

Autocorrélation aux décalages moyens (signature de la mémoire, Cattuto, Loreto & Pietronero 2007, PNAS) : C(dt) =
part des positions t du flux où le tag en t + dt est le même qu'en t, rapportée au plateau d'une séquence aléatoire
de mêmes fréquences, c = somme des p². Au décalage 1, C est tiré sous c parce qu'une page n'a pas de doublon : on
regarde donc dt de 10 à 100 (plusieurs pages d'écart).
- autocorr_100 : C(100) ;
- exces_100 = C(100) / c - 1 ;
- exces_10_100 = moyenne de C(dt) / c - 1 pour dt = 10 ... 100 (moins bruité que exces_100).

Grilles légères (run_grid.py --sans_brut) : pas de flux stocké, donc turnover_50 et l'autocorrélation sont repris
des colonnes calculées pendant le run (mêmes fonctions, dans simulation.py) ; seules les métriques de distribution
sont calculées ici.

Usage (depuis simulations/v3/, pandas requis) :
    python scripts/metriques_distribution.py --grid results/grid_fandom/chunks --n_jobs 8
Sorties, à côté de --grid : metriques.csv (une ligne par run : paramètres, métriques de la grille, nouvelles
métriques) et metriques_empirique.csv (une ligne : les mêmes métriques sur le corpus fandom).
"""
import argparse
import importlib.util
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UCL = ROOT.parents[1]  # racine ucl_stay
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import empirical_metrics as em  # noqa: E402
from run_grid import METRIC_KEYS, PARAM_COLUMNS, _metric, _param, chunk_path, list_chunks  # noqa: E402
from simulation import age_effectif, autocorr_exces, turnover_blocs  # noqa: E402,F401  (réexportées pour les notebooks)

T_BLOCS, K = 50, 10
DIST_KEYS = ["kl_emp", "hill_0", "hill_1", "hill_2", "hill_inf", "gini_obs", "chao1", "n_singletons", "n_doubletons"]
FLUX_KEYS = ["turnover_50", "autocorr_100", "exces_100", "exces_10_100"]   # déjà dans METRIC_KEYS (grilles récentes)
NEW_KEYS = DIST_KEYS + FLUX_KEYS


def _load_v2():
    """simulations/v2/simulation.py sous un autre nom (le module v3 s'appelle aussi simulation)."""
    spec = importlib.util.spec_from_file_location("simulation_v2", UCL / "simulations" / "v2" / "simulation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V2 = _load_v2()


def pages_reference():
    """Pages du corpus fandom (listes de tags, ordre chronologique), sans les pages sans tag."""
    return [p for p in em.pages_fandom() if p]


def counts_reference(pages):
    """Comptes par classe du corpus fandom, triés décroissant."""
    return np.array(sorted(Counter(t for p in pages for t in p).values(), reverse=True))


def distribution_metrics(counts, reference):
    """Métriques de la distribution finale (comptes par classe, zéros admis)."""
    c = np.asarray(counts)
    obs = c[c > 0]
    return {
        "kl_emp": V2.kl_divergence(c, reference),
        "hill_0": V2.hill_number(obs, 0),
        "hill_1": V2.hill_number(obs, 1),
        "hill_2": V2.hill_number(obs, 2),
        "hill_inf": V2.hill_number(obs, np.inf),
        "gini_obs": V2.gini(obs),
        "chao1": V2.chao1(obs),
        "n_singletons": int((obs == 1).sum()),
        "n_doubletons": int((obs == 2).sum()),
    }


def process_chunk(args):
    """Nouvelles métriques de tous les runs d'un bloc. Retour : DataFrame (paramètres, métriques, nouvelles)."""
    out_dir, cid, reference = args
    with np.load(chunk_path(out_dir, cid)) as z:
        df = pd.DataFrame({key: _param(z, key) for key in PARAM_COLUMNS})
        for key in METRIC_KEYS:
            if key not in FLUX_KEYS:
                df[key] = _metric(z, key)
        flux_stockees = {key: _metric(z, key) for key in FLUX_KEYS}
        dist_len = z["dist_len"].astype(np.int64)
        abund, mult = z["dist_abund"], z["dist_mult"]
        brut = "raw_tags" in z.files
        if brut:
            ntags_all, flux_all = z["raw_ntags"].astype(np.int64), z["raw_tags"].astype(np.int64)
        start = int(z["start"])
    df.insert(0, "run", start + np.arange(len(df)))

    o_d = np.r_[0, np.cumsum(dist_len)]
    o_n = np.r_[0, np.cumsum(df["Nf"].to_numpy(np.int64))]
    o_f = np.r_[0, np.cumsum(df["n_tags"].to_numpy(np.int64))]
    rows = []
    for i in range(len(df)):
        counts = np.repeat(abund[o_d[i]:o_d[i + 1]], mult[o_d[i]:o_d[i + 1]])
        assert counts.sum() == df["n_tags"].iat[i] and (counts > 0).sum() == df["n_classes"].iat[i], (cid, i)
        m = distribution_metrics(counts, reference)
        if brut:      # recalcul depuis le flux (blocs anciens sans ces colonnes)
            flux = flux_all[o_f[i]:o_f[i + 1]]
            m["turnover_50"] = turnover_blocs(ntags_all[o_n[i]:o_n[i + 1]], flux, T=T_BLOCS, k=K)
            m.update(autocorr_exces(flux))
        else:
            m.update({key: flux_stockees[key][i] for key in FLUX_KEYS})
        rows.append(m)
    return pd.concat([df, pd.DataFrame(rows, columns=NEW_KEYS)], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", required=True, help="dossier des blocs d'une grille (…/chunks)")
    parser.add_argument("--n_jobs", type=int, default=4)
    args = parser.parse_args()
    out_dir = Path(args.grid)

    pages = pages_reference()
    reference = counts_reference(pages)
    emp = em.metrics(pages, T=T_BLOCS, k=K)
    emp.update(distribution_metrics(reference, reference))
    ntags = np.array([len(p) for p in pages])
    vocab = {}
    flux = np.array([vocab.setdefault(t, len(vocab)) for p in pages for t in p])
    emp["turnover_50"] = turnover_blocs(ntags, flux, T=T_BLOCS, k=K)
    emp.update(autocorr_exces(flux))
    emp["age_effectif"] = age_effectif(flux)
    print(f"turnover empirique : {emp['turnover_50']:.4f} (tri stable), {emp['turnover_moy']:.4f} (empirical_metrics.py)")
    pd.DataFrame([emp]).to_csv(out_dir.parent / "metriques_empirique.csv", index=False)
    print(f"référence fandom : {len(pages)} pages, {reference.sum()} tags, {len(reference)} classes")

    cids = list_chunks(out_dir)
    tables = []
    with ProcessPoolExecutor(args.n_jobs) as ex:
        for j, t in enumerate(ex.map(process_chunk, [(out_dir, c, reference) for c in cids])):
            tables.append(t)
            print(f"[{j + 1}/{len(cids)}] blocs", flush=True)
    table = pd.concat(tables, ignore_index=True)
    path = out_dir.parent / "metriques.csv"
    table.to_csv(path, index=False, float_format="%.10g")
    print(f"écrit : {path} ({len(table)} runs) et {out_dir.parent / 'metriques_empirique.csv'}")


if __name__ == "__main__":
    main()
