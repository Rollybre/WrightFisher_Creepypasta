"""
Valeurs empiriques des métriques du modèle v3, calculées sur les plateformes réelles avec EXACTEMENT les mêmes
fonctions que pour les simulations (simulation.py). Écrit data/empirical_metrics.json, lu par simulation.py
pour afficher la comparaison à la fin d'un run.

Correspondance modèle / données :
- un individu = une page (une histoire) ; ses tags = l'ensemble de ses tags (doublons retirés) ;
- l'ordre d'arrivée = la date de création (tri stable : à date égale, ordre du fichier) ;
- f à la génération t = fréquences cumulées des tags des pages arrivées avant la fin du bloc t, les pages étant
  découpées en T blocs de taille égale (même rôle que les T générations ; le turnover dépend donc de T).

Sources et filtrage (identiques à simulation_V2) :
- fandom (wiki Creepypasta) : ../v2/data/fandom_data.csv, catégories « A|B » éclatées ;
- scp_<langue>, backrooms : ../../scraping/data/.../tags_long.csv, sans tags système ni tags structurels/formels
  (listes de simulations/v2/scripts/platform_data.py).

Usage (depuis simulations/v3/, en local : pandas requis) :
    python scripts/empirical_metrics.py                 # toutes les plateformes, T = 50
    python scripts/empirical_metrics.py --only fandom --T 100
"""
import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
UCL = ROOT.parents[1]  # racine ucl_stay
sys.path.insert(0, str(ROOT))
from simulation import autocorrelation, flux_tags, heaps, pente_queue, turnover  # noqa: E402

OUT = ROOT / "data" / "empirical_metrics.json"
FANDOM = UCL / "simulations" / "v2" / "data" / "fandom_data.csv"
SITE = UCL / "scraping" / "data"
SCP_LANGS = ("en", "fr", "ru", "jp", "cn", "es")


def _platform_filters():
    """is_system et structural_category de simulations/v2/scripts/platform_data.py (même filtrage qu'en v2)."""
    spec = importlib.util.spec_from_file_location("platform_data", UCL / "simulations" / "v2" / "scripts" / "platform_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.is_system, mod.structural_category


def pages_fandom():
    """Liste chronologique des ensembles de tags des pages du corpus fandom."""
    df = pd.read_csv(FANDOM)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", kind="stable")
    pages = []
    for lst in df["Category"].apply(ast.literal_eval):
        tags = list(dict.fromkeys(p.strip() for c in lst for p in c.split("|") if p.strip()))
        pages.append(tags)
    return pages


def pages_wiki(platform, path):
    """Liste chronologique des ensembles de tags des pages d'un wiki (tags_long.csv), après filtrage v2."""
    is_system, structural_category = _platform_filters()
    df = pd.read_csv(path, usecols=["id", "created_at", "tag"]).dropna(subset=["tag"])
    df = df[~df["tag"].map(is_system)]
    df = df[df["tag"].map(lambda t: structural_category(platform, t)).isna()]
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="ISO8601")
    first = df.groupby("id", sort=False)["created_at"].min().sort_values(kind="stable")
    tags = df.groupby("id", sort=False)["tag"].agg(lambda s: list(dict.fromkeys(s)))
    return [tags[i] for i in first.index]


def lam_poisson_tronquee(moyenne):
    """lam tel que E[X | X >= 1] = moyenne pour X ~ Poisson(lam) : lam / (1 - exp(-lam)) = moyenne."""
    if moyenne <= 1:
        return 0.0
    return brentq(lambda l: l / (1 - np.exp(-l)) - moyenne, 1e-9, 100)


def metrics(pages, T=50, k=10, max_lag=100):
    """Métriques du modèle v3 sur une liste chronologique de pages (listes de tags)."""
    pages = [p for p in pages if p]                       # pages sans tag thématique : pas d'individu
    vocab = {}
    d = {i: np.array([vocab.setdefault(t, len(vocab)) for t in p], dtype=int) for i, p in enumerate(pages)}
    flux = flux_tags(d)
    C = len(vocab)
    counts = np.bincount(flux, minlength=C)
    f_final = counts / counts.sum()

    # historique : f cumulé à la fin de chacun des T blocs de pages
    bornes = np.round(np.linspace(0, len(pages), T + 1)).astype(int)[1:]
    fin_tags = np.cumsum([len(p) for p in pages])
    H = np.zeros((T, C))
    for t, b in enumerate(bornes):
        c = np.bincount(flux[:fin_tags[b - 1]], minlength=C)
        H[t] = c / c.sum()

    _, beta = heaps(flux)
    _, Cl, c_ref = autocorrelation(flux, max_lag=max_lag)
    n_par_page = np.array([len(p) for p in pages])
    return {
        "n_pages": len(pages),
        "n_tags": int(len(flux)),
        "n_classes": C,
        "tags_par_page": float(n_par_page.mean()),
        "part_1_tag": float((n_par_page == 1).mean()),
        "lam_estime": float(lam_poisson_tronquee(n_par_page.mean())),
        "alpha": float(pente_queue(f_final)),
        "beta": float(beta),
        "turnover_moy": float(turnover(H, k=k).mean()),
        "autocorr_1": float(Cl[0]),
        f"autocorr_{max_lag}": float(Cl[-1]),
        "autocorr_ref": float(c_ref),
        "T": T, "k": k, "max_lag": max_lag,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", default=None, help="plateformes (défaut : toutes)")
    parser.add_argument("--T", type=int, default=50, help="nombre de blocs (générations) pour le turnover")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max_lag", type=int, default=100)
    args = parser.parse_args()

    sources = {"fandom": pages_fandom}
    sources.update({f"scp_{l}": (lambda l=l: pages_wiki(f"scp_{l}", SITE / "scp" / l / "tags_long.csv")) for l in SCP_LANGS})
    sources["backrooms"] = lambda: pages_wiki("backrooms", SITE / "backrooms" / "tags_long.csv")
    if args.only:
        sources = {k: v for k, v in sources.items() if k in args.only}

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, load in sources.items():
        out[name] = metrics(load(), T=args.T, k=args.k, max_lag=args.max_lag)
        m = out[name]
        print(f"{name:10s} pages {m['n_pages']:6d}  tags {m['n_tags']:7d}  classes {m['n_classes']:5d}  "
              f"tags/page {m['tags_par_page']:.2f}  alpha {m['alpha']:.2f}  beta {m['beta']:.2f}  "
              f"turnover {m['turnover_moy']:.2f}", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"écrit : {OUT}")


if __name__ == "__main__":
    main()
