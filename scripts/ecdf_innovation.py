"""
Calendrier empirique des innovations, pour simulation_ecdf.py.

Pour chaque plateforme, position (numéro de page, dans l'ordre chronologique) de la première apparition de chaque
classe de tag. La fonction de répartition empirique (ecdf) de ces positions, ramenée à la fraction de pages écoulées,
donne le nombre de classes apparues après n pages : c'est la courbe que suivent les innovations dans
simulation_ecdf.py. Mêmes corpus et mêmes filtres que scripts/empirical_metrics.py (pages sans tag retirées, comme
dans metriques_distribution.pages_reference).

Écrit data/ecdf_innovation.json : {plateforme: {"n_pages": N, "premiere_page": [indices triés]}}. Ce fichier est
versionné avec le code : le serveur n'a pas besoin des corpus.

Usage (depuis simulations/v3/, en local) :
    python scripts/ecdf_innovation.py                    # fandom
    python scripts/ecdf_innovation.py --only fandom scp_en backrooms
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import empirical_metrics as em  # noqa: E402

OUT = ROOT / "data" / "ecdf_innovation.json"


def premieres_pages(pages):
    """Indice de la page où chaque classe apparaît pour la première fois (triés croissants)."""
    vues, premiere = set(), []
    for i, tags in enumerate(pages):
        for t in tags:
            if t not in vues:
                vues.add(t)
                premiere.append(i)
    return premiere


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", default=["fandom"], help="plateformes (défaut : fandom)")
    args = parser.parse_args()

    sources = {"fandom": em.pages_fandom}
    sources.update({f"scp_{l}": (lambda l=l: em.pages_wiki(f"scp_{l}", em.SITE / "scp" / l / "tags_long.csv"))
                    for l in em.SCP_LANGS})
    sources["backrooms"] = lambda: em.pages_wiki("backrooms", em.SITE / "backrooms" / "tags_long.csv")

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in args.only:
        pages = [p for p in sources[name]() if p]
        prem = premieres_pages(pages)
        out[name] = {"n_pages": len(pages), "premiere_page": prem}
        print(f"{name:10s} {len(pages):6d} pages, {len(prem):5d} classes ; "
              f"apparues dans le premier 1 % des pages : {sum(p < len(pages) / 100 for p in prem)}")
    OUT.write_text(json.dumps(out))
    print(f"écrit : {OUT}")


if __name__ == "__main__":
    main()
