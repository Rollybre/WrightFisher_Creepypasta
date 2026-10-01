"""
Exécution d'une grosse grille (ex. 5 millions de simulations) de simulation.py, en parallèle,
avec sauvegarde des indices de diversité ET de la distribution finale de chaque run.

Principe :
- Les simulations sont découpées en blocs de --chunk_size lignes (le bloc i couvre les lignes
  [i*chunk_size, (i+1)*chunk_size)). Chaque bloc est traité par UN worker, qui tire lui-même ses
  paramètres (rng = default_rng([rng_seed, i]), donc reproductible et indépendant des autres blocs),
  exécute ses simulations et écrit son propre fichier <out_dir>/chunk_XXXXXX.npz.
  Rien ne transite par le processus principal et rien ne s'accumule en mémoire.
- Reprise automatique : un bloc dont le fichier existe déjà est sauté. Relancer la même commande
  après une interruption reprend là où on s'était arrêté (les fichiers sont écrits en .tmp puis
  renommés : un bloc interrompu ne laisse jamais de fichier corrompu).
- La graine de simulation de la ligne k est seed_start + k.

Distribution finale stockée sous forme d'histogramme des abondances (SANS PERTE pour tous les
indices, qui ne dépendent que de la multiplicité de chaque valeur de compte) : pour chaque run,
les valeurs de compte distinctes (uint32) et le nombre de classes ayant ce compte (uint16), soit
~300 paires par run au lieu de ~2700 classes. On reconstruit le vecteur de comptes trié avec
np.repeat(abondances, multiplicités) (la valeur 0 est présente si des classes initiales n'ont jamais
été échantillonnées : le Gini les compte, pas les nombres de Hill ni Chao1). Seule l'identité des classes est perdue (quelle classe
initiale a fini avec quel compte). Voir `load_counts`.

Contenu d'un bloc (np.load) : start, p_<param>, m_<métrique>, dist_len (nb de paires par run),
dist_abund, dist_mult (concaténés sur les runs du bloc).

Usage (depuis la racine du repo) :
    python run_grid.py --n_sims 5000000 --n_jobs 16 --out_dir grid_5M
    python run_grid.py --n_sims 5000000 --n_jobs 16 --out_dir grid_5M --merge   # + results.csv
    python run_grid.py --out_dir grid_5M --upgrade --n_jobs 8                    # ajoute les métriques manquantes (ex. kl_emp) aux blocs existants
    # bornes surchargées (n_f = taille réelle des données, mu rééchelonné) : la configuration est mémorisée
    # dans <out_dir>/grid_config.json et vérifiée à chaque reprise
    python run_grid.py --n_sims 1000000 --n_jobs 16 --out_dir grid_30k --fix n_f 30901 --range mu 0 0.02 --merge
    # test rapide : python run_grid.py --n_sims 2000 --chunk_size 500 --n_jobs 4 --out_dir /tmp/g

    # Lecture :
    #   from run_grid import load_results, load_counts
    #   res = load_results("grid_5M")            # dict de np.ndarray : res["q"], res["gini"], ...
    #   counts = load_counts("grid_5M", 12345)   # vecteur de comptes trié décroissant du run 12345
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from simulation import (FIXED_PARAMS, FLOAT_RANGES, INT_RANGES, PARAM_COLUMNS, Simulation, compute_metrics,
                           draw_params, kl_divergence)

METRIC_KEYS = ["gini", "hill_0", "hill_1", "hill_2", "hill_inf", "chao1", "n_classes", "kl_emp", "gini_obs"]
SIM_ARGS = ["n_i", "n_f", "C", "T", "alpha", "mu", "q"]


def chunk_path(out_dir, chunk_id):
    """Chemin du fichier du bloc `chunk_id`."""
    return os.path.join(out_dir, f"chunk_{chunk_id:06d}.npz")


def abundance_histogram(counts):
    """
    Histogramme des abondances d'un vecteur de comptes par classe. Les classes de compte 0 (classes
    initiales jamais échantillonnées) sont conservées : elles comptent dans n pour le Gini.

    Retour : (valeurs distinctes de compte en uint32 croissantes, nb de classes par valeur en uint16).
    """
    values, mult = np.unique(counts, return_counts=True)
    if mult.max() > np.iinfo(np.uint16).max:
        raise OverflowError(f"{mult.max()} classes de même compte : dépasse uint16")
    return values.astype(np.uint32), mult.astype(np.uint16)


def run_chunk(chunk_id, chunk_size, n_sims, out_dir, rng_seed, seed_start, space=None):
    """
    Traite un bloc de lignes : tire les paramètres, exécute les simulations, écrit
    <out_dir>/chunk_<id>.npz (via un .tmp renommé). Sans effet si le fichier existe déjà.

    `space` : dict {float_ranges, int_ranges, fixed} définissant l'espace de tirage (cf. `build_space`) ;
    None = bornes par défaut de simulation.py.

    Retour : (chunk_id, nombre de simulations exécutées, durée en s).
    """
    path = chunk_path(out_dir, chunk_id)
    if os.path.exists(path):
        return chunk_id, 0, 0.0

    t0 = time.perf_counter()
    start = chunk_id * chunk_size
    n = min(chunk_size, n_sims - start)
    params = draw_params(np.random.default_rng([rng_seed, chunk_id]), n, **(space or {}))
    params["seed"] = seed_start + start + np.arange(n)

    metrics = {k: np.empty(n) for k in METRIC_KEYS}
    dist_len = np.empty(n, dtype=np.uint16)
    abund, mult = [], []
    for i in range(n):
        args = {k: params[k][i].item() for k in SIM_ARGS}
        sim = Simulation(**args, seed=int(params["seed"][i])).run()
        for k in METRIC_KEYS:
            metrics[k][i] = sim.metrics[k]
        values, m = abundance_histogram(sim.archive)
        dist_len[i] = values.size
        abund.append(values)
        mult.append(m)

    payload = {"start": np.int64(start), "dist_len": dist_len,
               "dist_abund": np.concatenate(abund), "dist_mult": np.concatenate(mult)}
    payload.update({f"p_{k}": v for k, v in params.items()})
    payload.update({f"m_{k}": v for k, v in metrics.items()})

    tmp = path[:-len(".npz")] + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)
    return chunk_id, n, time.perf_counter() - t0


def _chunk_metric(z, key):
    """
    Métrique `key` d'un bloc ouvert (np.load). Si le bloc a été écrit avant l'ajout de cette métrique
    (ex. kl_emp sur la grille 5M v3), elle est recalculée à la volée depuis les distributions stockées :
    les histogrammes d'abondances permettent de reconstruire exactement le vecteur de comptes de chaque run.
    """
    name = f"m_{key}"
    if name in z.files:
        return z[name]
    lens = z["dist_len"].astype(np.int64)
    ends = np.cumsum(lens)
    abund, mult = z["dist_abund"], z["dist_mult"]
    fn = kl_divergence if key == "kl_emp" else (lambda c: compute_metrics(c)[key])
    out = np.empty(len(lens))
    for i in range(len(lens)):
        a, b = ends[i] - lens[i], ends[i]
        out[i] = fn(np.repeat(abund[a:b], mult[a:b].astype(np.int64)))
    return out


def build_space(ranges=(), fixes=()):
    """
    Espace de tirage = bornes par défaut de simulation.py avec surcharges.

    Paramètres :
    - ranges : liste de (nom, min, max) : remplace les bornes d'un paramètre déjà tiré au hasard
               (ex. ("mu", 0, 0.02), ("C", 20, 100))
    - fixes  : liste de (nom, valeur) : fixe un paramètre à une constante, qu'il soit tiré ou déjà fixe
               (ex. ("n_f", 30901))

    Retour : dict {float_ranges, int_ranges, fixed}, sérialisable en JSON.
    """
    fr, ir, fx = dict(FLOAT_RANGES), dict(INT_RANGES), dict(FIXED_PARAMS)
    for name, lo, hi in ranges:
        if name in fr:
            fr[name] = (float(lo), float(hi))
        elif name in ir:
            ir[name] = (int(float(lo)), int(float(hi)))
        else:
            raise ValueError(f"--range : '{name}' n'est pas un paramètre tiré au hasard ({sorted([*fr, *ir])}) ; utiliser --fix")
    for name, value in fixes:
        if name not in PARAM_COLUMNS or name == "seed":
            raise ValueError(f"--fix : paramètre inconnu '{name}' ({[c for c in PARAM_COLUMNS if c != 'seed']})")
        as_int = name in ir or isinstance(fx.get(name), int)
        fr.pop(name, None); ir.pop(name, None)
        fx[name] = int(float(value)) if as_int else float(value)
    return {"float_ranges": {k: list(v) for k, v in fr.items()}, "int_ranges": {k: list(v) for k, v in ir.items()},
            "fixed": fx}


def check_config(out_dir, config):
    """
    Écrit <out_dir>/grid_config.json (paramètres qui définissent le contenu des blocs) au premier
    lancement ; aux lancements suivants (reprise), vérifie qu'ils sont identiques et lève une
    erreur sinon : reprendre avec d'autres bornes mélangerait deux grilles dans un même dossier.
    """
    path = os.path.join(out_dir, "grid_config.json")
    config = json.loads(json.dumps(config))
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        diff = {k: (old.get(k), config.get(k)) for k in config if old.get(k) != config.get(k)}
        if diff:
            raise SystemExit(f"{out_dir} contient déjà une grille avec une autre configuration (ancien, nouveau) : {diff}\n"
                             "Utiliser un autre --out_dir, ou les mêmes options pour reprendre.")
    else:
        if list_chunks(out_dir):
            print("ATTENTION : blocs existants sans grid_config.json ; on suppose qu'ils viennent de cette configuration.")
        with open(path, "w") as f:
            json.dump(config, f, indent=1)


def list_chunks(out_dir):
    """Liste triée des identifiants de blocs présents (fichiers terminés uniquement)."""
    ids = [int(f[6:12]) for f in os.listdir(out_dir)
           if f.startswith("chunk_") and f.endswith(".npz") and ".tmp" not in f]
    return sorted(ids)


def _upgrade_chunk(path):
    """Ajoute à un bloc les métriques de METRIC_KEYS qui lui manquent (recalculées depuis ses distributions) ; réécriture atomique. Retourne le nb de métriques ajoutées."""
    with np.load(path) as z:
        missing = [k for k in METRIC_KEYS if f"m_{k}" not in z.files]
        if not missing:
            return 0
        payload = {name: z[name] for name in z.files}
        for k in missing:
            payload[f"m_{k}"] = _chunk_metric(z, k)
    tmp = path[:-len(".npz")] + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)
    return len(missing)


def upgrade_chunks(out_dir, n_jobs=1):
    """
    Met à jour tous les blocs d'une grille existante avec les métriques ajoutées depuis (ex. kl_emp) :
    calculées depuis les distributions stockées, sans relancer aucune simulation. Sans effet sur les
    blocs déjà à jour. Retour : nombre de blocs modifiés.
    """
    paths = [chunk_path(out_dir, c) for c in list_chunks(out_dir)]
    if n_jobs == 1:
        done = [_upgrade_chunk(p) for p in paths]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            done = list(pool.map(_upgrade_chunk, paths))
    return sum(d > 0 for d in done)


def load_results(out_dir):
    """
    Charge paramètres et métriques de tous les blocs terminés, dans l'ordre des lignes.

    Retour : dict {nom: np.ndarray} avec les clés de PARAM_COLUMNS puis de METRIC_KEYS.
    """
    columns = {k: [] for k in PARAM_COLUMNS + METRIC_KEYS}
    for cid in list_chunks(out_dir):
        with np.load(chunk_path(out_dir, cid)) as z:
            for k in PARAM_COLUMNS:
                columns[k].append(z[f"p_{k}"])
            for k in METRIC_KEYS:
                columns[k].append(_chunk_metric(z, k))
    return {k: np.concatenate(v) for k, v in columns.items()}


def load_counts(out_dir, run_index, chunk_size=None):
    """
    Reconstruit le vecteur de comptes (trié décroissant) de la simulation `run_index`.

    Paramètres :
    - out_dir    : dossier de la grille
    - run_index  : numéro de ligne global (0 = première simulation)
    - chunk_size : taille de bloc utilisée au lancement ; si None, déduite du 1er bloc
    """
    if chunk_size is None:
        with np.load(chunk_path(out_dir, 0)) as z:
            chunk_size = len(z["dist_len"])
    cid, i = divmod(run_index, chunk_size)
    with np.load(chunk_path(out_dir, cid)) as z:
        lens = z["dist_len"].astype(np.int64)
        offset = lens[:i].sum()
        values = z["dist_abund"][offset:offset + lens[i]]
        mult = z["dist_mult"][offset:offset + lens[i]]
    return np.repeat(values, mult)[::-1]


def merge_csv(out_dir, output):
    """
    Écrit un CSV (paramètres + métriques, une ligne par run) à partir des blocs, bloc par bloc.

    Les paramètres flottants (alpha, mu, q) sont écrits avec 17 chiffres significatifs (exacts) :
    la simulation est très sensible à ces valeurs (un arrondi à 1e-11 sur q suffit à changer le
    résultat d'un run), donc relire le CSV avec moins de précision ne permet plus de reproduire un
    run à partir de sa ligne. Les métriques sont écrites avec 10 chiffres.
    """
    header = PARAM_COLUMNS + METRIC_KEYS
    int_cols = {"C", "n_i", "n_f", "T", "seed", "n_classes"}
    fmt = ["%d" if k in int_cols else ("%.17g" if k in PARAM_COLUMNS else "%.10g") for k in header]
    with open(output, "w") as f:
        f.write(",".join(header) + "\n")
        for cid in list_chunks(out_dir):
            with np.load(chunk_path(out_dir, cid)) as z:
                block = np.column_stack([z[f"p_{k}"] if k in PARAM_COLUMNS else _chunk_metric(z, k) for k in header])
            np.savetxt(f, block, fmt=fmt, delimiter=",")
    return output


def main():
    parser = argparse.ArgumentParser(description="Grille de simulations par blocs, avec distributions finales.")
    parser.add_argument("--n_sims", type=int, default=None, help="nombre total de simulations (requis sauf avec --upgrade)")
    parser.add_argument("--out_dir", required=True, help="dossier de sortie (créé si besoin, reprise si existant)")
    parser.add_argument("--n_jobs", type=int, default=16, help="nb de processus (défaut 16 ; -1 = tous les threads)")
    parser.add_argument("--chunk_size", type=int, default=5000, help="simulations par bloc/fichier (défaut 5000)")
    parser.add_argument("--rng_seed", type=int, default=0, help="graine du tirage des paramètres")
    parser.add_argument("--seed_start", type=int, default=1, help="graine de simulation de la ligne 0")
    parser.add_argument("--range", nargs=3, action="append", default=[], metavar=("NOM", "MIN", "MAX"),
                        help="remplace les bornes d'un paramètre tiré au hasard (répétable), ex. --range mu 0 0.02")
    parser.add_argument("--fix", nargs=2, action="append", default=[], metavar=("NOM", "VALEUR"),
                        help="fixe un paramètre à une constante (répétable), ex. --fix n_f 30901")
    parser.add_argument("--upgrade", action="store_true",
                        help="ajoute aux blocs existants de --out_dir les métriques manquantes (ex. kl_emp), recalculées "
                             "depuis les distributions stockées, puis quitte (aucune simulation)")
    parser.add_argument("--merge", action="store_true", help="à la fin, écrit <out_dir>/results.csv (params + métriques)")
    args = parser.parse_args()

    n_jobs = (os.cpu_count() or 1) if args.n_jobs == -1 else args.n_jobs
    if args.upgrade:
        t0 = time.perf_counter()
        n = upgrade_chunks(args.out_dir, n_jobs)
        print(f"{n} blocs mis à jour dans {args.out_dir} ({time.perf_counter() - t0:.0f} s)")
        return
    if args.n_sims is None:
        parser.error("--n_sims est requis (sauf avec --upgrade)")
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        space = build_space(args.range, args.fix)
    except ValueError as e:
        parser.error(str(e))
    check_config(args.out_dir, {"rng_seed": args.rng_seed, "seed_start": args.seed_start,
                                "chunk_size": args.chunk_size, **space})
    print("espace de tirage :", json.dumps(space), flush=True)
    n_chunks = -(-args.n_sims // args.chunk_size)
    for f in os.listdir(args.out_dir):        # blocs interrompus : fichiers temporaires orphelins
        if f.endswith(".tmp.npz"):
            os.remove(os.path.join(args.out_dir, f))
    todo = [c for c in range(n_chunks) if not os.path.exists(chunk_path(args.out_dir, c))]
    print(f"{args.n_sims} simulations = {n_chunks} blocs de {args.chunk_size} ; "
          f"{n_chunks - len(todo)} déjà faits, {len(todo)} à faire ; n_jobs={n_jobs}", flush=True)

    t0 = time.perf_counter()
    done_sims = 0
    total_todo = sum(min(args.chunk_size, args.n_sims - c * args.chunk_size) for c in todo)
    if todo:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(todo))) as pool:
            futures = [pool.submit(run_chunk, c, args.chunk_size, args.n_sims, args.out_dir,
                                   args.rng_seed, args.seed_start, space) for c in todo]
            try:
                for k, fut in enumerate(as_completed(futures), 1):
                    _, n, _ = fut.result()
                    done_sims += n
                    elapsed = time.perf_counter() - t0
                    eta = elapsed / done_sims * (total_todo - done_sims)
                    print(f"[{k}/{len(todo)}] {done_sims}/{total_todo} sims, {elapsed / 60:.1f} min écoulées, "
                          f"reste ~{eta / 60:.0f} min", flush=True)
            except KeyboardInterrupt:
                print("Interrompu : relancer la même commande pour reprendre.", file=sys.stderr)
                pool.shutdown(wait=False, cancel_futures=True)
                raise SystemExit(130)
    print(f"Terminé en {(time.perf_counter() - t0) / 60:.1f} min.")

    if args.merge:
        out = merge_csv(args.out_dir, os.path.join(args.out_dir, "results.csv"))
        print(f"CSV écrit : {out}")


if __name__ == "__main__":
    main()
