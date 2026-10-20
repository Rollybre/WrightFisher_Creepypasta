"""
Exécution d'une grande grille de simulations (simulation.py) en parallèle, sur serveur.

Même principe que simulations/v2/run_grid.py :
- Les simulations sont découpées en blocs de --chunk_size lignes (le bloc i couvre les lignes
  [i*chunk_size, (i+1)*chunk_size)). Chaque bloc est traité par UN worker, qui tire lui-même ses
  paramètres (rng = default_rng([rng_seed, i]) : reproductible et indépendant des autres blocs),
  exécute ses simulations et écrit son propre fichier <out_dir>/chunk_XXXXXX.npz.
- Reprise automatique : un bloc dont le fichier existe déjà est sauté. Les fichiers sont écrits en
  .tmp puis renommés : un bloc interrompu ne laisse jamais de fichier corrompu.
- La graine de simulation de la ligne k est seed_start + k.
- <out_dir>/grid_config.json mémorise l'espace de tirage et refuse une reprise avec une autre
  configuration (sinon deux grilles se mélangeraient dans le même dossier).

Q n'est pas un paramètre de la grille : il est tiré dans simulation() à partir de la graine (6 valeurs
~ U(0, q_max), la dernière répétée). Ses 6 valeurs sont stockées comme sorties (q_0 ... q_5). --q_max vaut 4 par
défaut depuis le 2026-10-15 ; les grilles antérieures utilisaient 2 (pas de clé q_max dans leur grid_config.json :
pour les reprendre ou rejouer leurs runs, --q_max 2 / simulation(..., q_max=2), qui est le défaut de simulation()).

Modération (2026-10-19) : K (nombre de passes, passe finale à Nf comprise) et j (seuil de rareté, en compte de
tags) sont des paramètres de la grille, mais seulement si on les donne : --range K 0 20 / --fix K 5, idem pour j.
Sans ces options, l'espace de tirage et grid_config.json sont inchangés (K=0 : pas de modération, simulation
identique à l'ancienne), ce qui garde les anciennes grilles reprenables. Les blocs sans p_K / p_j se lisent
comme K=0, j=2 (cf. _param), et les métriques de modération (n_passes, n_fusionnees, n_doublons) comme 0.
Avec K > 0, les données brutes (raw_*) décrivent la population APRÈS modération : historique(run) n'est alors
plus identique à simulation()["historique"] (qui garde les f d'avant chaque passe).

Fitness (2026-10-19) : sigma (dispersion log-normale des fitness des classes, cf. simulation_opti.simulation) se
donne de la même façon (--range sigma 0 2 / --fix sigma 1) ; absent : sigma = 0 (pas de fitness), lu comme 0.
Métrique age_effectif (Spearman ordre d'apparition / effectif final) ajoutée le même jour : NaN pour les blocs anciens.

Distribution finale stockée sous forme d'histogramme des abondances (sans perte pour la courbe
fréquence-rang) : pour chaque run, les valeurs de compte distinctes et le nombre de classes ayant ce
compte. Voir `load_counts`.

Données brutes (sans perte) : chaque bloc garde aussi, pour chaque run, le nombre de tags de chaque individu
(dans l'ordre d'arrivée) et le flux de tous ses tags, ainsi que le nombre de classes à la fin de chaque
génération. On peut donc tout recalculer après coup : population finale, f à chaque génération (identique à
`historique`), distributions, Heaps, autocorrélation... Voir `load_run` et `historique`. Q est constant au sein
d'un run (tiré une fois) : ses 6 valeurs suffisent.

Contenu d'un bloc (np.load) : start, p_<param>, m_<métrique>, dist_len, dist_abund, dist_mult,
raw_ntags (tags par individu, Nf valeurs par run), raw_tags (flux, n_tags valeurs par run),
raw_C_par_gen (T valeurs par run).

Mode léger (--sans_brut) : pour les grandes grilles. Les raw_* ne sont pas écrits (environ 99 % du volume) ;
on garde la distribution finale (dist_*) et les métriques. Les métriques qui demandent le flux (turnover_50,
autocorrélation aux décalages 10-100) sont calculées pendant le run, dans les deux modes ; celles qui ne
demandent que la distribution finale (KL, Hill, Gini, Chao1, singletons) se calculent après coup avec
scripts/metriques_distribution.py. load_run et historique ne marchent pas sur une grille légère : rejouer le run
avec simulation() (même graine, même version de numpy).

Usage (depuis simulations/v3/) :
    python run_grid.py --n_sims 100000 --n_jobs 16 --out_dir results/grid_base/chunks
    python run_grid.py --n_sims 100000 --n_jobs 16 --out_dir results/grid_base/chunks --merge   # + results.csv
    # bornes surchargées / paramètres fixés (mémorisés dans grid_config.json) :
    python run_grid.py --n_sims 50000 --out_dir results/grid_mu/chunks --range mu 0 0.02 --fix Nf 30901
    # grande grille, sans données brutes :
    python run_grid.py --n_sims 1000000 --out_dir results/grid_big/chunks --sans_brut --merge
    # avec modération (K passes, seuil j) :
    python run_grid.py --n_sims 100000 --out_dir results/grid_mod/chunks --range K 0 20 --range j 2 10 --sans_brut
    # test rapide : python run_grid.py --n_sims 200 --chunk_size 50 --n_jobs 4 --out_dir /tmp/g

    # Lecture :
    #   from run_grid import load_results, load_counts
    #   res = load_results("results/grid_base/chunks")     # dict de np.ndarray : res["mu"], res["beta"], ...
    #   counts = load_counts("results/grid_base/chunks", 123)   # comptes triés décroissants du run 123
    #   run = load_run("results/grid_base/chunks", 123)         # paramètres, Q, d, flux, C_par_gen du run 123
    #   H = historique(run)                                     # f à chaque génération, (T, C final)
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from simulation_opti import (age_effectif, autocorr_exces, autocorrelation, flux_tags, heaps, pente_queue, simulation,
                             turnover, turnover_blocs)

# Espace de tirage par défaut (uniforme, indépendant par paramètre)
FLOAT_RANGES = {"mu": (0.0, 0.1), "lam": (1.0, 4.0)}          # uniforme continu
INT_RANGES = {"C": (2, 200), "Ni": (10, 1000), "T": (20, 500)}  # uniforme entier (bornes incluses)
FIXED_PARAMS = {"Nf": 10000}                                     # constants pour toutes les lignes
PARAM_COLUMNS = ["Ni", "Nf", "C", "T", "lam", "mu", "seed", "K", "j", "sigma"]   # K, j en fin : colonnes existantes inchangées
SIM_ARGS = ["Ni", "Nf", "C", "T", "lam", "mu"]
# modération : paramètres de simulation() présents dans l'espace seulement si --range / --fix les donnent
MOD_ARGS = ["K", "j", "sigma"]
MOD_DEFAULTS = {"K": 0, "j": 2, "sigma": 0.0}   # = défauts de simulation() : ni modération ni fitness
INT_PARAMS = {"Ni", "Nf", "C", "T", "K", "j", "seed"}

N_Q = 6  # nombre de valeurs distinctes de Q (cf. simulation())
METRIC_KEYS = ([f"q_{i}" for i in range(N_Q)]
               + ["n_tags", "C_final", "n_classes", "alpha", "beta", "turnover_moy", "autocorr_1", "autocorr_ref",
                  "turnover_50", "autocorr_100", "exces_100", "exces_10_100",
                  "n_passes", "n_fusionnees", "n_doublons", "age_effectif"])
# Métriques ajoutées le 2026-10-15 : absentes des blocs plus anciens (lues comme NaN, cf. _metric)
NEW_METRIC_KEYS = {"turnover_50", "autocorr_100", "exces_100", "exces_10_100", "age_effectif"}   # age_effectif : 2026-10-19
# Métriques de modération (2026-10-19) : absentes des blocs plus anciens, lues comme 0 (pas de modération)
MOD_METRIC_KEYS = {"n_passes", "n_fusionnees", "n_doublons"}


def draw_params(rng, n, float_ranges, int_ranges, fixed):
    """Tire n jeux de paramètres : uniforme continu (float_ranges), uniforme entier bornes incluses
    (int_ranges), constantes (fixed). Retour : dict {nom: np.ndarray de taille n}."""
    columns = {name: rng.uniform(min(b), max(b), size=n) for name, b in float_ranges.items()}
    columns.update({name: rng.integers(min(b), max(b), size=n, endpoint=True) for name, b in int_ranges.items()})
    columns.update({name: np.full(n, value) for name, value in fixed.items()})
    return columns


def build_space(ranges=(), fixes=()):
    """
    Espace de tirage = bornes par défaut avec surcharges.
    - ranges : liste de (nom, min, max) : nouvelles bornes d'un paramètre tiré (ex. ("mu", 0, 0.02))
    - fixes  : liste de (nom, valeur) : fixe un paramètre (ex. ("Nf", 30901))
    Retour : dict {float_ranges, int_ranges, fixed}, sérialisable en JSON.
    """
    fr, ir, fx = dict(FLOAT_RANGES), dict(INT_RANGES), dict(FIXED_PARAMS)
    for name, lo, hi in ranges:
        if name in INT_PARAMS and name != "seed":
            fx.pop(name, None); fr.pop(name, None)
            ir[name] = (int(float(lo)), int(float(hi)))
        elif name in SIM_ARGS + MOD_ARGS:
            fx.pop(name, None); ir.pop(name, None)
            fr[name] = (float(lo), float(hi))
        else:
            raise ValueError(f"--range : paramètre inconnu '{name}' ({SIM_ARGS + MOD_ARGS})")
    for name, value in fixes:
        if name not in SIM_ARGS + MOD_ARGS:
            raise ValueError(f"--fix : paramètre inconnu '{name}' ({SIM_ARGS + MOD_ARGS})")
        fr.pop(name, None); ir.pop(name, None)
        fx[name] = int(float(value)) if name in INT_PARAMS else float(value)

    # garde-fous : la population doit croître et Ni >= 1
    ni_max = ir["Ni"][1] if "Ni" in ir else fx["Ni"]
    ni_min = ir["Ni"][0] if "Ni" in ir else fx["Ni"]
    nf_min = ir["Nf"][0] if "Nf" in ir else fx["Nf"]
    if ni_min < 1 or ni_max >= nf_min:
        raise ValueError(f"il faut 1 <= Ni < Nf pour toutes les lignes (Ni max {ni_max}, Nf min {nf_min})")
    # modération : K >= 0 passes, seuil j >= 1 (j = 1 : aucune classe rare)
    for name, mini in (("K", 0), ("j", 1), ("sigma", 0)):
        lo = (ir.get(name) or fr.get(name) or [fx.get(name, mini)])[0]
        if lo < mini:
            raise ValueError(f"il faut {name} >= {mini} ({name} min {lo})")
    return {"float_ranges": {k: list(v) for k, v in fr.items()}, "int_ranges": {k: list(v) for k, v in ir.items()},
            "fixed": fx}


def run_metrics(res, k=10, max_lag=100):
    """Métriques scalaires d'un run (dict renvoyé par simulation()) et vecteur de comptes final."""
    flux = flux_tags(res["d"])
    _, beta = heaps(flux)
    _, Cl, c = autocorrelation(flux, max_lag=max_lag)
    to = turnover(res["historique"], k=k)
    ntags = np.array([len(res["d"][j]) for j in sorted(res["d"])])
    counts = np.bincount(flux, minlength=res["C"])
    m = {f"q_{i}": res["Q"][i] for i in range(N_Q)}
    m.update({
        "n_tags": len(flux),
        "C_final": res["C"],
        "n_classes": int((counts > 0).sum()),
        "alpha": pente_queue(res["f_final"]),
        "beta": beta,
        "turnover_moy": to.mean() if len(to) else np.nan,
        "autocorr_1": Cl[0],
        "autocorr_ref": c,
        "turnover_50": turnover_blocs(ntags, flux, k=k),
        # modération (0 si désactivée, ou pour un dict qui n'a pas ces clés)
        **{key: res.get(key, 0) for key in MOD_METRIC_KEYS},
        "age_effectif": age_effectif(flux),
    })
    m.update(autocorr_exces(flux))
    return m, counts


def abundance_histogram(counts):
    """Histogramme des abondances : (valeurs de compte distinctes, nb de classes par valeur)."""
    values, mult = np.unique(counts, return_counts=True)
    return values.astype(np.uint32), mult.astype(np.uint32)


def chunk_path(out_dir, chunk_id):
    return os.path.join(out_dir, f"chunk_{chunk_id:06d}.npz")


def run_chunk(chunk_id, chunk_size, n_sims, out_dir, rng_seed, seed_start, space, k, max_lag, brut=True, q_max=2.0):
    """
    Traite un bloc : tire les paramètres, exécute les simulations, écrit <out_dir>/chunk_<id>.npz
    (via un .tmp renommé). Sans effet si le fichier existe déjà.
    brut=False : n'écrit pas les données brutes (raw_*), seulement distribution finale et métriques.
    Retour : (chunk_id, nombre de simulations exécutées, durée en s).
    """
    path = chunk_path(out_dir, chunk_id)
    if os.path.exists(path):
        return chunk_id, 0, 0.0

    t0 = time.perf_counter()
    start = chunk_id * chunk_size
    n = min(chunk_size, n_sims - start)
    params = draw_params(np.random.default_rng([rng_seed, chunk_id]), n, **space)
    params["seed"] = seed_start + start + np.arange(n)

    metrics = {key: np.empty(n) for key in METRIC_KEYS}
    dist_len = np.empty(n, dtype=np.uint32)
    abund, mult = [], []
    raw_ntags, raw_tags, raw_C = [], [], []
    for i in range(n):
        args = {key: params[key][i].item() for key in SIM_ARGS + MOD_ARGS if key in params}
        res = simulation(**args, seed=int(params["seed"][i]), q_max=q_max)
        m, counts = run_metrics(res, k=k, max_lag=max_lag)
        for key in METRIC_KEYS:
            metrics[key][i] = m[key]
        values, mu = abundance_histogram(counts)
        dist_len[i] = values.size
        abund.append(values)
        mult.append(mu)
        if brut:
            raw_ntags.append(np.array([len(res["d"][j]) for j in sorted(res["d"])], dtype=np.uint16))
            raw_tags.append(flux_tags(res["d"]))
            raw_C.append(res["C_par_gen"])

    payload = {"start": np.int64(start), "dist_len": dist_len,
               "dist_abund": np.concatenate(abund), "dist_mult": np.concatenate(mult)}
    if brut:
        payload["raw_ntags"] = np.concatenate(raw_ntags)
        payload["raw_C_par_gen"] = np.concatenate(raw_C).astype(np.uint32)
        tags = np.concatenate(raw_tags)
        payload["raw_tags"] = tags.astype(np.uint16 if tags.max(initial=0) < 2 ** 16 else np.uint32)
    payload.update({f"p_{key}": v for key, v in params.items()})   # p_K, p_j seulement si dans l'espace
    payload.update({f"m_{key}": v for key, v in metrics.items()})

    tmp = path[:-len(".npz")] + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)
    return chunk_id, n, time.perf_counter() - t0


def check_config(out_dir, config):
    """Écrit <out_dir>/grid_config.json au premier lancement ; à la reprise, vérifie qu'il est identique."""
    path = os.path.join(out_dir, "grid_config.json")
    config = json.loads(json.dumps(config))
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        diff = {key: (old.get(key), config.get(key)) for key in set(old) | set(config) if old.get(key) != config.get(key)}
        if diff:
            raise SystemExit(f"{out_dir} contient déjà une grille avec une autre configuration (ancien, nouveau) : {diff}\n"
                             "Utiliser un autre --out_dir, ou les mêmes options pour reprendre.")
    else:
        if list_chunks(out_dir):
            print("ATTENTION : blocs existants sans grid_config.json ; on suppose qu'ils viennent de cette configuration.")
        with open(path, "w") as f:
            json.dump(config, f, indent=1)


def list_chunks(out_dir):
    """Identifiants triés des blocs terminés."""
    ids = [int(f[6:12]) for f in os.listdir(out_dir)
           if f.startswith("chunk_") and f.endswith(".npz") and ".tmp" not in f]
    return sorted(ids)


def _param(z, key):
    """Colonne p_<key> d'un bloc ; K, j absents (modération pas dans l'espace) : valeur par défaut (K=0, j=2)."""
    if f"p_{key}" not in z.files and key in MOD_DEFAULTS:
        return np.full(len(z["dist_len"]), MOD_DEFAULTS[key])
    return z[f"p_{key}"]


def _metric(z, key):
    """Colonne m_<key> d'un bloc ; NaN pour une métrique récente absente d'un bloc ancien, 0 pour une métrique de
    modération absente (bloc sans modération)."""
    if f"m_{key}" not in z.files:
        if key in NEW_METRIC_KEYS:
            return np.full(len(z["dist_len"]), np.nan)
        if key in MOD_METRIC_KEYS:
            return np.zeros(len(z["dist_len"]))
    return z[f"m_{key}"]


def load_results(out_dir):
    """Paramètres et métriques de tous les blocs terminés, dans l'ordre des lignes.
    Retour : dict {nom: np.ndarray} (clés de PARAM_COLUMNS puis de METRIC_KEYS)."""
    columns = {key: [] for key in PARAM_COLUMNS + METRIC_KEYS}
    for cid in list_chunks(out_dir):
        with np.load(chunk_path(out_dir, cid)) as z:
            for key in PARAM_COLUMNS:
                columns[key].append(_param(z, key))
            for key in METRIC_KEYS:
                columns[key].append(_metric(z, key))
    return {key: np.concatenate(v) if v else np.array([]) for key, v in columns.items()}


def load_counts(out_dir, run_index, chunk_size=None):
    """Vecteur de comptes final (trié décroissant) du run `run_index` (numéro de ligne global).
    chunk_size : taille de bloc du lancement ; si None, lue dans grid_config.json."""
    if chunk_size is None:
        with open(os.path.join(out_dir, "grid_config.json")) as f:
            chunk_size = json.load(f)["chunk_size"]
    cid, i = divmod(run_index, chunk_size)
    with np.load(chunk_path(out_dir, cid)) as z:
        lens = z["dist_len"].astype(np.int64)
        offset = lens[:i].sum()
        values = z["dist_abund"][offset:offset + lens[i]]
        mult = z["dist_mult"][offset:offset + lens[i]]
    return np.repeat(values, mult)[::-1]


def _offset(lengths, i):
    lengths = np.asarray(lengths, dtype=np.int64)
    return int(lengths[:i].sum()), int(lengths[i])


def load_run(out_dir, run_index, chunk_size=None):
    """Données brutes du run `run_index` (numéro de ligne global).
    Retour : dict avec les paramètres (Ni, Nf, C, T, lam, mu, seed), les métriques stockées, Q (vecteur complet
    comme dans simulation()), ntags (tags par individu, ordre d'arrivée), flux (tous les tags dans cet ordre),
    d ({individu: array des classes}, comme simulation()["d"]), C_par_gen et C_final."""
    if chunk_size is None:
        with open(os.path.join(out_dir, "grid_config.json")) as f:
            chunk_size = json.load(f)["chunk_size"]
    cid, i = divmod(run_index, chunk_size)
    with np.load(chunk_path(out_dir, cid)) as z:
        if "raw_tags" not in z.files:
            raise ValueError(f"{out_dir} : grille sans données brutes (--sans_brut) ; rejouer le run avec simulation()")
        run = {key: _param(z, key)[i].item() for key in PARAM_COLUMNS}
        run.update({key: _metric(z, key)[i].item() for key in METRIC_KEYS})
        o, n = _offset(z["p_Nf"], i)
        ntags = z["raw_ntags"][o:o + n].astype(np.int64)
        o, n = _offset(z["m_n_tags"], i)
        flux = z["raw_tags"][o:o + n].astype(np.int64)
        o, n = _offset(z["p_T"], i)
        C_par_gen = z["raw_C_par_gen"][o:o + n].astype(np.int64)
    q = np.array([run[f"q_{j}"] for j in range(N_Q)])
    run["Q"] = np.append(q, [q[-1]] * 10)
    run["ntags"], run["flux"], run["C_par_gen"] = ntags, flux, C_par_gen
    run["C_final"] = int(C_par_gen[-1])
    run["d"] = dict(enumerate(np.split(flux, np.cumsum(ntags)[:-1])))
    return run


def historique(run):
    """f à chaque génération (T, C final), recalculé depuis les données brutes d'un run (load_run) :
    ligne t = fréquences des classes parmi les tags des individus présents au début de la génération t.
    Identique à simulation()["historique"] sans modération (K=0). Avec K > 0, les données brutes sont celles de
    la population finale, modérée : ce n'est plus le f vu par chaque génération."""
    tailles = np.round(np.linspace(run["Ni"], run["Nf"], run["T"] + 1)).astype(int)
    fin = np.concatenate([[0], np.cumsum(run["ntags"])])[tailles[:-1]]   # nb de tags avant chaque génération
    H = np.zeros((run["T"], run["C_final"]))
    for t, n in enumerate(fin):
        H[t] = np.bincount(run["flux"][:n], minlength=run["C_final"]) / n
    return H


def merge_csv(out_dir, output):
    """CSV (paramètres + métriques, une ligne par run). Paramètres flottants à 17 chiffres (exacts) :
    relire avec pandas.read_csv(..., float_precision="round_trip") pour pouvoir rejouer un run."""
    header = PARAM_COLUMNS + METRIC_KEYS
    int_cols = INT_PARAMS | MOD_METRIC_KEYS | {"n_tags", "C_final", "n_classes"}
    fmt = ["%d" if key in int_cols else ("%.17g" if key in PARAM_COLUMNS else "%.10g") for key in header]
    with open(output, "w") as f:
        f.write(",".join(header) + "\n")
        for cid in list_chunks(out_dir):
            with np.load(chunk_path(out_dir, cid)) as z:
                block = np.column_stack([_param(z, key) if key in PARAM_COLUMNS else _metric(z, key) for key in header])
            np.savetxt(f, block, fmt=fmt, delimiter=",")
    return output


def main():
    parser = argparse.ArgumentParser(description="Grille de simulations par blocs (reprenable), avec distributions finales.")
    parser.add_argument("--n_sims", type=int, required=True, help="nombre total de simulations")
    parser.add_argument("--out_dir", required=True, help="dossier de sortie (créé si besoin, reprise si existant)")
    parser.add_argument("--n_jobs", type=int, default=16, help="nb de processus (défaut 16 ; -1 = tous les cœurs)")
    parser.add_argument("--chunk_size", type=int, default=500, help="simulations par bloc/fichier (défaut 500)")
    parser.add_argument("--rng_seed", type=int, default=0, help="graine du tirage des paramètres")
    parser.add_argument("--seed_start", type=int, default=1, help="graine de simulation de la ligne 0")
    parser.add_argument("--range", nargs=3, action="append", default=[], metavar=("NOM", "MIN", "MAX"),
                        help="bornes d'un paramètre tiré au hasard (répétable), ex. --range mu 0 0.02 ; "
                             "modération : --range K 0 20 --range j 2 10")
    parser.add_argument("--fix", nargs=2, action="append", default=[], metavar=("NOM", "VALEUR"),
                        help="fixe un paramètre (répétable), ex. --fix Nf 30901")
    parser.add_argument("--k", type=int, default=10, help="taille du top-k pour le turnover")
    parser.add_argument("--max_lag", type=int, default=100, help="lag maximal de l'autocorrélation")
    parser.add_argument("--q_max", type=float, default=4.0, help="borne haute du tirage de Q ~ U(0, q_max) (défaut 4)")
    parser.add_argument("--sans_brut", action="store_true",
                        help="n'écrit pas les données brutes (raw_*) : distribution finale et métriques seulement")
    parser.add_argument("--merge", action="store_true", help="à la fin, écrit <out_dir>/results.csv")
    args = parser.parse_args()

    n_jobs = (os.cpu_count() or 1) if args.n_jobs == -1 else args.n_jobs
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        space = build_space(args.range, args.fix)
    except ValueError as e:
        parser.error(str(e))
    check_config(args.out_dir, {"rng_seed": args.rng_seed, "seed_start": args.seed_start, "chunk_size": args.chunk_size,
                                "k": args.k, "max_lag": args.max_lag, "numpy": np.__version__, **space,
                                # clé absente des grilles complètes, pour pouvoir reprendre les anciennes
                                **({"brut": False} if args.sans_brut else {}),
                                **({"q_max": args.q_max} if args.q_max != 2 else {})})
    print("espace de tirage :", json.dumps(space), f"; Q ~ U(0, {args.q_max})", flush=True)

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
                                   args.rng_seed, args.seed_start, space, args.k, args.max_lag,
                                   not args.sans_brut, args.q_max) for c in todo]
            try:
                for i, fut in enumerate(as_completed(futures), 1):
                    _, n, _ = fut.result()
                    done_sims += n
                    elapsed = time.perf_counter() - t0
                    eta = elapsed / done_sims * (total_todo - done_sims)
                    print(f"[{i}/{len(todo)}] {done_sims}/{total_todo} sims, {elapsed / 60:.1f} min écoulées, "
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
