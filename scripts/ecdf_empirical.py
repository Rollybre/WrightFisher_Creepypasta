# %% [markdown]
# ECDF des dates des données empiriques (fandom_data.csv) et comparaison avec la croissance
# d'archive de la simulation.
#
# Chaque ligne du CSV est une HISTOIRE (datée par `Date_Num`, jours depuis le 1er dépôt) portant une liste de
# catégories. Dans le modèle, un individu est une OCCURRENCE de catégorie (n_f = 30 901) : on « explose » donc
# le df en une ligne par (histoire, catégorie), chaque occurrence héritant de la date de son histoire.
# L'ECDF F(d) = part des individus déjà présents au jour d, donc la taille d'archive empirique
# normalisée : n(d) = n_i + (n_f - n_i) * F(d), à comparer avec la croissance linéaire de Simulation.
# Attention à l'explosion : `df.explode("Category")` sur la colonne brute ne fait RIEN (c'est une chaîne qui
# représente une liste) ; il faut parser, exploser, puis éclater les catégories combinées « A|B ».
#
# Fichier découpé en cellules `# %%` : exécutables une par une (VS Code, Spyder, PyCharm, Jupytext)
# ou en entier : `python scripts/ecdf_empirical.py`.

# %% Imports
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Paramètres (à modifier ici)
T = 100          # nombre de générations
n_i = 1000       # taille initiale de l'archive
n_f = 10000      # taille finale de l'archive (30 901 = nombre d'occurrences empiriques)
UNIT = "occurrences"   # unité d'un individu : "occurrences" (une ligne par catégorie, après explosion) ou "stories" (une ligne par histoire)

# Racine du dépôt : fonctionne en script (__file__) comme en cellule interactive (cwd)
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()
    while ROOT != ROOT.parent and not (ROOT / "data" / "fandom_data.csv").exists():
        ROOT = ROOT.parent
DATA = ROOT / "data" / "fandom_data.csv"
OUT = ROOT / "results" / "ecdf_empirical.png"   # mettre None pour ne pas sauvegarder

# %% Fonctions
def ecdf(x):
    """Retourne (x triés, F(x)) avec F = rang / n."""
    x = np.sort(x)
    return x, np.arange(1, len(x) + 1) / len(x)


def empirical_sizes(dates, T, n_i, n_f):
    """
    Tailles d'archive après chaque génération t = 0..T, en découpant l'axe du temps en T intervalles
    égaux : n(t) = n_i + (n_f - n_i) * F(date_min + t/T * (date_max - date_min)).
    Retourne un array de longueur T+1 (n(0) = n_i, n(T) = n_f). np.diff(...) donne les n_t.
    """
    grid = np.linspace(dates.min(), dates.max(), T + 1)
    F = np.searchsorted(dates, grid, side="right") / len(dates)
    F = (F - F[0]) / (F[-1] - F[0])  # force n(0) = n_i, n(T) = n_f
    return np.rint(n_i + (n_f - n_i) * F).astype(np.int64)


def explode_categories(df):
    """
    Une ligne par (histoire, catégorie) : le df « explose » de 13 128 histoires à 30 901 occurrences.
    - `Category` est une chaîne qui représente une liste Python : `df.explode("Category")` directement ne ferait
      RIEN, il faut d'abord la parser (ast.literal_eval) ;
    - les catégories combinées « A|B » sont éclatées en deux catégories ;
    - les histoires sans catégorie (liste vide) disparaissent ;
    - `story_id` (index d'origine) permet de retrouver l'histoire (nunique), et sa date est répétée sur chaque ligne.
    """
    out = df.assign(story_id=df.index, Category=df["Category"].apply(ast.literal_eval)).explode("Category")
    out = out.dropna(subset=["Category"])
    out = out.assign(Category=out["Category"].str.split("|")).explode("Category")
    out["Category"] = out["Category"].str.strip()
    return out[out["Category"] != ""].reset_index(drop=True)


FREQS = {"jour": "D", "semaine": "W", "mois": "MS"}     # granularités comparées (pandas : jour, semaine, début de mois)


def additions_per_period(dates, freq):
    """
    Nombre d'individus ajoutés par période (jour "D", semaine "W", mois "MS"), périodes vides comprises (= 0).
    Si une génération du modèle correspond à une période, c'est la série empirique des n_t.
    """
    return pd.Series(1, index=pd.to_datetime(dates)).sort_index().resample(freq).sum()

# %% Chargement des données
df = pd.read_csv(DATA)
occ = explode_categories(df)                       # une ligne par (histoire, catégorie)
print(f"{len(df)} histoires -> {len(occ)} occurrences de catégories ({len(occ) / len(df):.2f} par histoire), "
      f"{occ['Category'].nunique()} catégories, {occ['story_id'].nunique()} histoires avec au moins une catégorie")
assert occ["Date_Num"].notna().all()

dates_stories = np.sort(df["Date_Num"].to_numpy())
dates_occ = np.sort(occ["Date_Num"].to_numpy())
dates = dates_occ if UNIT == "occurrences" else dates_stories      # dates des individus selon l'unité choisie
print(f"unité : {UNIT} ({len(dates)} individus) ; Date_Num de {dates.min()} à {dates.max()}")
df.head()

# %% Explosion du df : histoires contre occurrences
grid = np.linspace(dates_stories.min(), dates_stories.max(), 2001)
F_st = np.searchsorted(dates_stories, grid, side="right") / len(dates_stories)
F_occ = np.searchsorted(dates_occ, grid, side="right") / len(dates_occ)
ks = np.abs(F_st - F_occ).max()
print(f"écart max entre les deux ECDF : {ks:.3f}  (jusqu'à {ks * len(dates_occ):.0f} individus pour n_f = {len(dates_occ)})")

tags = occ.groupby("story_id").size()
print("catégories par histoire : moyenne %.2f, médiane %d, max %d" % (tags.mean(), tags.median(), tags.max()))
print("occurrences datées du même jour : jusqu'à", pd.Series(dates_occ).value_counts().max())

year = pd.to_datetime(df["date"]).dt.year
per_year = (occ.assign(year=pd.to_datetime(occ["date"]).dt.year).groupby("year").size() / year.value_counts().sort_index())
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].step(grid, F_st, where="post", label=f"histoires (n={len(dates_stories)})")
ax[0].step(grid, F_occ, where="post", label=f"occurrences (n={len(dates_occ)})")
ax[0].set(xlabel="Date_Num (jours)", ylabel="ECDF", title="L'unité change la croissance"); ax[0].legend()
ax[1].bar(per_year.index, per_year.values)
ax[1].set(xlabel="année", ylabel="catégories par histoire", title="Catégories par histoire au cours du temps")
fig.tight_layout(); plt.show()

# %% Granularité temporelle : jour / semaine / mois
dates_cal = pd.to_datetime(occ["date"] if UNIT == "occurrences" else df["date"])      # dates calendaires des individus
series = {name: additions_per_period(dates_cal, freq) for name, freq in FREQS.items()}

rows = []
for name, a in series.items():
    T_impl = len(a)
    rows.append({"granularité": name, "T implicite": T_impl, "dans [300, 1000]": 300 <= T_impl <= 1000,
                 "ajouts moyens / période": a.mean(), "médiane": a.median(), "max": a.max(),
                 "% périodes sans ajout": 100 * (a == 0).mean(), "CV (écart-type / moyenne)": a.std() / a.mean()})
stats = pd.DataFrame(rows).set_index("granularité")
print(f"{len(dates_cal)} individus ({UNIT}) ; modèle linéaire : n_t = n_total / T, constant (CV = 0, aucune période vide)")
print(stats.round(2).to_string())

# Ajouts par période, avec le niveau constant du modèle linéaire
fig, axes = plt.subplots(3, 1, figsize=(11, 8))
for ax, (name, a) in zip(axes, series.items()):
    ax.step(a.index, a.values, where="post", lw=0.8)
    ax.axhline(a.mean(), color="k", ls="--", label=f"modèle linéaire : {a.mean():.1f} / {name}")
    ax.set(ylabel=f"ajouts / {name}", title=f"Par {name} : T = {len(a)} générations, CV = {a.std() / a.mean():.2f}")
    ax.legend(loc="upper left")
fig.tight_layout(); plt.show()

# Écart de l'archive empirique à la croissance linéaire, et irrégularité (CV) selon la granularité
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, a in series.items():
    T_impl = len(a)
    t = np.arange(1, T_impl + 1) / T_impl
    ax[0].plot(t, a.cumsum().to_numpy() - a.sum() * t, label=name)
ax[0].axhline(0, color="k", lw=0.8, ls="--")
ax[0].set(xlabel="temps normalisé", ylabel="archive empirique - archive linéaire (individus)", title="Écart à la croissance linéaire")
ax[0].legend()
ax[1].bar(stats.index, stats["CV (écart-type / moyenne)"])
ax[1].set(ylabel="CV des ajouts par période", title="Irrégularité des ajouts (0 = linéaire)")
fig.tight_layout(); plt.show()

# %% ECDF empirique
x, F = ecdf(dates)
plt.figure(figsize=(6, 4))
plt.step(x, F, where="post")
plt.xlabel("Date_Num (jours)"); plt.ylabel("ECDF"); plt.title(f"ECDF empirique (n={len(x)})")
plt.show()

# %% Tailles d'archive : empirique vs linéaire
n_emp = empirical_sizes(dates, T, n_i, n_f)
n_lin = np.rint(np.linspace(n_i, n_f, T + 1)).astype(np.int64)
n_t = np.diff(n_emp)   # individus ajoutés à chaque génération (à injecter dans la simulation)
print(n_emp[:10], "...", n_emp[-3:])

# %% Figure finale
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].step(x, F, where="post")
ax[0].set(xlabel="Date_Num (jours)", ylabel="ECDF", title=f"ECDF empirique (n={len(x)})")
ax[1].plot(n_lin, label="simulation actuelle (linéaire)", ls="--")
ax[1].plot(n_emp, label="ECDF empirique rééchantillonnée")
ax[1].set(xlabel="génération t", ylabel="taille de l'archive", title=f"T={T}")
ax[1].legend()
fig.tight_layout()
if OUT is not None:
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("->", OUT)
plt.show()
