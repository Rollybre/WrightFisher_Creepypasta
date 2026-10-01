# %% [markdown]
# ECDF des dates des données empiriques (fandom_data.csv) et comparaison avec la croissance
# d'archive de la simulation.
#
# Chaque ligne du CSV = un individu (une oeuvre) daté par `Date_Num` (jours depuis le 1er dépôt).
# L'ECDF F(d) = part des individus déjà présents au jour d, donc la taille d'archive empirique
# normalisée : n(d) = n_i + (n_f - n_i) * F(d), à comparer avec la croissance linéaire de Simulation.
#
# Fichier découpé en cellules `# %%` : exécutables une par une (VS Code, Spyder, PyCharm, Jupytext)
# ou en entier : `python scripts/ecdf_empirical.py`.

# %% Imports
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Paramètres (à modifier ici)
T = 100          # nombre de générations
n_i = 1000       # taille initiale de l'archive
n_f = 10000      # taille finale de l'archive

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

# %% Chargement des données
df = pd.read_csv(DATA)
dates = np.sort(df["Date_Num"].to_numpy())
print(len(dates), "individus ; Date_Num de", dates.min(), "à", dates.max())
df.head()

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
