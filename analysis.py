"""
Exploration des résultats des grilles de simulation.py / run_grid.py : un pair plot global, puis
un examen détaillé d'un indice à la fois, avec les valeurs empiriques en repère.

Usage (notebook ou script) :
    from analysis import Explorer

    ex = Explorer("results/grid_n30901/chunks")   # dossier de blocs .npz, ou un DataFrame, ou un CSV de résultats
    ex.pairplot()                                  # vue globale : paramètres et indices, empirique en rouge
    ex.pairplot(mode="cross")                      # seulement paramètres (colonnes) x indices (lignes)
    ex.index("chao1")                              # détail d'un indice : distribution + dépendance à chaque paramètre

    near = ex.closest(10_000)                      # les 10 000 runs les plus proches de l'empirique (ABC)
    near.pairplot()                                # zone de paramètres compatible avec l'empirique (points colorés par la distance)
    ex.filter(q=(0.5, 0.9), T=(500, None)).index("gini")   # sous-ensemble par bornes (None = illimité)

Toutes les méthodes de tracé renvoient la figure matplotlib (fig.savefig(...)) ; `index` renvoie aussi
un tableau récapitulatif. Les résultats sont sous-échantillonnés pour les tracés (les 5 millions de
points ne se lisent pas), mais les statistiques (distance, percentiles) portent sur tous les runs.
"""

import os

import numpy as np
import pandas as pd

PARAMS = ["q", "mu", "alpha", "C", "n_i", "T"]
INDICES = ["gini", "gini_obs", "hill_0", "hill_1", "hill_2", "hill_inf", "chao1", "kl_emp"]
LOG_VARS = {"hill_0", "hill_1", "hill_2", "hill_inf", "chao1", "n_classes", "kl_emp"}   # affichés en log10

# Indices empiriques (data/fandom_data.csv : 176 catégories, 30 901 occurrences), calculés avec les
# fonctions de simulation.py ; mêmes valeurs que la section « Indices empiriques » du notebook.
EMPIRICAL = {
    "gini": 0.7963273842270476,
    "gini_obs": 0.7963273842270476,   # identique : toutes les catégories empiriques sont observées
    "hill_0": 176.0,
    "hill_1": 45.506516340063556,
    "hill_2": 26.53617935151933,
    "hill_inf": 9.784990500316656,
    "chao1": 200.0,
    "n_classes": 176,
    "kl_emp": 0.0,   # divergence à l'empirique : 0 par définition (non tracée en log10)
}


def _load(source):
    """DataFrame depuis un DataFrame, un CSV (relu exactement, cf. merge_csv) ou un dossier de blocs .npz."""
    if isinstance(source, pd.DataFrame):
        return source
    if os.path.isdir(source):
        from run_grid import load_results
        return pd.DataFrame(load_results(source))
    # float_precision="round_trip" : relecture exacte des paramètres flottants (rejeu des runs possible)
    return pd.read_csv(source, float_precision="round_trip")


def _axis_name(var):
    """Nom affiché d'une variable (préfixe log10 pour les variables tracées en log)."""
    return f"log10 {var}" if var in LOG_VARS else var


def _bins(x, n_target=40):
    """Bins d'histogramme : alignés sur les entiers (largeur entière) pour une variable entière, sinon n_target bins égaux."""
    x = np.asarray(x, dtype=float)
    if np.all(x == np.round(x)):
        width = max(1, round((x.max() - x.min() + 1) / n_target))
        return np.arange(x.min() - 0.5, x.max() + 0.5 + width, width)
    return n_target


# ============================================================================
# AJUSTEMENT MLE DE LOI DE PUISSANCE (rang-fréquence)
# ============================================================================
def _mle_alpha(x, xmin):
    """Exposant MLE d'une loi de puissance discrète tronquée à xmin (approximation continue, éq. 3.7 de Clauset et al. 2009)."""
    return 1 + len(x) / np.sum(np.log(x / (xmin - 0.5)))


def _ks_distance(x, alpha, xmin):
    """Distance KS entre la CDF complémentaire empirique et celle du modèle (zeta de Hurwitz normalisée à xmin)."""
    from scipy.special import zeta as hurwitz_zeta
    xs = np.sort(x)
    n = len(xs)
    return np.max(np.abs(np.arange(n, 0, -1) / n - hurwitz_zeta(alpha, xs) / hurwitz_zeta(alpha, xmin)))


def fit_power_law_mle(freq, xmin_min=1, min_tail_size=5):
    """
    Ajuste une loi de puissance discrète sur un vecteur de comptes par classe (valeurs > 0), avec xmin
    choisi par minimisation de la distance KS (Clauset, Shalizi & Newman 2009). Même estimateur que
    code/plot_archive_rate_mle.py.

    Paramètres :
    - freq          : comptes par classe (array-like ; les zéros sont ignorés)
    - xmin_min      : plus petite valeur de xmin candidate
    - min_tail_size : nombre minimal de points au-dessus de xmin pour qu'un xmin soit candidat

    Retour : dict {alpha, sigma, xmin, n_tail, ks}, ou None si aucun xmin n'est valide.
    """
    freq = np.asarray(freq, dtype=float)
    freq = freq[freq > 0]
    best = None
    for xmin in np.unique(freq[freq >= xmin_min]):
        tail = freq[freq >= xmin]
        if len(tail) < min_tail_size:
            continue
        alpha = _mle_alpha(tail, xmin)
        d = _ks_distance(tail, alpha, xmin)
        if best is None or d < best[0]:
            best = (d, alpha, xmin, len(tail))
    if best is None:
        return None
    d, alpha, xmin, n_tail = best
    return {"alpha": alpha, "sigma": (alpha - 1) / np.sqrt(n_tail), "xmin": xmin, "n_tail": n_tail, "ks": d}


def draw_fit(ax, counts, fit, scale=1, **kw):
    """
    Trace sur un graphe rang-fréquence log-log la droite MLE f(rang) ~ rang^-(alpha-1) de la queue
    (valeurs >= xmin) d'un vecteur de comptes trié par ordre décroissant, ancrée sur la première valeur
    de la queue. `scale` divise les comptes (ex. taille d'échantillon pour des fréquences relatives).
    """
    counts = counts[counts > 0]
    ranks = np.arange(1, len(counts) + 1)[counts >= fit["xmin"]]
    ax.loglog(ranks, counts[counts >= fit["xmin"]][0] * (ranks / ranks[0]) ** -(fit["alpha"] - 1) / scale, **kw)


class Explorer:
    """
    Résultats d'une grille de simulations + outils d'exploration graphique.

    Paramètres :
    - source    : DataFrame, chemin d'un CSV de résultats, ou dossier d'une grille (blocs .npz).
                  Colonnes attendues : PARAMS (q, mu, alpha, C, n_i, T) et INDICES (gini, hill_0,
                  hill_1, hill_2, hill_inf, chao1) ; les autres colonnes sont conservées.
    - empirical : dict {indice: valeur empirique} tracé en rouge (défaut : EMPIRICAL)
    """

    def __init__(self, source, empirical=None):
        self.df = _load(source)
        self.empirical = dict(EMPIRICAL if empirical is None else empirical)
        missing = [c for c in PARAMS + INDICES if c not in self.df.columns]
        if missing:
            raise ValueError(f"colonnes manquantes : {missing}")

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return f"Explorer({len(self):,} runs, colonnes : {', '.join(self.df.columns)})"

    def _sub(self, df):
        return Explorer(df, self.empirical)

    # ------------------------------------------------------------------ sélection
    def filter(self, **bounds):
        """
        Sous-ensemble par bornes : ex. filter(q=(0.5, 0.9), T=(500, None), gini=(None, 0.85)).
        Chaque valeur est un couple (min, max) inclusif ; None = pas de borne de ce côté.
        Retour : nouvel Explorer.
        """
        mask = np.ones(len(self.df), dtype=bool)
        for col, (lo, hi) in bounds.items():
            if col not in self.df.columns:
                raise KeyError(col)
            if lo is not None:
                mask &= (self.df[col] >= lo).to_numpy()
            if hi is not None:
                mask &= (self.df[col] <= hi).to_numpy()
        return self._sub(self.df[mask])

    def sample(self, n, seed=0):
        """Échantillon aléatoire de n runs (sans remise) sous forme de DataFrame ; tout si n >= len."""
        if n is None or n >= len(self.df):
            return self.df
        return self.df.sample(n, random_state=seed)

    def distance(self, metrics=("gini", "hill_1", "hill_2", "hill_inf", "chao1"), log=("chao1",)):
        """
        Distance normalisée de chaque run à l'empirique : norme euclidienne des écarts en unités
        d'écart-type (z-score, moyenne et écart-type calculés sur tous les runs de cet Explorer).
        `log` : indices passés en log avant la normalisation (chao1 : queue très lourde).

        Retour : pd.Series alignée sur self.df.
        """
        cols = {}
        for m in metrics:
            x = self.df[m].to_numpy(dtype=float)
            e = float(self.empirical[m])
            if m in log:
                x, e = np.log(x), np.log(e)
            cols[m] = (x - x.mean()) / x.std(), (e - x.mean()) / x.std()
        d2 = sum((z - ze) ** 2 for z, ze in cols.values())
        return pd.Series(np.sqrt(d2), index=self.df.index, name="distance")

    def closest(self, n=5000, **distance_kwargs):
        """
        Les n runs les plus proches de l'empirique (cf. `distance`), avec une colonne "distance".
        Retour : nouvel Explorer.
        """
        d = self.distance(**distance_kwargs)
        idx = d.nsmallest(n).index
        out = self.df.loc[idx].copy()
        out["distance"] = d.loc[idx]
        return self._sub(out)

    def describe(self, cols=None):
        """Résumé statistique (transposé) des paramètres et indices, avec l'empirique et son percentile."""
        cols = cols or PARAMS + INDICES
        t = self.df[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
        t["empirique"] = [self.empirical.get(c, np.nan) for c in cols]
        t["percentile_emp"] = [100 * (self.df[c] < self.empirical[c]).mean() if c in self.empirical else np.nan
                               for c in cols]
        return t

    # ------------------------------------------------------------------ vue globale
    def pairplot(self, vars=None, mode="all", n=3000, color_by="auto", empirical=True, log=True,
                 cmap="icefire", seed=0):
        """
        Pair plot global, dans le style du notebook : grille carrée complète, densités (KDE remplie)
        sur la diagonale, nuages de points hors diagonale colorés par une variable (par défaut la
        distance à l'empirique quand elle existe, cf. `closest`), une seule barre de couleur à droite.
        Les valeurs empiriques sont marquées par des lignes rouges fines.

        Paramètres :
        - vars      : liste de colonnes à tracer (défaut : PARAMS + INDICES)
        - mode      : "all" = grille carrée complète ; "cross" = paramètres en colonnes x indices en
                      lignes (seulement les liens paramètre -> indice)
        - n         : nombre de runs tracés (échantillon aléatoire ; 3000 par défaut)
        - color_by  : colonne numérique pour colorer les points ; "auto" = "distance" si présente,
                      sinon une couleur unie ; None = couleur unie
        - empirical : trace les repères empiriques (lignes rouges)
        - log       : trace en log10 les variables de LOG_VARS (Hill, Chao1, n_classes, kl_emp)
        - cmap      : palette des points colorés (défaut "icefire", comme le notebook)
        - seed      : graine de l'échantillon

        Retour : la figure matplotlib.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if color_by == "auto":
            color_by = "distance" if "distance" in self.df.columns else None
        cols = list(vars) if vars else PARAMS + INDICES
        sub = self.sample(n, seed)
        use_log = lambda c: log and c in LOG_VARS
        name_of = lambda c: _axis_name(c) if use_log(c) else c
        data = pd.DataFrame({name_of(c): np.log10(sub[c]) if use_log(c) else sub[c] for c in cols})
        emp = {name_of(c): (np.log10(self.empirical[c]) if use_log(c) else self.empirical[c])
               for c in cols if c in self.empirical and (not use_log(c) or self.empirical[c] > 0)} if empirical else {}
        color = sub[color_by] if color_by else None
        names = list(data.columns)

        if mode == "cross":
            xs = [name_of(c) for c in cols if c in PARAMS]
            ys = [name_of(c) for c in cols if c not in PARAMS]
            g = sns.PairGrid(data, x_vars=xs, y_vars=ys, height=1.9)
        elif mode == "all":
            g = sns.PairGrid(data, vars=names, diag_sharey=False, height=1.7)
            g.map_diag(sns.kdeplot, fill=True, color="#4C72B0")
        else:
            raise ValueError("mode doit valoir 'all' ou 'cross'")

        def scatter(x, y, **kw):
            ax = plt.gca()
            if color is None:
                ax.scatter(x, y, s=15, alpha=0.25, edgecolor="none", color="#4C72B0")
            else:
                ax.scatter(x, y, c=color.loc[x.index], cmap=cmap, vmin=color.min(), vmax=color.max(),
                           s=15, alpha=0.25, edgecolor="none")

        if mode == "cross":
            g.map(scatter)
        else:
            g.map_offdiag(scatter)

        # repères empiriques
        for i, y in enumerate(g.y_vars):
            for j, x in enumerate(g.x_vars):
                ax = g.axes[i, j]
                if ax is None:
                    continue
                if x in emp:
                    ax.axvline(emp[x], color="red", lw=0.9, alpha=0.8)
                if y in emp and x != y:
                    ax.axhline(emp[y], color="red", lw=0.9, alpha=0.8)

        if color is not None:
            # une seule barre de couleur pour toute la grille, à droite (norme = étendue réelle de la variable)
            cbar_ax = g.figure.add_axes([1.02, 0.15, 0.02, 0.7])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(color.min(), color.max()))
            g.figure.colorbar(sm, cax=cbar_ax, label=color_by if color_by != "distance" else "distance à l'empirique")
        g.figure.suptitle(f"{len(sub):,} runs tirés parmi {len(self):,}" + (" (rouge = empirique)" if emp else ""), y=1.01)
        return g.figure

    # ------------------------------------------------------------------ détail d'un indice
    def index(self, name, n=500_000, n_bins=20, seed=0):
        """
        Examen détaillé d'un indice : sa distribution (avec l'empirique et son percentile) et, pour
        chaque paramètre, sa dépendance : médiane et bande 10-90 % de l'indice par tranche de paramètre.

        Paramètres :
        - name   : indice à examiner (gini, hill_0, hill_1, hill_2, hill_inf, chao1, n_classes, ...)
        - n      : nb de runs utilisés pour les courbes par tranche (les statistiques du tableau
                   utilisent tous les runs)
        - n_bins : nombre de tranches par paramètre
        - seed   : graine de l'échantillon

        Retour : (figure, tableau récapitulatif), le tableau donnant les quantiles de l'indice,
        l'empirique, son percentile, et la corrélation de Spearman de l'indice avec chaque paramètre.
        """
        import matplotlib.pyplot as plt

        if name not in self.df.columns:
            raise KeyError(name)
        log = name in LOG_VARS
        emp = self.empirical.get(name)
        if emp is not None and log and emp <= 0:
            emp = None                     # repère non représentable en log (ex. kl_emp = 0)
        x_all = self.df[name].to_numpy(dtype=float)
        sub = self.sample(n, seed)

        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        axes = axes.ravel()
        ax = axes[0]
        ax.hist(np.log10(x_all) if log else x_all, bins=80, color="#4C72B0", alpha=0.75)
        if emp is not None:
            ax.axvline(np.log10(emp) if log else emp, color="red", lw=2,
                       label=f"empirique = {emp:.4g} (percentile {100 * (x_all < emp).mean():.0f} %)")
            ax.legend(fontsize=8)
        ax.set_title(f"{name} — distribution ({len(x_all):,} runs)")
        ax.set_xlabel(_axis_name(name))

        rho = {}
        for ax, p in zip(axes[1:], PARAMS):
            px, py = sub[p].to_numpy(dtype=float), sub[name].to_numpy(dtype=float)
            edges = np.linspace(px.min(), px.max(), n_bins + 1)
            which = np.clip(np.digitize(px, edges) - 1, 0, n_bins - 1)
            mids, q10, q50, q90 = [], [], [], []
            for b in range(n_bins):
                v = py[which == b]
                if v.size:
                    mids.append((edges[b] + edges[b + 1]) / 2)
                    q10.append(np.percentile(v, 10)); q50.append(np.median(v)); q90.append(np.percentile(v, 90))
            ax.fill_between(mids, q10, q90, alpha=0.3, color="#4C72B0", label="10-90 %")
            ax.plot(mids, q50, color="#4C72B0", label="médiane")
            if emp is not None:
                ax.axhline(emp, color="red", lw=1.2, label="empirique")
            if log:
                ax.set_yscale("log")
            ax.set_xlabel(p)
            ax.set_ylabel(name)
            r = pd.Series(px).corr(pd.Series(py), method="spearman")
            rho[p] = r
            ax.set_title(f"{name} vs {p} (Spearman {r:+.2f})")
        axes[1].legend(fontsize=8)
        axes[7].axis("off")
        fig.tight_layout()

        s = pd.Series(x_all).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        summary = s.to_frame(name).T
        summary["empirique"] = emp if emp is not None else np.nan
        summary["percentile_emp"] = 100 * (x_all < emp).mean() if emp is not None else np.nan
        for p, r in rho.items():
            summary[f"spearman_{p}"] = r
        return fig, summary
