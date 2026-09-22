"""
Ewens-Watterson homozygosity test on the empirical thematic category distribution.

Null model: the Ewens Sampling Formula (infinite-alleles model, neutral drift +
mutation/innovation at rate theta). This is the standard population-genetics test
of whether an observed allele/category frequency spectrum is compatible with
neutrality, given only the sample size n (total tag occurrences) and the number
of distinct categories K.

Steps:
  1. Estimate theta_hat (Watterson's estimator) from observed K and n via:
       E[K | theta, n] = sum_{i=0}^{n-1} theta / (theta + i)  =  K_obs
  2. Compute observed homozygosity F_obs = sum_i (n_i/n)^2  (= Gini-Simpson complement).
  3. Simulate the null distribution of F under ESF(theta_hat, n) via the Chinese
     Restaurant Process (exact sampler from the Ewens Sampling Formula), vectorised
     across N_REPLICATES replicates.
  4. Two-sided Monte Carlo p-value comparing F_obs to the null distribution.

Caveat: the ESF assumes independent draws. The 30,901 tag occurrences analysed here
come from 13,128 multi-tagged stories, so tags within a story are not independent
draws (same caveat as noted for the power-law fit in analysis_stats.py).
"""

import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

RNG_SEED = 42
N_REPLICATES = 2000

ROOT = Path(__file__).parent
DATA_PATH = ROOT.parent / "data" / "fandom_data.csv"

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD EMPIRICAL CATEGORY COUNTS
# ═══════════════════════════════════════════════════════════════════════════

df = pd.read_csv(DATA_PATH)
df["Category_list"] = df["Category"].apply(ast.literal_eval)


def clean_cats(cats):
    return [p.strip() for c in cats for p in c.split("|") if p.strip()]


df["Category_clean"] = df["Category_list"].apply(clean_cats)
all_cats = [cat for cats in df["Category_clean"] for cat in cats]
counter = Counter(all_cats)

counts = np.array(sorted(counter.values(), reverse=True), dtype=float)
n_obs = int(counts.sum())
k_obs = len(counts)

print(f"n (total tag occurrences) = {n_obs}")
print(f"K (distinct categories)   = {k_obs}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. OBSERVED HOMOZYGOSITY
# ═══════════════════════════════════════════════════════════════════════════

p_obs = counts / n_obs
F_obs = np.sum(p_obs ** 2)
print(f"F_obs (homozygosity, sum p_i^2) = {F_obs:.5f}")
print(f"Gini-Simpson diversity (1-F_obs) = {1 - F_obs:.5f}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. WATTERSON ESTIMATOR OF THETA FROM (K_obs, n_obs)
# ═══════════════════════════════════════════════════════════════════════════


def expected_k(theta, n):
    i = np.arange(n)
    return np.sum(theta / (theta + i))


def root_fn(theta):
    return expected_k(theta, n_obs) - k_obs


theta_hat = brentq(root_fn, 1e-6, 10_000)
print(f"\ntheta_hat (Watterson estimator) = {theta_hat:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. MONTE CARLO SIMULATION FROM THE EWENS SAMPLING FORMULA
# ═══════════════════════════════════════════════════════════════════════════
# Exact CRP sampler, vectorised across N_REPLICATES replicates.
# Customer i (0-indexed, i = 1..n-1 already-placed customers before this draw):
#   - starts a new table w.p. theta / (theta + i)
#   - otherwise joins the table of a uniformly random previous customer j < i
#     (equivalent to "proportional to current table size")


def simulate_F_null(theta, n, n_reps, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    labels = np.zeros((n_reps, n), dtype=np.int32)
    next_label = np.ones(n_reps, dtype=np.int32)  # label 0 used by customer 0

    for i in range(1, n):
        p_new = theta / (theta + i)
        is_new = rng.random(n_reps) < p_new
        join_target = rng.integers(0, i, size=n_reps)
        joined_label = labels[np.arange(n_reps), join_target]
        labels[:, i] = np.where(is_new, next_label, joined_label)
        next_label += is_new

    F_sim = np.empty(n_reps)
    k_sim = np.empty(n_reps, dtype=int)
    for r in range(n_reps):
        counts_r = np.bincount(labels[r])
        p_r = counts_r / n
        F_sim[r] = np.sum(p_r ** 2)
        k_sim[r] = len(counts_r)
    return F_sim, k_sim


print(f"\nSimulating {N_REPLICATES} samples from ESF(theta={theta_hat:.3f}, n={n_obs})...")
F_sim, k_sim = simulate_F_null(theta_hat, n_obs, N_REPLICATES)

print(f"  Null F: mean={F_sim.mean():.5f}, sd={F_sim.std():.5f}, "
      f"[5%,95%]=[{np.percentile(F_sim, 5):.5f}, {np.percentile(F_sim, 95):.5f}]")
print(f"  Null K: mean={k_sim.mean():.1f} (sanity check, should be close to {k_obs})")

# ═══════════════════════════════════════════════════════════════════════════
# 5. P-VALUE
# ═══════════════════════════════════════════════════════════════════════════

p_low = np.mean(F_sim <= F_obs)   # P(null at least as even/diverse as observed)
p_high = np.mean(F_sim >= F_obs)  # P(null at least as concentrated as observed)
p_two_sided = min(2 * min(p_low, p_high), 1.0)

print(f"\nF_obs = {F_obs:.5f} vs null distribution")
print(f"  P(F_null <= F_obs) = {p_low:.4f}")
print(f"  P(F_null >= F_obs) = {p_high:.4f}")
print(f"  two-sided Monte Carlo p-value = {p_two_sided:.4f}")

if F_obs > np.percentile(F_sim, 97.5):
    verdict = "MORE concentrated (higher homozygosity) than neutral ESF expectation"
elif F_obs < np.percentile(F_sim, 2.5):
    verdict = "MORE even/diverse (lower homozygosity) than neutral ESF expectation"
else:
    verdict = "compatible with neutral ESF expectation (within 95% band)"
print(f"\nVerdict: observed distribution is {verdict}")
