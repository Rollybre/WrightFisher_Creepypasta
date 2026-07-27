"""
Statistical analysis for Phase 4:
  - Power-law MLE fit on empirical and model distributions
  - KS goodness-of-fit test
  - Averaged Figure 6 (N_RUNS simulations)
  - Regenerated log-log figure with fitted curves

Outputs (saved to article/illustration/):
  - loglog_fit.png          : log-log empirical + model + fitted lines
  - loglog_model_avg.png    : averaged model distribution (N_RUNS runs)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter
from pathlib import Path
import ast
import pandas as pd
import powerlaw
from scipy.stats import ks_2samp
from tqdm import tqdm

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DATA_PATH  = ROOT / "fandom_data.csv"
OUTPUT_DIR = ROOT / "article" / "illustration"
OUTPUT_DIR.mkdir(exist_ok=True)

# Number of empirical categories to compare against the model (= n_classes)
N_TOP = 40

# ── Style ────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN EMPIRICAL DATA
# ═══════════════════════════════════════════════════════════════════════════

print("Loading empirical data...")
df = pd.read_csv(DATA_PATH)
df["Category_list"] = df["Category"].apply(ast.literal_eval)

def clean_cats(cats):
    return [p.strip() for c in cats for p in c.split("|") if p.strip()]

df["Category_clean"] = df["Category_list"].apply(clean_cats)
all_cats = [cat for cats in df["Category_clean"] for cat in cats]
counter  = Counter(all_cats)

# All empirical frequencies sorted descending
emp_freq_all = np.array(sorted(counter.values(), reverse=True), dtype=float)

# Top-N for model comparison (same resolution as model: n=40)
emp_freq = emp_freq_all[:N_TOP]
emp_ranks = np.arange(1, len(emp_freq) + 1)

coverage = emp_freq.sum() / emp_freq_all.sum() * 100
print(f"  {len(df)} stories, {len(emp_freq_all)} categories total")
print(f"  Top {N_TOP} categories: {int(emp_freq.sum())} occurrences ({coverage:.1f}% of total)")

# ═══════════════════════════════════════════════════════════════════════════
# 2. POWER-LAW FIT ON EMPIRICAL DATA
# ═══════════════════════════════════════════════════════════════════════════

# Fit on full empirical distribution (176 categories)
print(f"\nFitting power-law to full empirical distribution (176 categories)...")
fit_emp = powerlaw.Fit(emp_freq_all, discrete=True, verbose=False)
alpha_emp = fit_emp.alpha
sigma_emp = fit_emp.sigma
xmin_emp  = fit_emp.xmin
print(f"  α_emp = {alpha_emp:.3f} ± {sigma_emp:.3f}  (xmin = {xmin_emp:.0f})")

# ═══════════════════════════════════════════════════════════════════════════
# 3. WF SIMULATION (single canonical run, same params as paper)
# ═══════════════════════════════════════════════════════════════════════════

# ── n=40 canonical parameters (used in the paper's main figures) ──────────
N_CLASSES_PAPER = 40
N_INIT          = 1_000
N_FINAL         = 2_000
T_MAX           = 1_000
ARCHIVE_RATE    = 0.05

# ── n=176 validation parameters (matches empirical resolution) ────────────
# Scale n_i / n_f proportionally to keep the same per-class density
# (original: 1000 individuals / 40 classes = 25 per class on average)
N_CLASSES_VALID = 176
N_INIT_VALID    = int(N_INIT  * N_CLASSES_VALID / N_CLASSES_PAPER)   # 4400
N_FINAL_VALID   = int(N_FINAL * N_CLASSES_VALID / N_CLASSES_PAPER)   # 8800

RNG = np.random.default_rng(42)

def run_simulation(rng, n_classes, n_init, n_final, t_max, archive_rate,
                   init_probs=None):
    """Single WF run with cumulative archiving. Returns final archive."""
    if init_probs is not None:
        state = rng.choice(n_classes, size=n_init, p=init_probs)
    else:
        ranks_init = np.arange(1, n_classes + 1, dtype=float)
        probs = ranks_init ** -1.0
        probs /= probs.sum()
        state = rng.choice(n_classes, size=n_init, p=probs)

    pop_sizes = [
        int(n_init + t * (n_final - n_init) / t_max)
        for t in range(t_max)
    ]
    archive = []
    for t in range(t_max):
        state = rng.choice(state, replace=True, size=pop_sizes[t])
        n_arch = int(len(state) * archive_rate)
        if n_arch > 0:
            archive.extend(rng.choice(state, replace=False, size=n_arch))
    return np.array(archive)

# ── Canonical n=40 run (paper figures) ───────────────────────────────────
print("\nRunning canonical simulation (n=40, seed=42)...")
archive_canonical = run_simulation(
    RNG, N_CLASSES_PAPER, N_INIT, N_FINAL, T_MAX, ARCHIVE_RATE
)
counts_model = Counter(archive_canonical)
model_freq = np.array(sorted(counts_model.values(), reverse=True), dtype=float)
model_ranks = np.arange(1, len(model_freq) + 1)
print(f"  Archive size: {len(archive_canonical)}, {len(model_freq)} classes represented")

# ── Validation n=176 run (goodness-of-fit comparison) ────────────────────
# Initialize from actual empirical proportions
emp_probs_176 = emp_freq_all / emp_freq_all.sum()

print(f"\nRunning validation simulation (n=176, n_i={N_INIT_VALID}, seed=42)...")
rng_valid = np.random.default_rng(42)
archive_valid = run_simulation(
    rng_valid, N_CLASSES_VALID, N_INIT_VALID, N_FINAL_VALID, T_MAX, ARCHIVE_RATE,
    init_probs=emp_probs_176
)
counts_valid = Counter(archive_valid)
# Map back to sorted frequencies (same ordering as empirical)
valid_freq = np.array(
    [counts_valid.get(i, 0) for i in range(N_CLASSES_VALID)], dtype=float
)
# Sort descending for rank-frequency plot
valid_freq_sorted = np.sort(valid_freq)[::-1]
valid_ranks = np.arange(1, N_CLASSES_VALID + 1)
print(f"  Archive size: {len(archive_valid)}, "
      f"{(valid_freq > 0).sum()} classes represented")

# ═══════════════════════════════════════════════════════════════════════════
# 4. POWER-LAW FIT ON MODEL DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════

# Fit on canonical n=40 model (for figure annotation)
print("\nFitting power-law to canonical model distribution (n=40)...")
fit_model = powerlaw.Fit(model_freq, discrete=True, verbose=False)
alpha_model = fit_model.alpha
sigma_model = fit_model.sigma
print(f"  α_mod (n=40) = {alpha_model:.3f} ± {sigma_model:.3f}")

# Fit on validation n=176 model (for GoF comparison)
print("\nFitting power-law to validation model distribution (n=176)...")
valid_nonzero = valid_freq_sorted[valid_freq_sorted > 0]
fit_valid = powerlaw.Fit(valid_nonzero, discrete=True, verbose=False)
alpha_valid = fit_valid.alpha
sigma_valid = fit_valid.sigma
print(f"  α_mod (n=176) = {alpha_valid:.3f} ± {sigma_valid:.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. KS TEST (empirical vs model)
# ═══════════════════════════════════════════════════════════════════════════

print("\nKolmogorov-Smirnov test (n=176: empirical vs validation model)...")
# Both distributions have 176 data points — fair comparison
emp_prop_176   = emp_freq_all / emp_freq_all.sum()
valid_prop_176 = valid_freq / valid_freq.sum()   # includes zeros
ks_stat, ks_p = ks_2samp(emp_prop_176, valid_prop_176)
print(f"  KS statistic = {ks_stat:.4f},  p-value = {ks_p:.4f}")
print(f"  Sample sizes: n_emp={len(emp_prop_176)}, n_mod={len(valid_prop_176)}")

# ═══════════════════════════════════════════════════════════════════════════
# 6. FIGURE A — LOG-LOG WITH FITTED LINES
# ═══════════════════════════════════════════════════════════════════════════

print("\nGenerating loglog_fit.png...")

def zipf_fit_line(fit, ranks):
    """Generate fitted Zipf line from a powerlaw.Fit object over given ranks."""
    # p(x) ∝ x^{-α}  →  on log-log axes this is a straight line with slope -α
    # We scale to match the max observed value
    x0 = fit.xmin
    ref_val = fit.power_law.pdf(x0) * fit.n  # expected count at xmin
    return ref_val * (ranks / x0) ** (-fit.alpha)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

# ── Left: empirical (all 176) ────────────────────────────────────────────
ax = axes[0]
emp_ranks_all = np.arange(1, len(emp_freq_all) + 1)
ax.loglog(emp_ranks_all, emp_freq_all, "o", ms=4, alpha=0.8, color="#2171b5",
          label="Empirical data (176 categories)")
tail_mask = emp_freq_all >= xmin_emp
tail_ranks = emp_ranks_all[tail_mask]
fit_line_emp = (emp_freq_all[tail_mask][0]) * (tail_ranks / tail_ranks[0]) ** -(alpha_emp - 1)
ax.loglog(tail_ranks, fit_line_emp, "--", color="#d73027", lw=2,
          label=fr"Power-law fit ($\hat{{\alpha}}={alpha_emp:.2f}\pm{sigma_emp:.2f}$)")
ax.set_xlabel("Rank", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Empirical distribution (176 categories)", fontsize=13)
ax.legend(fontsize=10)

# ── Right: validation model (n=176) ──────────────────────────────────────
ax = axes[1]
valid_nonzero_mask = valid_freq_sorted > 0
valid_plot_ranks = np.arange(1, valid_nonzero_mask.sum() + 1)
valid_plot_freq  = valid_freq_sorted[valid_nonzero_mask]
ax.loglog(valid_plot_ranks, valid_plot_freq, "s", ms=5, alpha=0.7, color="#41ab5d",
          label=f"Model (n=176, single run)")
tail_mask_v = valid_plot_freq >= fit_valid.xmin
if tail_mask_v.any():
    tail_ranks_v = valid_plot_ranks[tail_mask_v]
    fit_line_v = (valid_plot_freq[tail_mask_v][0]) * (tail_ranks_v / tail_ranks_v[0]) ** -(alpha_valid - 1)
    ax.loglog(tail_ranks_v, fit_line_v, "--", color="#d73027", lw=2,
              label=fr"Power-law fit ($\hat{{\alpha}}={alpha_valid:.2f}\pm{sigma_valid:.2f}$)")
ax.set_xlabel("Rank", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Model distribution (n=176, cumulative archive)", fontsize=13)
ax.legend(fontsize=10)

fig.suptitle(
    f"Rank–frequency distributions with MLE power-law fits\n"
    f"KS test (n=176 empirical vs. model): $D$={ks_stat:.3f}, $p$={ks_p:.3f}",
    fontsize=12, y=1.02
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "loglog_fit.png", bbox_inches="tight")
plt.close()
print("  Saved loglog_fit.png")

# ═══════════════════════════════════════════════════════════════════════════
# 7. FIGURE B — AVERAGED LOG-LOG (N_RUNS runs)
# ═══════════════════════════════════════════════════════════════════════════

N_RUNS = 20
print(f"\nRunning {N_RUNS} simulations for averaged Figure 6 (n=176)...")
rng_avg = np.random.default_rng(0)

all_run_freqs = []
for run in tqdm(range(N_RUNS), desc="Runs"):
    arch = run_simulation(
        rng_avg, N_CLASSES_VALID, N_INIT_VALID, N_FINAL_VALID, T_MAX, ARCHIVE_RATE,
        init_probs=emp_probs_176
    )
    c = Counter(arch)
    # Keep all 176 classes (include zeros for unrepresented ones)
    freqs = np.array(sorted(
        [c.get(i, 0) for i in range(N_CLASSES_VALID)], reverse=True
    ), dtype=float)
    all_run_freqs.append(freqs)

# Pad to max length with zeros and compute mean / percentiles
max_len = max(len(f) for f in all_run_freqs)
padded  = np.array([
    np.pad(f, (0, max_len - len(f))) for f in all_run_freqs
], dtype=float)

mean_freq  = np.mean(padded, axis=0)
p05        = np.percentile(padded, 5,  axis=0)
p95        = np.percentile(padded, 95, axis=0)
avg_ranks  = np.arange(1, max_len + 1)

# Trim trailing zeros from mean
nonzero = mean_freq > 0
mean_freq = mean_freq[nonzero]
p05       = p05[nonzero]
p95       = p95[nonzero]
avg_ranks = avg_ranks[nonzero]

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

ax.fill_between(avg_ranks, p05, p95,
                alpha=0.2, color="#41ab5d", label="5–95th percentile")
ax.loglog(avg_ranks, mean_freq, "-", color="#41ab5d", lw=2,
          label=f"Mean over {N_RUNS} runs")
ax.loglog(emp_ranks, emp_freq, "o", ms=4, alpha=0.6, color="#2171b5",
          label=f"Empirical data (top {N_TOP})")

ax.set_xlabel("Rank", fontsize=13)
ax.set_ylabel("Frequency", fontsize=13)
ax.set_title(
    f"Averaged model distribution vs empirical data\n"
    f"($n={N_CLASSES_VALID}$, $n_i={N_INIT_VALID}$, $n_f={N_FINAL_VALID}$, "
    f"$T={T_MAX}$, $\\alpha={ARCHIVE_RATE}$; {N_RUNS} runs)",
    fontsize=12
)
ax.legend(fontsize=11)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "loglog_model_avg.png", bbox_inches="tight")
plt.close()
print("  Saved loglog_model_avg.png")

# ═══════════════════════════════════════════════════════════════════════════
# 8. SUMMARY FOR ARTICLE TEXT
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("RESULTS TO REPORT IN THE ARTICLE")
print("="*60)
print(f"Empirical power-law exponent (176 cat) : α = {alpha_emp:.2f} ± {sigma_emp:.2f}")
print(f"Model power-law exponent (n=40)        : α = {alpha_model:.2f} ± {sigma_model:.2f}")
print(f"Model power-law exponent (n=176)       : α = {alpha_valid:.2f} ± {sigma_valid:.2f}")
print(f"KS statistic (n=176, fair comparison)  : D = {ks_stat:.4f}")
print(f"KS p-value                             : p = {ks_p:.4f}")
print(f"Interpretation: {'compatible (p>0.05)' if ks_p > 0.05 else 'significant difference (p<0.05)'}")
