"""What the ligand-only ceiling actually bounds, and which representations it binds.

THE CLAIM BEING REPAIRED. Earlier drafts derived a ceiling of 0.8875 from the
within-structure variance of potency and then applied it to all three comparison
representations, concluding for instance that PaDEL at 0.567 had reached 64 per cent
of its attainable maximum. That reasoning is only valid for a representation that
assigns one and the same vector to every row of a given structure. It was never
checked. It is false for at least one of the four sets, so the arithmetic it
supported was wrong.

WHAT IS ACTUALLY TRUE. The decomposition itself is sound. Write y_ij for the potency
of row j of structure i, y_i. for the mean of structure i and y.. for the grand mean:

    SS_total  = sum_ij (y_ij - y..)^2
    SS_within = sum_ij (y_ij - y_i.)^2
    R2_max    = 1 - SS_within / SS_total

Let f be any predictor that is constant across the rows of a structure, taking the
value c_i on structure i. Its residual sum of squares splits exactly, with no cross
term, because the deviations y_ij - y_i. sum to zero inside each structure:

    SS_res = sum_ij (y_ij - c_i)^2
           = sum_ij (y_ij - y_i.)^2 + sum_i n_i (y_i. - c_i)^2
           = SS_within + sum_i n_i (y_i. - c_i)^2
           >= SS_within

so R2 = 1 - SS_res / SS_total <= R2_max, with equality only when c_i = y_i. for
every structure. No unbiasedness, linearity or distributional assumption is needed.
The conditions are that R2 is the usual one minus squared error over total sum of
squares on the rows being evaluated, and that the predictor has no row-varying input
beyond the representation. Equality is an oracle in-sample maximum, not a claim that
a learner can estimate every structure mean for an unseen structure.

ONE CAVEAT THE EARLIER DRAFT MISSED. R2_max above is computed on all 122 rows, but
every number it is compared against is a held-out coefficient of determination
computed on a test partition of 24 to 33 rows, whose total sum of squares is taken
about that partition's own mean. The bound therefore has to be recomputed inside each
partition to be commensurable with what it is being compared to, and partitions differ
sharply: one that happens to contain few recurring structures has a ceiling close to
one. Both the pooled figure and the per-partition distribution are reported, and the
per-partition median is the one the results should be read against.

THREE MEASUREMENTS PER REPRESENTATION.

1. Within-structure share of representational variance. Standardise every column,
   then apply the same decomposition to the features instead of the target and
   average over columns:

       rho_X = mean_k [ SS_within(x_k) / SS_total(x_k) ]

   The mean gives every retained column equal weight, including columns whose small
   raw variance is amplified by standardisation. Its median, upper tail and fraction
   at zero are therefore reported too. rho_X = 0 is exact constancy, and the ceiling
   binds absolutely. The number is meaningless without its null. A column carrying
   no relation whatever to the
   grouping has expectation

       rho_0 = (m - r) / (n - 1)

   where n is the number of rows, r the number of structures occurring more than
   once and m the number of rows belonging to them, because the singleton
   structures contribute nothing to the within-structure sum of squares. Here
   rho_0 = (47 - 19) / 121 = 0.2314. A representation at rho_0 separates two rows
   of one molecule exactly as much as it separates two unrelated molecules, which
   is the opposite of the constancy the ceiling assumes, and a representation at
   zero is exactly constant. Both extremes occur in these four sets.

   A correctly computed MACCS fingerprint is carried through the whole analysis as
   a positive control, since it is a function of the molecular graph and must give
   rho = 0 exactly. It does. The MACCS block stored in the study matrix does not,
   which is how the block was found to be corrupt; see maccs_fix.py.

2. Separation. Median Euclidean distance between rows of the same structure over
   median distance between rows of different structures, in the standardised space.
   This says whether the variation in 1 is large enough to matter next to the
   variation the representation uses to tell molecules apart.

3. Alignment is an exploratory test of whether within-structure variation looks
   useful. Over all pairs of rows sharing a structure, correlate representational
   distance with absolute potency difference. It does not decide whether the hard
   bound applies. Any genuine variation removes that bound for an unrestricted
   predictor, whether or not this particular monotone distance association is found.

   The 38 pairs are not independent. A structure contributing four rows contributes
   six pairs, and pairs sharing a row are correlated by construction, so the p value
   scipy attaches to a Spearman coefficient is wrong here and is not used. The null
   is instead generated by permuting potency within structures. That permutation is
   exact under the hypothesis being tested, namely that rows of one structure are
   exchangeable given the representation, and it reproduces the dependence among
   pairs in every permuted sample because the pair structure is a property of the
   design and is held fixed. The statistic may be dependent; the randomisation test
   around it is still valid. Eleven of the 19 recurring structures have only two
   rows. Swapping those two labels leaves their absolute difference unchanged, so
   only eight structures contribute to the randomisation distribution.

   Power is the real limitation, and it is reported rather than left implicit. Even
   the optimistic calculation treating all 38 pairs as independent needs a one-sided
   correlation near 0.40 for eighty per cent power. Treating the 19 structures as
   independent units raises that benchmark to about 0.55. Actual power also depends
   on a row-level outcome model and cannot be recovered from the pair count alone.
   A cluster bootstrap over structures supplies an approximate confidence interval.

The honest conclusion is narrower. Exact constancy makes 0.8875 a mathematical
bound. For a representation that varies, the bound is not established and a
nonsignificant alignment test cannot put it back. It says only that these data do
not resolve whether the variation is useful.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "benchmark_work" / "src"))
import datasets  # noqa: E402
sys.path.insert(0, str(ROOT))
from maccs_fix import maccs_frame  # noqa: E402

RESULTS = ROOT / "results"
SETS = ["md_core89", "padel", "mol2desc", "dft", "maccs_correct", "maccs167"]
LABEL = {"md_core89": "Simulation derived (89)", "padel": "PaDEL (1444)",
         "mol2desc": "PyDescriptor (62571)", "dft": "Quantum chemical (109)",
         "maccs_correct": "MACCS keys (167), recomputed, positive control",
         "maccs167": "MACCS keys (167) as stored in the study matrix"}


def load_set(key):
    if key == "maccs_correct":
        return maccs_frame()[0]
    return datasets.load(key)[0]


N_PERM = 20000
N_BOOT = 10000
SEED = 20260815


def detectable_correlation(n, alpha=0.05, power=0.80):
    """One-sided Fisher approximation for independent observations."""
    z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)
    return float(np.tanh(z / np.sqrt(n - 3)))


DETECTABLE_PAIR = detectable_correlation(38)
DETECTABLE_STRUCTURE = detectable_correlation(19)


def decompose(values, groups):
    """SS_total and SS_within for a one-dimensional quantity under a grouping."""
    v = np.asarray(values, dtype=float)
    ss_total = float(((v - v.mean()) ** 2).sum())
    ss_within = 0.0
    for g in np.unique(groups):
        m = groups == g
        ss_within += float(((v[m] - v[m].mean()) ** 2).sum())
    return ss_total, ss_within


def standardise(X, constant_groups=None):
    """Impute nonfinite cells, then z-score and drop constant columns."""
    A = X.to_numpy(dtype=float)
    finite = np.isfinite(A)
    n_imputed = int((~finite).sum())
    # Missing positive-control keys reflect molecule parsing, not a descriptor
    # value. Copy a key from the same structure where possible. If the whole block
    # is missing, a common zero fill still preserves the property being controlled.
    if constant_groups is not None:
        for g in np.unique(constant_groups):
            idx = np.flatnonzero(constant_groups == g)
            block = A[idx]
            block_finite = np.isfinite(block)
            block_counts = block_finite.sum(axis=0)
            block_sums = np.where(block_finite, block, 0.0).sum(axis=0)
            block_means = np.divide(
                block_sums, block_counts,
                out=np.zeros(A.shape[1], dtype=float), where=block_counts > 0)
            A[idx] = np.where(block_finite, block, block_means[None, :])
        finite = np.isfinite(A)
    counts = finite.sum(axis=0)
    sums = np.where(finite, A, 0.0).sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros(A.shape[1], dtype=float),
                      where=counts > 0)
    A = np.where(finite, A, means[None, :])
    A -= A.mean(axis=0)
    sd = A.std(axis=0)
    keep = sd > 0
    return A[:, keep] / sd[keep], int(keep.sum()), n_imputed


def pair_index(groups):
    """All row pairs that share a structure and all pairs that do not."""
    same = []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                same.append((idx[a], idx[b]))
    n = len(groups)
    diff = [(a, b) for a in range(n) for b in range(a + 1, n) if groups[a] != groups[b]]
    return np.array(same), np.array(diff)


def pair_distances(A, pairs, batch=256):
    """Euclidean distances without allocating one full wide pair matrix."""
    out = np.empty(len(pairs))
    scale = np.sqrt(A.shape[1])
    for start in range(0, len(pairs), batch):
        q = pairs[start:start + batch]
        delta = A[q[:, 0]] - A[q[:, 1]]
        out[start:start + len(q)] = np.sqrt(
            np.einsum("ij,ij->i", delta, delta)) / scale
    return out


def cluster_bootstrap_ci(d_same, dy, pair_groups, rng):
    """Percentile interval from resampling whole recurring structures."""
    blocks = [np.flatnonzero(pair_groups == g) for g in np.unique(pair_groups)]
    boot = np.full(N_BOOT, np.nan)
    for b in range(N_BOOT):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        take = np.concatenate([blocks[k] for k in chosen])
        x, z = d_same[take], dy[take]
        if np.ptp(x) > 0 and np.ptp(z) > 0:
            boot[b] = stats.spearmanr(x, z).statistic
    boot = boot[np.isfinite(boot)]
    if len(boot) == 0:
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.quantile(boot, [0.025, 0.975]))


def holm_adjust(p_values):
    """Holm adjustment for the four scientific comparison sets."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    _, y, meta = datasets.load("md_core89")
    groups = datasets.groups_for(meta)
    y = y.to_numpy(dtype=float)

    ss_total, ss_within = decompose(y, groups)
    ceiling = 1 - ss_within / ss_total
    rmse_floor = float(np.sqrt(ss_within / len(y)))
    same, diff = pair_index(groups)
    recurring = int(sum(len(np.flatnonzero(groups == g)) > 1 for g in np.unique(groups)))
    recurring_sizes = [(groups == g).sum() for g in np.unique(groups)
                       if (groups == g).sum() > 1]
    two_row = int(sum(n == 2 for n in recurring_sizes))
    randomisable = int(sum(n > 2 for n in recurring_sizes))
    randomisable_pairs = int(sum(n * (n - 1) // 2 for n in recurring_sizes if n > 2))

    header = pd.DataFrame([{
        "Rows": len(y), "Distinct structures": len(np.unique(groups)),
        "Structures occurring more than once": recurring,
        "Row pairs sharing a structure": len(same),
        "Two-row recurring structures": two_row,
        "Structures contributing to the alignment randomisation": randomisable,
        "Pairs in those structures": randomisable_pairs,
        "SS total": round(ss_total, 3), "SS within structure": round(ss_within, 3),
        "Ceiling for a structure-constant predictor": round(ceiling, 4),
        "Root mean squared error floor": round(rmse_floor, 4),
        "Largest within-structure potency range":
            round(float(max(y[groups == g].max() - y[groups == g].min()
                            for g in np.unique(groups))), 3)}])
    header.to_csv(RESULTS / "ceiling_target.csv", index=False)

    # the same bound recomputed inside each held-out partition, which is the only
    # form comparable with a held-out coefficient of determination
    sys.path.insert(0, str(ROOT))
    from robust import outer_splits  # noqa: E402  (imported here to keep the
    per = []                        # module usable without the sweep dependencies)
    for rep, (_tr, te) in enumerate(outer_splits(groups, 10)):
        t, w = decompose(y[te], groups[te])
        rec = sum(1 for g in np.unique(groups[te]) if (groups[te] == g).sum() > 1)
        per.append({"partition": rep, "rows": len(te),
                    "structures": int(len(np.unique(groups[te]))),
                    "recurring structures": int(rec),
                    "ceiling": round(1 - w / t, 4)})
    per = pd.DataFrame(per)
    per.to_csv(RESULTS / "ceiling_per_partition.csv", index=False)

    # expectation of the within-structure variance share for a column unrelated to
    # the grouping: only rows in recurring structures contribute to the numerator
    m_rows = int(sum((groups == g).sum() for g in np.unique(groups)
                     if (groups == g).sum() > 1))
    rho_null = (m_rows - recurring) / (len(y) - 1)

    dy = np.abs(y[same[:, 0]] - y[same[:, 1]])
    pair_groups = groups[same[:, 0]]
    perm_blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)
                   if (groups == g).sum() > 1]
    rows = []
    for set_number, key in enumerate(SETS):
        X = load_set(key)
        assert len(X) == len(y), f"{key}: {len(X)} rows, expected {len(y)}"
        A, n_kept, n_imputed = standardise(
            X, groups if key == "maccs_correct" else None)

        # 1. within-structure share of representational variance. Vectorised over
        # columns: PyDescriptor has 62571 of them, and a per-column Python loop over
        # 94 groups would be a quarter of a billion slice operations.
        codes = pd.factorize(groups)[0]
        means = np.zeros((codes.max() + 1, A.shape[1]))
        counts = np.bincount(codes).astype(float)
        np.add.at(means, codes, A)
        means /= counts[:, None]
        ss_w = ((A - means[codes]) ** 2).sum(axis=0)
        ss_t = (A ** 2).sum(axis=0)          # columns are already centred by standardise
        ratios = ss_w[ss_t > 0] / ss_t[ss_t > 0]
        # The mean gives each standardised column equal weight. Quantiles and the
        # zero fraction reveal whether it describes a typical column or a mixture.
        rho = float(np.mean(ratios))
        rho_med = float(np.median(ratios))
        rho_95 = float(np.quantile(ratios, 0.95))
        rho_99 = float(np.quantile(ratios, 0.99))
        frac_const = float(np.mean(ratios < 1e-12))

        # 2. separation, normalised by dimension so widths are comparable. Batches
        # avoid a multi-gigabyte temporary array for the widest representation.
        d_same = pair_distances(A, same)
        d_diff = pair_distances(A, diff)
        ratio = float(np.median(d_same) / np.median(d_diff))

        # 3. alignment, with a within-structure permutation null. An exactly constant
        # representation gives zero distance for every same-structure pair, so the
        # correlation is undefined rather than zero: there is nothing to align. That
        # is the positive control passing, not a failure, and it is reported as such.
        exact_constant = bool(np.all(d_same == 0))
        if np.ptp(d_same) == 0:
            obs, p_perm, lo, hi = (float("nan"),) * 4
        else:
            obs = float(stats.spearmanr(d_same, dy).statistic)
            lo, hi = cluster_bootstrap_ci(
                d_same, dy, pair_groups,
                np.random.default_rng(SEED + 1000 + set_number))
            null = np.empty(N_PERM)
            yp = y.copy()
            rng = np.random.default_rng(SEED + set_number)
            for p in range(N_PERM):
                for b in perm_blocks:
                    yp[b] = rng.permutation(y[b])
                null[p] = stats.spearmanr(
                    d_same, np.abs(yp[same[:, 0]] - yp[same[:, 1]])).statistic
            p_perm = float((1 + (null >= obs).sum()) / (1 + N_PERM))

        # Exact constancy is the condition for the mathematical bound. A test that
        # fails to detect alignment cannot make a varying representation constant.
        if exact_constant:
            verdict = "applies exactly, representation constant within structure"
        elif key == "maccs167":
            verdict = "not interpretable, stored representation is corrupt"
        else:
            verdict = "not a proven bound, representation varies within structure"
        if key == "maccs_correct":
            # Centring a column then subtracting group means leaves residue at the
            # scale of the floating point epsilon, so the gate is a tolerance rather
            # than an equality. Measured here: rho = 1.1e-34, largest column ratio
            # 4.8e-33, and every same-structure distance exactly zero.
            assert rho < 1e-20 and exact_constant, (
                f"recomputed MACCS positive control is not constant: rho = {rho:.3e}, "
                f"all same-structure distances zero = {exact_constant}")
        rows.append({
            "Representation": LABEL[key],
            "Columns with variance": n_kept,
            "Nonfinite cells imputed": n_imputed,
            "Within-structure share of variance, mean": round(rho, 4),
            "Within-structure share of variance, median": round(rho_med, 4),
            "Within-structure share, 95th percentile": round(rho_95, 4),
            "Within-structure share, 99th percentile": round(rho_99, 4),
            "Columns at numerical zero within structure": round(frac_const, 4),
            "Share relative to the unrelated-column null": round(rho / rho_null, 3),
            "Same-structure over different-structure distance": round(ratio, 4),
            "Alignment with potency difference":
                "not defined, constant" if np.isnan(obs) else round(obs, 4),
            "Alignment 95 per cent interval":
                "not defined, constant" if np.isnan(obs)
                else f"{lo:.3f} to {hi:.3f}",
            "Permutation p": ("not defined, constant" if np.isnan(p_perm)
                              else round(p_perm, 4)),
            "Holm p for four comparison sets": "pending",
            "Alignment conclusion": "pending",
            "Status of the ligand-only ceiling": verdict,
            "_p": p_perm})
        print(f"{key:11s} rho={rho:.4f} ratio={ratio:.4f} align={obs:+.4f} "
              f"p={p_perm:.4f}", flush=True)

    adjusted = holm_adjust([r["_p"] for r in rows[:4]])
    for i, row in enumerate(rows):
        if i < 4:
            row["Holm p for four comparison sets"] = round(float(adjusted[i]), 4)
            row["Alignment conclusion"] = (
                "positive alignment detected" if adjusted[i] <= 0.05
                else "undetermined at this sample size")
        elif np.isnan(row["_p"]):
            row["Holm p for four comparison sets"] = "not defined, constant"
            row["Alignment conclusion"] = "not testable, constant"
        else:
            row["Holm p for four comparison sets"] = "diagnostic only"
            row["Alignment conclusion"] = "diagnostic only, block is corrupt"
        del row["_p"]

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "ceiling_representations.csv", index=False)
    print()
    print(header.T.to_string(header=False))
    print(f"\nunrelated-column null for the variance share: {rho_null:.4f}")
    print(f"optimistic rank correlation detectable at 80 per cent power, "
          f"38 independent pairs: {DETECTABLE_PAIR:.3f}")
    print(f"rank correlation detectable at 80 per cent power, "
          f"19 independent structures: {DETECTABLE_STRUCTURE:.3f}")
    print()
    print("ceiling recomputed inside each held-out partition:")
    print(per.to_string(index=False))
    print(f"median {per.ceiling.median():.4f}, range {per.ceiling.min():.4f} to "
          f"{per.ceiling.max():.4f}")
    print()
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
