"""Load-time normalization of the signed adjacency matrix W[i_post, j_pre].

    raw        W unchanged
    in_degree  row i divided by sum_j |W[i, j]|        (total |input| of every neuron = 1)
    sqrt_in    row i divided by sqrt(sum_j |W[i, j]|)

Neurons with no incoming weight keep an all-zero row (scale 1), so nothing
divides by zero. w_scale is NOT applied here; it stays a free scalar applied to
the matmul result at run time.
"""
import numpy as np
import scipy.sparse as sp

MODES = ("raw", "in_degree", "sqrt_in")


def normalize(W, mode):
    """Return (normalized float32 CSR copy of W, stats dict)."""
    if mode not in MODES:
        raise ValueError(f"normalization must be one of {MODES}, got {mode!r}")
    W = sp.csr_matrix(W, dtype=np.float32, copy=True)
    W.sort_indices()
    in_abs = np.asarray(abs(W).sum(axis=1), dtype=np.float64).ravel()
    zero = in_abs == 0
    stats = {"mode": mode, "neurons": W.shape[0], "zero_in_degree": int(zero.sum()),
             "in_abs_median_before": float(np.median(in_abs)), "in_abs_max_before": float(in_abs.max())}

    if mode != "raw":
        denom = in_abs if mode == "in_degree" else np.sqrt(in_abs)
        scale = np.ones_like(in_abs)
        scale[~zero] = 1.0 / denom[~zero]
        rows = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr))
        W.data *= scale[rows].astype(np.float32)

    if not np.isfinite(W.data).all():
        raise FloatingPointError(f"normalization {mode!r} produced non-finite weights")
    after = np.asarray(abs(W).sum(axis=1), dtype=np.float64).ravel()
    stats.update(in_abs_median_after=float(np.median(after)), in_abs_max_after=float(after.max()),
                 weight_abs_max_after=float(np.abs(W.data).max()))
    return W, stats


def apply_inhibitory_gain(W, g_inh):
    """Multiply the negative entries of W (in place) by g_inh. Returns (W, E/I stats)."""
    if not g_inh > 0:
        raise ValueError(f"g_inh must be > 0, got {g_inh}")
    neg = W.data < 0
    if g_inh != 1.0:
        W.data[neg] *= np.float32(g_inh)
    if not np.isfinite(W.data).all():
        raise FloatingPointError("inhibitory gain produced non-finite weights")
    exc = float(W.data[~neg].sum(dtype=np.float64))
    inh = float(-W.data[neg].sum(dtype=np.float64))
    return W, {"g_inh": float(g_inh), "exc_weight": exc, "inh_weight": inh,
               "exc_fraction": exc / (exc + inh) if exc + inh > 0 else float("nan")}


if __name__ == "__main__":
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    W = sp.load_npz(root / "cache" / "adjacency.npz")
    print(f"W {W.shape}, nnz {W.nnz:,}")
    for m in MODES:
        Wn, s = normalize(W, m)
        print(s)
        for g in (1.0, 2.0, 3.0, 5.0, 8.0):
            print("  ", apply_inhibitory_gain(Wn.copy(), g)[1])
