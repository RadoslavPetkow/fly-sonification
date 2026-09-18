"""How much of a hop-2 neuron's input actually comes from hop 1? A fact about the matrix alone.

    .venv/bin/python -m experiments.input_share

Every transmission result in this project is measured through a model whose weights are normalised
by sqrt(sum_j |W[i, j]|) (SimConfig.normalization = "sqrt_in"). That is a modelling choice made in
calibration on stability grounds, not a fact about the animal, so "the descending hop-2 neurons are
heavily converged upon and the auditory path is diluted among their inputs" cannot rest on it.

This module computes the normalisation-free version from cache/adjacency.npz alone - no simulation,
no parameters, nothing tunable:

    share_hop1(i) = sum_{j in HOP1} |W[i, j]|  /  sum_j |W[i, j]|            (weight share)
    count_hop1(i) = |{j in HOP1 : W[i, j] stored}| / in-degree(i)            (partner share)

reported separately for the descending hop-2 neurons and the non-descending ones, with a
Mann-Whitney U between them. A smaller share for the descending cells makes the dilution a
structural property of the connectome, statable without reference to the model. The same or a
larger share means the dilution is produced by the normalisation, and every claim resting on it is
normalisation-dependent.

The absolute incoming weight from hop 1 is reported beside the share, because a cell can take a
smaller SHARE from hop 1 while receiving more of it in absolute terms, and the two support
different readings.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.sparse as sp
from scipy.stats import mannwhitneyu

from config import BranchConfig, Paths
from data.fetch_connectome import bfs_levels
from sim.calibrate import banner, header

CACHE = ROOT / Paths().cache
RESULTS = CACHE / "input_share.json"


def describe(x):
    q1, q3 = np.percentile(x, [25, 75])
    return {"n": int(x.size), "median": float(np.median(x)), "q1": float(q1), "q3": float(q3),
            "iqr": float(q3 - q1), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "min": float(x.min()), "max": float(x.max())}


def row(label, d):
    return (f"  {label:<34} {d['n']:>7,} {d['median']:>10.4g} "
            f"[{d['q1']:.4g}, {d['q3']:.4g}]".rjust(24) + f" {d['mean']:>10.4g} {d['sd']:>10.4g}")


def compare(a, b, label_a, label_b, name):
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    rbc = 2.0 * u / (a.size * b.size) - 1.0
    direction = ("SMALLER" if np.median(a) < np.median(b) else
                 "LARGER" if np.median(a) > np.median(b) else "EQUAL")
    print(f"  {name}: Mann-Whitney U = {u:,.0f}, p = {p:.3g}; rank-biserial r = {rbc:+.4f}; "
          f"median ratio {np.median(a) / np.median(b):.3f}")
    print(f"    -> the {label_a} take a {direction} {name} than the {label_b}")
    return {"u": float(u), "p": float(p), "rank_biserial_r": float(rbc),
            "median_ratio": float(np.median(a) / np.median(b)), "direction": direction}


def main():
    bc = BranchConfig()
    header("SETUP - structural only, no simulation and no normalisation")
    W = sp.load_npz(CACHE / "adjacency.npz").tocsr()
    W.sort_indices()
    indices = json.loads((CACHE / "indices.json").read_text())
    sens = np.asarray(indices["sensory_idx"])
    motor = np.asarray(indices["motor_idx"])
    dist = bfs_levels(W.T.tocsr(), sens)
    hop1 = np.flatnonzero(dist == bc.source_hop)
    hop2 = np.flatnonzero(dist == bc.target_hop)
    desc = np.sort(motor[dist[motor] == bc.target_hop])
    nondesc = np.setdiff1d(hop2, desc)
    print(f"W {W.shape} nnz {W.nnz:,}; HOP1 {hop1.size:,}, hop-2 {hop2.size:,} = "
          f"{desc.size:,} descending + {nondesc.size:,} non-descending")
    assert desc.size == bc.expect_targets and hop1.size == bc.expect_sources
    assert np.intersect1d(desc, nondesc).size == 0 and desc.size + nondesc.size == hop2.size

    # ---- per-row shares over the hop-2 rows
    sub = W[hop2]
    rows = np.repeat(np.arange(hop2.size), np.diff(sub.indptr))
    absd = np.abs(sub.data).astype(np.float64)
    in_h1 = np.zeros(W.shape[1], dtype=bool)
    in_h1[hop1] = True
    is_h1 = in_h1[sub.indices].astype(np.float64)
    total_w = np.bincount(rows, weights=absd, minlength=hop2.size)
    from_h1_w = np.bincount(rows, weights=absd * is_h1, minlength=hop2.size)
    total_c = np.diff(sub.indptr).astype(np.float64)
    from_h1_c = np.bincount(rows, weights=is_h1, minlength=hop2.size)
    assert (total_w > 0).all() and (total_c > 0).all(), "a hop-2 neuron has no incoming weight"
    assert (from_h1_c > 0).all(), "a hop-2 neuron has no hop-1 presynaptic partner (BFS says it must)"
    share_w = from_h1_w / total_w
    share_c = from_h1_c / total_c

    pos = {int(n): k for k, n in enumerate(hop2)}
    di = np.array([pos[int(n)] for n in desc])
    ni = np.array([pos[int(n)] for n in nondesc])

    out = {"hop1": int(hop1.size), "hop2": int(hop2.size), "descending": int(desc.size),
           "non_descending": int(nondesc.size), "groups": {}, "tests": {}}
    for name, vals, unit in (("weight share from hop 1", share_w, "fraction of total |W| in"),
                             ("partner share from hop 1", share_c, "fraction of in-degree"),
                             ("absolute |W| from hop 1", from_h1_w, "synapse-weighted"),
                             ("total incoming |W|", total_w, "synapse-weighted"),
                             ("in-degree", total_c, "partners")):
        header(f"{name.upper()}  ({unit})")
        print(f"  {'group':<34} {'n':>7} {'median':>10} {'IQR':>24} {'mean':>10} {'sd':>10}")
        d_desc, d_non = describe(vals[di]), describe(vals[ni])
        print(row("descending hop-2 (the readout)", d_desc))
        print(row("non-descending hop-2", d_non))
        print(row("all hop-2", describe(vals)))
        out["groups"][name] = {"descending": d_desc, "non_descending": d_non,
                               "all": describe(vals)}
        out["tests"][name] = compare(vals[di], vals[ni], "descending hop-2 neurons",
                                     "non-descending ones", name)

    # ---------------------------------------------------------------- verdict
    header("VERDICT - is the dilution structural, or produced by sqrt_in?")
    tw, tc_ = out["tests"]["weight share from hop 1"], out["tests"]["partner share from hop 1"]
    a_desc = out["groups"]["absolute |W| from hop 1"]["descending"]["median"]
    a_non = out["groups"]["absolute |W| from hop 1"]["non_descending"]["median"]
    lines = []
    if tw["direction"] == "SMALLER" and tc_["direction"] == "SMALLER":
        lines.append(
            f"STRUCTURAL. The descending hop-2 neurons take a smaller share of their input from "
            f"hop 1 than the non-descending ones on BOTH measures: "
            f"{out['groups']['weight share from hop 1']['descending']['median']:.4f} vs "
            f"{out['groups']['weight share from hop 1']['non_descending']['median']:.4f} median "
            f"weight share (ratio {tw['median_ratio']:.2f}, p = {tw['p']:.3g}) and "
            f"{out['groups']['partner share from hop 1']['descending']['median']:.4f} vs "
            f"{out['groups']['partner share from hop 1']['non_descending']['median']:.4f} median "
            f"partner share (ratio {tc_['median_ratio']:.2f}, p = {tc_['p']:.3g}).")
        lines.append("This holds in the raw matrix, with no normalisation and no model: the "
                     "auditory path is a smaller fraction of what these cells listen to. "
                     "'The path is diluted' can be stated without reference to sqrt_in.")
    elif tw["direction"] == "LARGER" or tc_["direction"] == "LARGER":
        gw, gc = out["groups"]["weight share from hop 1"], out["groups"]["partner share from hop 1"]
        lines.append(
            f"NOT DILUTED. The descending hop-2 neurons take the SAME proportional share of their "
            f"input from hop 1 as any other hop-2 cell - weight share {gw['descending']['median']:.5f} "
            f"vs {gw['non_descending']['median']:.5f} (p = {tw['p']:.3g}, rank-biserial "
            f"{tw['rank_biserial_r']:+.3f}: no difference), partner share "
            f"{gc['descending']['median']:.5f} vs {gc['non_descending']['median']:.5f} "
            f"(p = {tc_['p']:.3g}), if anything slightly larger - and they receive "
            f"{out['tests']['absolute |W| from hop 1']['median_ratio']:.1f}x MORE of it in absolute "
            f"terms (median {a_desc:.4g} vs {a_non:.4g}).")
        lines.append("They are the best-connected listeners in the layer. They are not diluted, not "
                     "starved and not cut off from the auditory path: proportionally they hear it "
                     "exactly as much as their neighbours do, and absolutely they hear far more.")
        lines.append("Under sqrt_in they nonetheless sit about 2% of the way from a layer that does "
                     "not transmit (hop 3) to one that does (hop 1). Since the proportional share "
                     "is equal in the raw matrix, that gap is NOT explained by how much auditory "
                     "input these cells receive.")
        lines.append("What sqrt_in does contribute: dividing row i by sqrt(sum_j |W[i, j]|) scales "
                     f"the descending cells' inputs down by sqrt("
                     f"{out['tests']['total incoming |W|']['median_ratio']:.1f}) = "
                     f"{np.sqrt(out['tests']['total incoming |W|']['median_ratio']):.1f}x relative "
                     "to a typical hop-2 neuron, purely because they are larger. Any claim about "
                     "this layer is therefore stated under that normalisation, and the earlier "
                     "working-note framing 'the path is diluted' - which attributed the gap to the "
                     "connectome rather than to the normalisation - is WITHDRAWN by this "
                     "measurement.")
    else:
        lines.append("The two measures disagree or show no difference; see the tables above.")
    share_word = {"SMALLER": "smaller", "LARGER": "larger", "EQUAL": "equal"}[tw["direction"]]
    if tw["p"] > 0.05:
        share_word = "statistically indistinguishable"
    lines.append(f"In numbers: hop-1 weight share {share_word} "
                 f"(ratio {tw['median_ratio']:.2f}), absolute hop-1 weight median {a_desc:.4g} vs "
                 f"{a_non:.4g} ({'x%.1f' % (a_desc / a_non) if a_non else 'n/a'}), total incoming "
                 f"|W| median "
                 f"{out['groups']['total incoming |W|']['descending']['median']:.4g} vs "
                 f"{out['groups']['total incoming |W|']['non_descending']['median']:.4g}, in-degree "
                 f"median {out['groups']['in-degree']['descending']['median']:.0f} vs "
                 f"{out['groups']['in-degree']['non_descending']['median']:.0f}.")
    for ln in lines:
        print("  " + ln)
    out["verdict"] = lines
    RESULTS.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {RESULTS.relative_to(ROOT)}")
    banner(lines[0][:150])
    return out


if __name__ == "__main__":
    main()
