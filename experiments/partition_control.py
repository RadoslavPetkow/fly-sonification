"""Is the structural band selectivity just JO type structure?  (No simulation.)

    .venv/bin/python -m experiments.partition_control

PREDICTION (stated before computing): if the selectivity comes from JO type structure, the type-sorted partition
of the 86 sensory neurons shows a large real-minus-shuffled entropy gap, and random partitions into groups of the
same sizes show little or none. A comparable gap under random partitions would mean the effect is not about type.

For each partition (the type-sorted one, and ShuffleConfig.partition_n random ones, rng seed partition_seed0 + p):
  real entropy      mean normalized entropy of the 56 hop-1 motor neurons' sensory input across the groups
                    (experiments/structural_selectivity.py definitions)
  shuffled entropy  mean of the same over partition_shuffles sensory-only re-targetings with that partition's own
                    seeds (experiments.shuffle_graph.retarget; the re-targeting does not depend on the partition)
  gap               real - shuffled (negative = more selective than chance); z = gap / SD of that ensemble
Report: the type-sorted gap as a percentile of the random-partition gaps.
Writes cache/partition_control.json, figures/partition_control.png, experiments/partition_control.md.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from config import AudioConfig, Paths, ShuffleConfig
from data.fetch_connectome import bfs_levels
from data.groups import load_neurons_and_indices, ordered_groups
from experiments.shuffle_graph import retarget
from experiments.structural_selectivity import input_matrix, selectivity, sensory_edges
from sim.calibrate import CATEGORICAL, GRID, TEXT, TEXT_2, header, save, style

CACHE = ROOT / Paths().cache
OUT_JSON = CACHE / "partition_control.json"
OUT_MD = ROOT / "experiments" / "partition_control.md"


def main():
    sc = ShuffleConfig()
    neurons, indices = load_neurons_and_indices()
    N = len(neurons)
    sens = np.asarray(indices["sensory_idx"])
    motor = np.asarray(indices["motor_idx"])
    n_groups = AudioConfig().n_bins
    type_groups = ordered_groups(neurons, sens, n_groups)
    sizes = [len(g) for g in type_groups]
    W = sp.load_npz(CACHE / "adjacency.npz").tocsr()
    voices = motor[bfs_levels(W.T.tocsr(), sens)[motor] == 1]
    e_pre, e_post, e_w = sensory_edges(W, sens)
    header(f"PARTITION CONTROL - {sc.partition_n} random partitions (sizes {sizes}) x {sc.partition_shuffles} shuffles each")
    print("PREDICTION: type structure -> large gap for the type-sorted partition, little or none for random partitions")

    def group_of_from(groups):
        g_of = np.full(N, -1)
        for g, members in enumerate(groups):
            g_of[members] = g
        return g_of

    def evaluate(group_of, seed_base):
        real = np.nanmean(selectivity(input_matrix(e_pre, e_post, e_w, voices, group_of, n_groups))["entropy"])
        sh = []
        for i in range(sc.partition_shuffles):
            new_post, _ = retarget(e_pre, e_post, N, np.random.default_rng(seed_base + i), sc.max_repair_iterations)
            sh.append(np.nanmean(selectivity(input_matrix(e_pre, new_post, e_w, voices, group_of, n_groups))["entropy"]))
        sh = np.array(sh)
        return {"real": float(real), "shuffled_mean": float(sh.mean()), "shuffled_sd": float(sh.std(ddof=1)),
                "gap": float(real - sh.mean()), "z": float((real - sh.mean()) / sh.std(ddof=1)),
                "p_one_sided": float((1 + np.sum(sh <= real)) / (1 + sh.size))}

    typ = evaluate(group_of_from(type_groups), sc.partition_seed0 - sc.partition_shuffles)
    print(f"type-sorted: real {typ['real']:.4f}, shuffled {typ['shuffled_mean']:.4f} (sd {typ['shuffled_sd']:.4f}), "
          f"gap {typ['gap']:+.4f}, z {typ['z']:+.2f}")
    rand = []
    splits = np.cumsum(sizes)[:-1]
    for p in range(sc.partition_n):
        perm = np.random.default_rng(sc.partition_seed0 + p).permutation(sens.size)
        groups = [sens[part] for part in np.split(perm, splits)]
        r = evaluate(group_of_from(groups), sc.partition_seed0 + (p + 1) * sc.partition_shuffles)
        rand.append(r)
        if (p + 1) % 50 == 0:
            gaps = np.array([x["gap"] for x in rand])
            print(f"  {p + 1} partitions: gap mean {gaps.mean():+.4f}, sd {gaps.std(ddof=1):.4f}", flush=True)
    gaps = np.array([x["gap"] for x in rand])
    zs = np.array([x["z"] for x in rand])
    pct_below = float(np.mean(gaps <= typ["gap"]) * 100)
    res = {"prediction": "type structure -> large gap for type-sorted partition, little or none for random partitions",
           "group_sizes": sizes, "type_sorted": typ, "random": rand,
           "random_gap_mean": float(gaps.mean()), "random_gap_sd": float(gaps.std(ddof=1)),
           "random_gap_percentiles": {str(q): float(np.percentile(gaps, q)) for q in (0, 5, 25, 50, 75, 95, 100)},
           "type_gap_percentile_among_random": pct_below,
           "random_frac_significant": float(np.mean([x["p_one_sided"] < sc.alpha for x in rand])),
           "type_gap_over_random_mean": float(typ["gap"] / gaps.mean()) if gaps.mean() != 0 else None}
    OUT_JSON.write_text(json.dumps(res, indent=2))
    print(f"random partitions: gap mean {gaps.mean():+.4f} sd {gaps.std(ddof=1):.4f}, range {gaps.min():+.4f}..{gaps.max():+.4f}; "
          f"z mean {zs.mean():+.2f}; {res['random_frac_significant']:.0%} significantly selective (one-sided p < {sc.alpha})")
    print(f"type-sorted gap {typ['gap']:+.4f} is at the {pct_below:.1f}th percentile of random-partition gaps "
          f"({pct_below:.1f}% of random partitions have a gap at least as negative); ratio to random mean "
          f"{res['type_gap_over_random_mean']:.2f}")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(gaps, bins=30, color=CATEGORICAL[1], alpha=0.8, edgecolor="white", linewidth=0.4,
            label=f"{gaps.size} random partitions")
    ax.axvline(typ["gap"], color=CATEGORICAL[0], linewidth=2.5, label="type-sorted partition")
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2)
    ax.grid(True, color=GRID, linewidth=0.6, axis="y")
    style(ax, "Real minus shuffled mean input entropy of the 56 voices, per partition of the sensory set",
          "entropy gap (negative = more selective than shuffled)", "partitions")
    save(fig, "partition_control.png")

    if typ["gap"] >= 0:
        verdict = "The type-sorted partition shows no selectivity gap in this ensemble, so there is nothing for random partitions to explain."
    elif pct_below == 0 and gaps.mean() >= 0:
        verdict = ("The type-sorted gap is more extreme than every random-partition gap, and random partitions show NO "
                   f"selectivity at all: their mean gap is {gaps.mean():+.4f}, i.e. with type-blind groups the real input is "
                   "slightly LESS concentrated than shuffled. The structural selectivity is entirely JO TYPE structure.")
    elif pct_below == 0 and gaps.mean() > typ["gap"] / 2:
        verdict = ("The type-sorted gap is larger than every random-partition gap, and random partitions show at most "
                   "half of it on average: the structural selectivity is largely JO TYPE structure.")
    elif pct_below <= 5:
        verdict = ("The type-sorted gap is more extreme than at least 95% of random-partition gaps, but random partitions "
                   "also show a substantial gap: type structure adds to a selectivity that exists without it.")
    else:
        verdict = ("Random partitions show a gap comparable to the type-sorted one: the selectivity is NOT specific to "
                   "type structure.")
    md = ["## Control 2: is the structural selectivity just JO type structure? (no simulation)", "",
          "**Prediction, stated before computing:** if the effect comes from JO type structure, the type-sorted partition "
          "shows a large real-vs-shuffled entropy gap and random partitions show little or none. If random partitions show "
          "a comparable gap, the effect is not about type.", "",
          f"Method: `experiments/partition_control.py`. {sc.partition_n} random partitions of the 86 sensory neurons into "
          f"groups of the type partition's sizes {sizes}; for every partition (and for the type-sorted one) its own ensemble "
          f"of {sc.partition_shuffles} sensory-only re-targetings with distinct seeds. Gap = real minus shuffled mean "
          "normalized entropy over the 56 hop-1 motor neurons.", "",
          "| partition | real entropy | shuffled mean (sd) | gap | z |", "|---|---|---|---|---|",
          f"| type-sorted | {typ['real']:.4f} | {typ['shuffled_mean']:.4f} ({typ['shuffled_sd']:.4f}) | {typ['gap']:+.4f} | {typ['z']:+.2f} |",
          f"| random, mean of {gaps.size} | {np.mean([x['real'] for x in rand]):.4f} | {np.mean([x['shuffled_mean'] for x in rand]):.4f} | "
          f"{gaps.mean():+.4f} (sd {gaps.std(ddof=1):.4f}) | {zs.mean():+.2f} |", "",
          f"Random-partition gaps: min {gaps.min():+.4f}, 5th pct {np.percentile(gaps, 5):+.4f}, median {np.median(gaps):+.4f}, "
          f"95th pct {np.percentile(gaps, 95):+.4f}, max {gaps.max():+.4f}; {res['random_frac_significant']:.0%} of random "
          f"partitions are individually significant (one-sided p < {sc.alpha}).", "",
          f"**The type-sorted gap ({typ['gap']:+.4f}) lies at the {pct_below:.1f}th percentile of the random-partition gaps** "
          f"({pct_below:.1f}% of random partitions are at least as selective); it is {res['type_gap_over_random_mean']:.2f}x "
          "the random-partition mean.", "", "![partition control](../figures/partition_control.png)", "",
          f"**Result:** {verdict}", ""]
    OUT_MD.write_text("\n".join(md))
    print(f"wrote {OUT_JSON.relative_to(ROOT)}, {OUT_MD.relative_to(ROOT)}")
    print("RESULT: " + verdict)


if __name__ == "__main__":
    main()
