"""Export the anatomy set's positions and its per-tick spikes for the web player.

No re-simulation: experiments/audio_run.py already recorded `counts` (ticks x N, spikes per
tick for every neuron) and the cached runs have it. This only projects, subsets and packs.

    .venv/bin/python -m experiments.export_anatomy [--clip STEM] [--no-figure]

docs/anatomy_<clip>.json
  meta    axes, extents, tick_ms, source/layer counts, spike totals
  layers  ["sensory", "motor", "hop1", "hop2", "hop3"]   names for the `layer` codes
  layer   int per neuron, index into `layers`
  pos     [[u...], [v...]]  projection normalized to 0..1 over the set's own extent;
                            u = neuPrint x (left-right), v = neuPrint z (brain -> nerve cord,
                            so v increases DOWN the screen)
  depth   [d...]            neuPrint y (dorso-ventral within brain/VNC), normalized to 0..1
  act     {"offsets": [ticks+1 ints], "idx": [...]}   sparse per-tick activity: the neurons
                            that fired in tick t are idx[offsets[t]:offsets[t+1]], as indices
                            into the neuron arrays above

figures/anatomy_projection.png  the same projection as a static scatter, coloured by layer.

Both are written from cache/positions.parquet (data/fetch_positions.py) and
cache/audio_runs/<clip>.npz (experiments/audio_run.py); neither is re-fetched or re-simulated.
"""
import argparse
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
import pandas as pd

from config import AnatomyConfig, Paths

PATHS = Paths()
CACHE = ROOT / PATHS.cache
DOCS = ROOT / PATHS.docs
FIGURES = ROOT / PATHS.figures
POSITIONS = CACHE / "positions.parquet"
RUNS = CACHE / "audio_runs"

LAYER_STYLE = {                      # draw order: last drawn is on top
    "hop3":    ("#c9d3de", 2.0, 0.55),
    "hop2":    ("#6f9fd8", 2.6, 0.70),
    "hop1":    ("#f0a03c", 4.5, 0.90),
    "motor":   ("#d93b3b", 16.0, 1.00),
    "sensory": ("#2fb37a", 16.0, 1.00),
}


def load_positions(cfg):
    if not POSITIONS.exists():
        raise FileNotFoundError(f"{POSITIONS} missing; run: .venv/bin/python -m data.fetch_positions")
    pos = pd.read_parquet(POSITIONS)
    unknown = set(pos["layer"]) - set(cfg.layers)
    if unknown:
        raise ValueError(f"positions.parquet has layers not in AnatomyConfig.layers: {sorted(unknown)}")
    return pos


def project(pos, cfg):
    """(u, v, depth) in 0..1 over the anatomy set's own extent, plus the extents used."""
    out, extent = {}, {}
    for name, ax in (("u", cfg.proj_horizontal), ("v", cfg.proj_vertical), ("depth", cfg.proj_depth)):
        v = pos[ax].to_numpy(dtype=np.float64)
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            raise ValueError(f"axis {ax} has zero extent ({lo})")
        out[name] = (v - lo) / (hi - lo)
        extent[name] = {"neuprint_axis": ax, "min_voxels": lo, "max_voxels": hi}
    return out, extent


def load_counts(clip, matrix_index):
    """The (ticks, n_set) spike counts of the anatomy set, sliced out of the cached run."""
    npz = RUNS / f"{clip}.npz"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} missing; it is written by experiments/audio_run.py")
    with np.load(npz, allow_pickle=True) as d:
        counts = d["counts"]
        meta = json.loads(str(d["meta"]))
    if counts.ndim != 2:
        raise ValueError(f"{npz}: counts is {counts.shape}, expected (ticks, N)")
    if matrix_index.max() >= counts.shape[1]:
        raise ValueError(f"{npz}: counts has {counts.shape[1]} neurons, "
                         f"anatomy set needs index {int(matrix_index.max())}")
    return counts[:, matrix_index], meta


def sparse_activity(sub):
    """Flat (offsets, idx) CSR-style pair over ticks; a tick's firing neurons are idx[o[t]:o[t+1]]."""
    fired = sub > 0
    per_tick = fired.sum(axis=1)
    offsets = np.concatenate([[0], np.cumsum(per_tick)]).astype(np.int64)
    idx = np.concatenate([np.flatnonzero(row) for row in fired]) if offsets[-1] else np.array([], np.int64)
    if idx.size != offsets[-1]:
        raise ValueError("sparse activity lost spikes while packing")
    return offsets, idx


def export(clip, pos, proj, extent, cfg):
    matrix_index = pos["matrix_index"].to_numpy()
    sub, run_meta = load_counts(clip, matrix_index)
    offsets, idx = sparse_activity(sub)
    layer_code = np.array([cfg.layers.index(l) for l in pos["layer"]], dtype=np.int64)
    r = cfg.pos_decimals

    doc = {
        "clip": clip,
        "meta": {
            "source_run": f"cache/audio_runs/{clip}.npz",
            "source_positions": "cache/positions.parquet",
            "audio_clip": run_meta["clip"],
            "offset_s": run_meta["offset_s"],
            "seconds": run_meta["seconds"],
            "n_ticks": int(sub.shape[0]),
            "tick_ms": run_meta["tick_ms"],
            "n_neurons": int(sub.shape[1]),
            "axes": {"u": f"neuPrint {cfg.proj_horizontal} (left-right)",
                     "v": f"neuPrint {cfg.proj_vertical} (brain -> ventral nerve cord, increases downward)",
                     "depth": f"neuPrint {cfg.proj_depth} (dorso-ventral within brain / VNC)"},
            "extent_voxels": extent,
            "by_layer": {l: int((pos["layer"] == l).sum()) for l in cfg.layers},
            "by_source": {k: int(v) for k, v in pos["source"].value_counts().items()},
            "spikes_total": int(sub.sum()),
            "tick_neuron_pairs": int(offsets[-1]),
            "multi_spike_pairs": int((sub > 1).sum()),
            "mean_active_per_tick": float(offsets[-1] / sub.shape[0]),
        },
        "layers": list(cfg.layers),
        "layer": layer_code.tolist(),
        "pos": [np.round(proj["u"], r).tolist(), np.round(proj["v"], r).tolist()],
        "depth": np.round(proj["depth"], r).tolist(),
        "act": {"offsets": offsets.tolist(), "idx": idx.tolist()},
    }
    DOCS.mkdir(exist_ok=True)
    out = DOCS / f"anatomy_{clip}.json"
    out.write_text(json.dumps(doc, separators=(",", ":")))
    size_mb = out.stat().st_size / 1e6
    print(f"{out.relative_to(ROOT)}: {size_mb:.2f} MB  "
          f"{doc['meta']['n_neurons']:,} neurons, {doc['meta']['n_ticks']} ticks, "
          f"{doc['meta']['tick_neuron_pairs']:,} tick-neuron pairs "
          f"({doc['meta']['mean_active_per_tick']:.1f} active/tick), "
          f"{doc['meta']['spikes_total']:,} spikes "
          f"({doc['meta']['multi_spike_pairs']:,} pairs had >1 spike in the tick)")
    if size_mb > cfg.max_json_mb:
        print(f"  OVER the {cfg.max_json_mb:g} MB budget: cut AnatomyConfig.hop2_sample / hop3_sample "
              f"and re-run data.fetch_positions --force, do NOT drop spikes")
    return doc, size_mb


def plot_projection(pos, cfg, path):
    """The shipped projection plus, for the record, the (x, y) one it was chosen over."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.4))
    panels = ((cfg.proj_horizontal, cfg.proj_vertical, f"shipped: ({cfg.proj_horizontal}, {cfg.proj_vertical})"),
              ("x", "y", "not shipped: (x, y)"))
    for ax, (ha, va, title) in zip(axes, panels):
        for layer, (colour, size, alpha) in LAYER_STYLE.items():
            m = (pos["layer"] == layer).to_numpy()
            ax.scatter(pos.loc[m, ha], pos.loc[m, va], s=size, c=colour, alpha=alpha, lw=0,
                       label=f"{layer} ({int(m.sum())})")
        ax.set_xlabel(f"neuPrint {ha} (voxels)")
        ax.set_ylabel(f"neuPrint {va} (voxels)")
        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_aspect("equal")
    axes[0].legend(loc="lower left", fontsize=8, markerscale=2.5, framealpha=0.9)
    fig.suptitle(f"anatomy set: {len(pos):,} neurons at their real neuPrint positions "
                 f"({int((pos['source'] == 'soma').sum()):,} soma, "
                 f"{int((pos['source'] == 'synapse_centroid').sum()):,} synapse centroid)")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(path, dpi=cfg.figure_dpi)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", action="append", help="run stem (default: AnatomyConfig.clips)")
    ap.add_argument("--no-figure", action="store_true", help="skip figures/anatomy_projection.png")
    args = ap.parse_args()

    cfg = AnatomyConfig()
    pos = load_positions(cfg)
    proj, extent = project(pos, cfg)
    print(f"cache/positions.parquet: {len(pos):,} neurons; "
          + ", ".join(f"{l} {int((pos['layer'] == l).sum())}" for l in cfg.layers))
    for name in ("u", "v", "depth"):
        e = extent[name]
        print(f"  {name} = neuPrint {e['neuprint_axis']}: "
              f"{e['min_voxels']:,.0f} .. {e['max_voxels']:,.0f} voxels -> 0..1")
    for clip in (args.clip or list(cfg.clips)):
        export(clip, pos, proj, extent, cfg)
    if not args.no_figure:
        plot_projection(pos, cfg, FIGURES / "anatomy_projection.png")
