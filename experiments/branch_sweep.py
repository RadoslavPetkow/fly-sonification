"""How many added synapses does the connectome need before the second hop transmits?

    .venv/bin/python -m experiments.branch_sweep --stage 1|2|3|4|report [--force]

experiments/transmission.py found the 56 MONOsynaptic descending neurons above the circular-shift
null while the 1,057 hop-2 descending neurons sit at chance - even though they are already connected
to the hop-1 population. This measures how much extra wiring that layer needs before its mutual
information with the input leaves the null.

Everything except the graph is held fixed:
  * the simulation parameters come from cache/calibration.json unchanged - w_scale and g_inh are
    NEVER re-tuned inside the sweep (Stage 4 is a separate, clearly-labelled contingency);
  * LAYER MEMBERSHIP is computed ONCE from the UNMODIFIED matrix and reused at every k, so neurons
    cannot move between layers when edges are added;
  * the OU drive seed is fixed, so conditions differ only in the graph.

Reused, not reimplemented: experiments.transmission.simulate / make_signals (drive + simulation),
.build_layers (layers) and .analyze (MI, |r| and the joint circular-shift null); LIFNetwork's
adjacency_path argument; experiments.add_branches for the surgery.

DEVIATION, stated up front: duration_ms is BranchConfig.duration_ms (30 s), half of
TransmissionConfig.duration_ms (60 s), for runtime. The k = 0 condition re-runs the published
experiment at 30 s so the effect of the shortening is itself measured.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from config import AudioConfig, BranchConfig, CalibConfig, MidiConfig, Paths, SimConfig, TransmissionConfig
from data.groups import load_neurons_and_indices, ordered_groups
from experiments.add_branches import branch_path, build, hop_distances, load_original, source_target_sets
from experiments.transmission import CALIBRATED_KEYS, analyze, build_layers, simulate
from sim.calibrate import CATEGORICAL, GRID, TEXT_2, banner, header, rheobase, save, style

CACHE = ROOT / Paths().cache
RESULTS = CACHE / BranchConfig().results_json


class Tee:
    """stdout duplicated into runs/branch_sweep.log."""

    def __init__(self, path):
        self.f = open(path, "a", buffering=1)
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


# =============================================================================
# fixed context: parameters, groups, layer membership
# =============================================================================

class Context:
    """Everything held constant across conditions."""

    def __init__(self, bc):
        self.bc = bc
        self.cc = CalibConfig()
        self.calib = json.loads((CACHE / "calibration.json").read_text())
        self.sc = replace(SimConfig(), **{k: self.calib[k] for k in CALIBRATED_KEYS})
        for k, v in self.calib["sim_config"].items():
            if k not in CALIBRATED_KEYS and getattr(self.sc, k) != v:
                raise ValueError(f"SimConfig.{k} = {getattr(self.sc, k)} differs from the calibration's {v}")
        self.tc = replace(TransmissionConfig(), duration_ms=bc.duration_ms)
        if self.tc.n_shuffles != 20 or self.tc.null_mode != "joint":
            raise ValueError(f"expected the joint null with 20 shuffles, got {self.tc.null_mode!r} "
                             f"/ {self.tc.n_shuffles}")
        neurons, self.indices = load_neurons_and_indices()
        self.neurons = neurons
        self.sens = np.asarray(self.indices["sensory_idx"])
        self.motor = np.asarray(self.indices["motor_idx"])
        self.in_groups = ordered_groups(neurons, self.sens, AudioConfig().n_bins)
        self.motor_groups = ordered_groups(neurons, self.motor, MidiConfig().n_groups)

        # ---- layer membership: computed ONCE from the UNMODIFIED matrix ----
        W, _ = load_original()
        self.dist = hop_distances(W, self.indices)
        self.sources, self.targets = source_target_sets(W, self.indices, self.dist, bc)
        self.nnz_original = int(W.nnz)
        row_abs = np.asarray(abs(W).sum(axis=1)).ravel()
        self.row_abs_targets_before = row_abs[self.targets]
        self.in_degree_targets_before = np.diff(W.indptr)[self.targets]
        # The wiring that ALREADY exists in the block this experiment adds to, measured here rather
        # than quoted: k / this is how many times over the inter-layer wiring has to be rewritten.
        self.interlayer_edges = int(W[self.targets][:, self.sources].nnz)
        self.interlayer_positions = int(self.targets.size * self.sources.size)
        self.adjacency_sha12 = sha12(CACHE / "adjacency.npz")
        del W

        print(f"calibration.json (protocol {self.calib['protocol']}): "
              + ", ".join(f"{k}={self.calib[k]}" for k in CALIBRATED_KEYS)
              + f", i_ext_max={self.calib['i_ext_max']:.6g}")
        print(f"duration {self.tc.duration_ms:g} ms (TransmissionConfig.duration_ms is "
              f"{TransmissionConfig().duration_ms:g} ms - HALVED for runtime), bin {self.tc.bin_ms:g} ms, "
              f"null {self.tc.null_mode!r} x {self.tc.n_shuffles}, drive seed {self.tc.seed}")
        hops = {h: int((self.dist == h).sum()) for h in range(self.tc.max_hop + 1)}
        mhops = {h: int((self.dist[self.motor] == h).sum()) for h in range(self.tc.max_hop + 1)}
        print(f"layer membership from the UNMODIFIED matrix: hops {hops}, motor by hop {mhops}, "
              f"nnz {self.nnz_original:,}")
        print(f"SOURCES {self.sources.size} (hop {bc.source_hop}), TARGETS {self.targets.size} "
              f"(motor at hop {bc.target_hop}) - the primary readout layer {bc.readout_layer!r}")
        print(f"cache/adjacency.npz sha256[:12] {self.adjacency_sha12}; the hop-1 -> hop-2-descending "
              f"block already holds {self.interlayer_edges:,} edges of {self.interlayer_positions:,} "
              f"possible positions ({self.interlayer_edges / self.interlayer_positions:.2%} filled)")


# =============================================================================
# one condition
# =============================================================================

def condition_id(k, policy, seed, targets_frac):
    """One key per condition, encoding k, policy, targets_frac and seed in fixed positions.

    Every field appears in every key (k = 0 is the unmodified matrix, where policy, targets_frac
    and seed do not exist and are spelled out as such), so two different conditions can never
    produce the same key - and therefore never the same simulation-cache path."""
    if k == 0:
        return "k0_none_fnone_seednone"
    f = targets_frac if policy == "concentrated" else 1.0
    return f"k{k}_{policy}_f{f:g}_seed{seed}"


def sim_cache_path(bc, cid):
    """The simulation cache for a condition: its key plus duration_ms, which is not part of the key."""
    return CACHE / bc.branch_subdir / f"sim_{cid}_d{int(round(bc.duration_ms))}.npz"


def sha12(path):
    """First 12 hex characters of the sha256 of the file actually loaded."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def descriptor(ctx, k, policy, seed, targets_frac, adj_path, w_scale, sim_seed=None):
    """The condition's identity, stamped into the simulation cache and re-checked on every load.

    sim_seed (the LIF noise seed) appears only when it is not the calibrated default, so caches
    written before replicates existed still validate; a replicate has its own key and path anyway."""
    d = {"sim_seed": int(sim_seed)} if sim_seed is not None else {}
    return d | {"k": int(k), "policy": policy if k else "none",
            "seed": None if k == 0 else int(seed),
            "targets_frac": (float(targets_frac) if policy == "concentrated" else 1.0) if k else None,
            "duration_ms": float(ctx.bc.duration_ms), "w_scale": float(w_scale),
            "g_inh": float(ctx.sc.g_inh), "normalization": ctx.sc.normalization,
            "adjacency": str(adj_path), "adjacency_sha12": sha12(adj_path)}


def descriptor_mismatch(meta, want):
    """Fields of the cached meta that do not describe the requested condition."""
    bad = {}
    for f, v in want.items():
        got = meta.get(f, "<missing>")
        if isinstance(v, float) and isinstance(got, (int, float)):
            if abs(float(got) - v) > 1e-9 * max(1.0, abs(v)):
                bad[f] = (got, v)
        elif got != v:
            bad[f] = (got, v)
    return bad


def regime_metrics(counts, ctx):
    T = counts.shape[0] * ctx.tc.bin_ms / 1e3
    total = counts.sum(axis=0, dtype=np.int64)
    n = counts.shape[1]
    active = total > 0
    return {"rate_hz": float(total.sum() / n / T),
            "active_frac": float(active.mean()),
            "rate_per_active_hz": float(total.sum() / active.sum() / T) if active.any() else 0.0,
            "sensory_rate_hz": float(total[ctx.sens].mean() / T),
            "motor_rate_hz": float(total[ctx.motor].mean() / T)}


def regime_failures(m, bc):
    lo, hi = bc.regime_rate_hz
    f = []
    if not lo <= m["rate_hz"] <= hi:
        f.append(f"rate_hz {m['rate_hz']:.3g} outside [{lo:g}, {hi:g}]")
    if m["rate_per_active_hz"] > bc.regime_max_rate_per_active_hz:
        f.append(f"rate_per_active_hz {m['rate_per_active_hz']:.3g} > {bc.regime_max_rate_per_active_hz:g}")
    return f


def crossed(layer_result, tc, stat="mi"):
    s = layer_result[stat]
    return bool(s["z"] is not None and s["z"] >= tc.null_z_min and s["mean_best"] > s["null_max"])


def run_condition(ctx, k, policy="spread", seed=None, targets_frac=None, force=False, w_scale=None,
                  tag="", sim_seed=None):
    """Simulate and analyze one graph. w_scale is None everywhere except the Stage 4 contingency.

    sim_seed re-runs an IDENTICAL graph under a different LIF noise realization. That is not a
    condition of the sweep: it is the run-to-run control, which measures how far the statistic moves
    when nothing about the graph or the input changes at all."""
    bc, tc, sc = ctx.bc, ctx.tc, ctx.sc
    if sim_seed is not None:
        sc = replace(sc, seed=int(sim_seed))
        tag = tag + f"_ns{sim_seed}"
    cid = condition_id(k, policy, seed, targets_frac) + tag
    header(f"CONDITION {cid}")
    if k == 0:
        adj, manifest = None, None
        adj_path = CACHE / "adjacency.npz"
    else:
        adj, manifest = build(k, seed, policy, targets_frac, force=False, bc=bc, verbose=True)
        adj_path = adj
    if w_scale is not None:
        sc = replace(sc, w_scale=w_scale)
        print(f"  Stage 4 contingency: w_scale {ctx.sc.w_scale:.6g} -> {w_scale:.6g}")
    sim_cache = sim_cache_path(bc, cid)
    want = descriptor(ctx, k, policy, seed, targets_frac, adj_path, sc.w_scale, sim_seed)
    if sim_seed is not None:
        print(f"  RUN-TO-RUN CONTROL: identical graph, LIF noise seed {sc.seed} instead of "
              f"{ctx.sc.seed}; the drive is unchanged")
    print(f"  adjacency {Path(adj_path).name} sha256[:12] {want['adjacency_sha12']}")

    # The cache is reused ONLY if its stamped descriptor matches this condition in every field.
    force_sim = force
    if sim_cache.exists() and not force:
        cached = json.loads(str(np.load(sim_cache)["meta"]))
        bad = descriptor_mismatch(cached, want)
        if bad:
            force_sim = True
            print(f"  cache {sim_cache.name} does NOT describe this condition, re-running. Mismatched: "
                  + "; ".join(f"{f}: cached {g!r} != requested {w!r}" for f, (g, w) in bad.items()))
        elif cached.get("meta_migrated"):
            print(f"  cache {sim_cache.name} validated (descriptor back-filled by the Sep-2026 cache "
                  f"migration, not stamped at run time)")

    t0 = time.perf_counter()
    counts, drive_binned, z, meta = simulate(sc, ctx.cc, tc, ctx.calib, ctx.in_groups, force_sim,
                                             adjacency_path=adj, sim_cache=sim_cache, extra_meta=want)
    bad = descriptor_mismatch(meta, want)
    if bad:
        raise ValueError(f"simulation {sim_cache} still does not describe this condition: {bad}")
    total_spikes = int(counts.sum(dtype=np.int64))
    print(f"  total spikes {total_spikes:,} over {counts.shape[0]} bins")

    m = regime_metrics(counts, ctx)
    fails = regime_failures(m, bc)
    print(f"  regime: rate {m['rate_hz']:.4g} Hz, active {m['active_frac']:.2%}, per-active "
          f"{m['rate_per_active_hz']:.4g} Hz, sensory {m['sensory_rate_hz']:.4g} Hz, motor "
          f"{m['motor_rate_hz']:.4g} Hz -> " + ("PASS" if not fails else "FAIL: " + "; ".join(fails)))

    rng_layers = np.random.default_rng(tc.seed + 1)
    total = counts.sum(axis=0, dtype=np.int64)
    rates = total / (counts.shape[0] * tc.bin_ms / 1e3)
    layers, layer_info = build_layers(counts, rates, total, ctx.sens, ctx.motor, ctx.dist,
                                      ctx.in_groups, ctx.motor_groups, tc, rng_layers, verbose=True)
    ta = time.perf_counter()
    results, shifts = analyze(counts, drive_binned, layers, tc, np.random.default_rng(tc.seed + 2))
    print(f"  analysis {time.perf_counter() - ta:.1f} s (shifts {shifts.min()}..{shifts.max()} bins)")

    out = {"cid": cid, "k": int(k), "policy": policy if k else "none",
           "seed": None if k == 0 else int(seed),
           "targets_frac": (float(targets_frac) if policy == "concentrated" else 1.0) if k else None,
           "sim_seed": None if sim_seed is None else int(sim_seed),
           "w_scale": float(sc.w_scale), "g_inh": float(sc.g_inh),
           "adjacency": str(adj.relative_to(ROOT)) if adj is not None else "cache/adjacency.npz",
           "adjacency_sha12": want["adjacency_sha12"], "total_spikes": total_spikes,
           "sim_cache": str(sim_cache.relative_to(ROOT)),
           "regime": m, "regime_fail": fails, "layers": results, "layer_info": layer_info,
           "manifest": manifest, "sim_meta": meta, "wall_s": round(time.perf_counter() - t0, 1)}
    r = results[bc.readout_layer]["mi"]
    n_units = results[bc.readout_layer]["n_units"]
    out["readout"] = {"layer": bc.readout_layer, "n_units": n_units, "mi": r["mean_best"],
                      "null_mean": r["null_mean"], "null_sd": r["null_sd"], "null_max": r["null_max"],
                      "z": r["z"], "percentile": r["percentile"],
                      "crossed": crossed(results[bc.readout_layer], tc),
                      "units_above_own_null_max": int(round(r["frac_units_above_own_null_max"] * n_units))}
    for name, label in ((bc.positive_control_layer, "positive_control"),
                        (bc.negative_control_layer, "negative_control")):
        s = results[name]["mi"]
        out[label] = {"layer": name, "n_units": results[name]["n_units"], "mi": s["mean_best"],
                      "null_mean": s["null_mean"], "null_max": s["null_max"], "z": s["z"],
                      "crossed": crossed(results[name], tc),
                      "units_above_own_null_max": int(round(s["frac_units_above_own_null_max"]
                                                            * results[name]["n_units"]))}
    zf = "n/a" if r["z"] is None else f"{r['z']:.2f}"
    print(f"  READOUT {bc.readout_layer}: MI {r['mean_best']:.4g} vs null {r['null_mean']:.4g} "
          f"+- {r['null_sd']:.3g} (max {r['null_max']:.4g}), z {zf}, "
          f"{out['readout']['percentile']:.0f}th pctile, crossed {out['readout']['crossed']}, "
          f"{out['readout']['units_above_own_null_max']}/{n_units} units above their own null max")
    print(f"  positive control ({bc.positive_control_layer}): crossed {out['positive_control']['crossed']}"
          f"; negative control ({bc.negative_control_layer}): crossed {out['negative_control']['crossed']}")
    return out


# =============================================================================
# result store
# =============================================================================

OLD_CID_RE = None


def migrate_store(store, bc):
    """One-off: move pre-hardening conditions onto the collision-proof key and cache path, and
    back-fill the descriptor into their simulation caches so the load-time check has something to
    check. Back-filled descriptors are flagged meta_migrated so they are never mistaken for ones
    stamped at run time; the independent evidence that these runs are distinct is their total spike
    counts, which are computed from the stored counts themselves."""
    moved = []
    for old_cid in list(store["conditions"]):
        c = store["conditions"][old_cid]
        tag = "_recal" if old_cid.endswith("_recal") else ""
        new_cid = condition_id(c["k"], c["policy"] if c["k"] else "spread", c["seed"],
                               c["targets_frac"]) + tag
        if new_cid == old_cid:
            continue
        old_path = CACHE / bc.branch_subdir / f"sim_{old_cid}_d{int(round(bc.duration_ms))}.npz"
        new_path = sim_cache_path(bc, new_cid)
        if old_path.exists() and not new_path.exists():
            d = np.load(old_path)
            meta = json.loads(str(d["meta"]))
            adj_path = ROOT / c["adjacency"]
            meta.update({"k": c["k"], "policy": c["policy"], "seed": c["seed"],
                         "targets_frac": c["targets_frac"], "duration_ms": float(bc.duration_ms),
                         "w_scale": c["w_scale"], "g_inh": c["g_inh"],
                         "normalization": c["sim_meta"].get("normalization", None) or
                         json.loads((CACHE / "calibration.json").read_text())["normalization"],
                         "adjacency": str(adj_path), "adjacency_sha12": sha12(adj_path),
                         "meta_migrated": True})
            np.savez_compressed(new_path, counts=d["counts"], drive_binned=d["drive_binned"],
                                z=d["z"], meta=json.dumps(meta))
            old_path.unlink()
            c.setdefault("adjacency_sha12", meta["adjacency_sha12"])
            c.setdefault("total_spikes", int(np.asarray(d["counts"]).sum(dtype=np.int64)))
            c["sim_cache"] = str(new_path.relative_to(ROOT))
        c["cid"] = new_cid
        store["conditions"][new_cid] = c
        del store["conditions"][old_cid]
        moved.append((old_cid, new_cid))
    if moved:
        print(f"migrated {len(moved)} condition(s) onto the collision-proof cache key:")
        for a, b in moved:
            print(f"  {a}  ->  {b}")
        assert_no_key_collisions(store)
        save_store(store)
    return store


def assert_no_key_collisions(store):
    """Two distinct conditions must never map to one key or one cache path."""
    idents = {}
    for cid, c in store["conditions"].items():
        ident = (c["k"], c["policy"], c["seed"], c["targets_frac"], c.get("sim_seed"),
                 cid.endswith("_recal"))
        # the key must be a FUNCTION of the identity and injective in it, in both directions
        suffix = ""
        if c.get("sim_seed") is not None:
            suffix += f"_ns{c['sim_seed']}"
        if cid.endswith("_recal"):
            suffix += "_recal"
        expect = condition_id(c["k"], c["policy"] if c["k"] else "spread", c["seed"],
                              c["targets_frac"]) + suffix
        if expect != cid:
            raise AssertionError(f"condition {cid} is not the key its own fields produce ({expect})")
        if ident in idents:
            raise AssertionError(f"key collision: {idents[ident]} and {cid} describe the same "
                                 f"condition {ident}")
        idents[ident] = cid
    paths = {}
    for cid, c in store["conditions"].items():
        p = c.get("sim_cache")
        if p is None:
            continue
        if p in paths and paths[p] != cid:
            raise AssertionError(f"cache-path collision: {paths[p]} and {cid} both map to {p}")
        paths[p] = cid


def assert_distinct_spike_totals(store):
    """Two conditions with different k producing exactly the same total spike count would mean one
    simulation was reused for both. That is a bug, not a result: fail loudly."""
    by_total = {}
    for cid, c in store["conditions"].items():
        t = c.get("total_spikes")
        if t is None:
            continue
        by_total.setdefault(t, []).append((cid, c["k"]))
    for t, group in by_total.items():
        ks = {k for _, k in group}
        if len(group) > 1 and len(ks) > 1:
            raise AssertionError(
                f"identical total spike count {t:,} for different k: "
                + ", ".join(f"{cid} (k={k})" for cid, k in group)
                + " - a simulation was reused across conditions")
    return by_total


def load_store():
    if RESULTS.exists():
        return migrate_store(json.loads(RESULTS.read_text()), BranchConfig())
    return {"conditions": {}, "stages": {}}


def save_store(store):
    RESULTS.write_text(json.dumps(store, indent=2, default=str))
    print(f"wrote {RESULTS.relative_to(ROOT)}")


def get_or_run(store, ctx, k, policy="spread", seed=None, targets_frac=None, force=False, **kw):
    cid = condition_id(k, policy, seed, targets_frac) + kw.get("tag", "")
    if cid in store["conditions"] and not force:
        c = store["conditions"][cid]
        now = sha12(ROOT / c["adjacency"])
        if c.get("adjacency_sha12") != now:
            print(f"[stale] {cid}: {c['adjacency']} now hashes {now}, recorded "
                  f"{c.get('adjacency_sha12')}; re-running")
        else:
            r = c["readout"]
            zf = "n/a" if r["z"] is None else f"{r['z']:.2f}"
            print(f"[cached] {cid}: sha {now}, {c.get('total_spikes', 0):,} spikes, hop-2 MI "
                  f"{r['mi']:.4g}, z {zf}, crossed {r['crossed']}, "
                  f"regime {'FAIL' if c['regime_fail'] else 'pass'}")
            return c
    c = run_condition(ctx, k, policy, seed, targets_frac, force=force, **kw)
    store["conditions"][cid] = c
    assert_no_key_collisions(store)
    assert_distinct_spike_totals(store)
    save_store(store)
    return c


# =============================================================================
# stages
# =============================================================================

def stage1(store, ctx, force=False):
    bc = ctx.bc
    header("STAGE 1 - coarse bracket, one seed, policy spread")
    print(f"k in {list(bc.stage1_k)}, seed {bc.stage1_seed}")
    run = []
    stopped = None
    for k in bc.stage1_k:
        c = get_or_run(store, ctx, k, "spread", bc.stage1_seed, force=force)
        run.append(c)
        if k == 0:
            r = c["readout"]
            if r["crossed"]:
                banner("STOP: the k = 0 control does NOT reproduce the published result - the hop-2 "
                       "descending layer is ABOVE the null at 30 s with no edges added. Nothing further "
                       "is measurable until that is explained.")
                store["stages"]["1"] = {"k": [0], "aborted": "k=0 control crossed", "seed": bc.stage1_seed}
                save_store(store)
                return run, None
            if not c["positive_control"]["crossed"]:
                banner("STOP: the k = 0 positive control (motor at hop 1) is NOT above the null at 30 s; "
                       "the published result does not reproduce at this duration.")
                store["stages"]["1"] = {"k": [0], "aborted": "k=0 positive control failed",
                                        "seed": bc.stage1_seed}
                save_store(store)
                return run, None
            print("k = 0 reproduces the published result: hop-2 descending at chance, hop-1 descending "
                  "above the null.")
        if c["regime_fail"]:
            stopped = k
            banner(f"REGIME FAIL at k = {k:,}: " + "; ".join(c["regime_fail"])
                   + " - ascending sweep stopped here. w_scale is NOT changed.")
            break
    bracket = find_bracket(run, ctx)
    store["stages"]["1"] = {"k": [c["k"] for c in run], "seed": bc.stage1_seed,
                            "regime_fail_at": stopped, "bracket": bracket}
    save_store(store)
    return run, bracket


def find_bracket(conds, ctx):
    """(largest k at chance and in regime, smallest k that crossed) or (·, None) if nothing crossed."""
    ok = [c for c in conds if not c["regime_fail"]]
    crossing = sorted([c["k"] for c in ok if c["readout"]["crossed"]])
    chance = sorted([c["k"] for c in ok if not c["readout"]["crossed"]])
    if crossing:
        lo = max([k for k in chance if k < crossing[0]], default=0)
        return {"lo": lo, "hi": crossing[0], "crossed": True}
    return {"lo": max(chance, default=0), "hi": None, "crossed": False,
            "largest_in_regime": max([c["k"] for c in ok], default=0)}


def half_decade_grid(lo, hi):
    """Half-decade steps strictly inside (lo, hi], endpoints included where they are positive."""
    if hi is None:
        return []
    start = math.log10(lo) if lo > 0 else math.log10(max(hi / 100.0, 1.0))
    stop = math.log10(hi)
    n = max(int(round((stop - start) / 0.5)), 1)
    ks = sorted({int(round(10 ** (start + 0.5 * i))) for i in range(n + 1)} | ({lo} if lo > 0 else set()) | {hi})
    return [k for k in ks if k > 0]


def stage2(store, ctx, force=False):
    bc = ctx.bc
    st1 = store["stages"].get("1")
    if not st1 or st1.get("aborted"):
        raise RuntimeError("Stage 1 has not completed successfully; run --stage 1 first")
    br = st1["bracket"]
    header("STAGE 2 - refine in half-decade steps, three seeds")
    if br["crossed"]:
        ks = half_decade_grid(br["lo"], br["hi"])
        print(f"Stage 1 bracket: at chance up to k = {br['lo']:,}, crossed at k = {br['hi']:,}; "
              f"half-decade grid {ks}")
    else:
        hi = br["largest_in_regime"]
        lo = max([k for k in bc.stage1_k if 0 < k < hi], default=max(hi // 10, 1))
        ks = half_decade_grid(lo, hi)
        print(f"Stage 1 found NO crossing; largest in-regime k = {hi:,}. Refining the top decade "
              f"{lo:,}..{hi:,}: {ks}")
    rows = []
    for k in ks:
        for seed in bc.seeds:
            c = get_or_run(store, ctx, k, "spread", seed, force=force)
            rows.append(c)
            if c["regime_fail"]:
                print(f"  (regime FAIL at k={k:,} seed={seed}; recorded, larger k not attempted)")
    store["stages"]["2"] = {"k": ks, "seeds": list(bc.seeds),
                            "summary": summarize_by_k(rows, ctx)}
    save_store(store)
    return rows


def summarize_by_k(rows, ctx):
    out = {}
    for c in rows:
        key = str(c["k"])
        out.setdefault(key, {"z": [], "mi": [], "crossed": [], "rate_hz": [], "seeds": [],
                             "units_above": []})
        o = out[key]
        o["z"].append(c["readout"]["z"])
        o["mi"].append(c["readout"]["mi"])
        o["crossed"].append(c["readout"]["crossed"])
        o["rate_hz"].append(c["regime"]["rate_hz"])
        o["units_above"].append(c["readout"]["units_above_own_null_max"])
        o["seeds"].append(c["seed"])
    for key, o in out.items():
        zs = [v for v in o["z"] if v is not None]
        o["z_mean"] = float(np.mean(zs)) if zs else None
        o["z_sd"] = float(np.std(zs, ddof=1)) if len(zs) > 1 else 0.0
        o["z_min"], o["z_max"] = (float(np.min(zs)), float(np.max(zs))) if zs else (None, None)
        o["mi_mean"] = float(np.mean(o["mi"]))
        o["n_crossed"] = int(sum(o["crossed"]))
    return out


def stage3(store, ctx, force=False):
    bc = ctx.bc
    st2 = store["stages"].get("2")
    if not st2:
        raise RuntimeError("Stage 2 has not run; run --stage 2 first")
    header("STAGE 3 - concentration arm")
    summ = st2["summary"]
    crossing_ks = sorted(int(k) for k, o in summ.items() if o["n_crossed"] > 0)
    if crossing_ks:
        k = crossing_ks[0]
        why = f"smallest k that crossed in Stage 2 ({summ[str(k)]['n_crossed']}/{len(bc.seeds)} seeds)"
    else:
        in_regime = [int(k) for k, o in summ.items() if all(r < bc.regime_rate_hz[1] for r in o["rate_hz"])]
        k = max(in_regime) if in_regime else max(int(x) for x in summ)
        why = "nothing crossed in Stage 2; using the largest in-regime k"
    print(f"k = {k:,} ({why}); policy concentrated at targets_frac {list(bc.concentrated_fracs)}, "
          f"seeds {list(bc.seeds)}")
    rows = []
    for f in bc.concentrated_fracs:
        n_sub = math.ceil(f * ctx.targets.size)
        if k > n_sub * (ctx.sources.size - 5):
            print(f"  SKIPPED targets_frac {f:g}: k = {k:,} exceeds the "
                  f"{n_sub * (ctx.sources.size - 5):,} positions available to {n_sub} targets")
            continue
        for seed in bc.seeds:
            rows.append(get_or_run(store, ctx, k, "concentrated", seed, targets_frac=f, force=force))
    store["stages"]["3"] = {"k": k, "reason": why, "fracs": list(bc.concentrated_fracs),
                            "seeds": list(bc.seeds),
                            "summary": {f"f{c['targets_frac']:g}": None for c in rows}}
    by_f = {}
    for c in rows:
        by_f.setdefault(f"f{c['targets_frac']:g}", []).append(c)
    store["stages"]["3"]["summary"] = {
        key: {"z_mean": float(np.mean([c["readout"]["z"] for c in cs])),
              "mi_mean": float(np.mean([c["readout"]["mi"] for c in cs])),
              "n_crossed": int(sum(c["readout"]["crossed"] for c in cs)),
              "units_above_mean": float(np.mean([c["readout"]["units_above_own_null_max"] for c in cs])),
              "units_above": [c["readout"]["units_above_own_null_max"] for c in cs],
              "seeds": [c["seed"] for c in cs]}
        for key, cs in by_f.items()}
    save_store(store)
    return rows


def stage4(store, ctx, force=False):
    """ONLY if a regime FAIL occurred: re-derive w_scale from the anchor formula in sim/calibrate.py
    against the MODIFIED matrix, and re-run that single condition. Two changed variables at once."""
    from sim.normalize import normalize

    bc = ctx.bc
    failed = [c for c in store["conditions"].values() if c["regime_fail"]]
    header("STAGE 4 - recalibration contingency")
    if not failed:
        print("no condition failed the regime check; Stage 4 does not apply.")
        store["stages"]["4"] = {"applies": False}
        save_store(store)
        return []
    failed.sort(key=lambda c: c["k"])
    c0 = failed[0]
    print(f"failing condition {c0['cid']} (k = {c0['k']:,}): " + "; ".join(c0["regime_fail"]))
    W = sp.load_npz(ROOT / c0["adjacency"]).tocsr()
    Wn, _ = normalize(W, ctx.sc.normalization)
    has_in = np.diff(W.indptr) > 0
    S = float(np.median(np.asarray(Wn.sum(axis=1)).ravel()[has_in]))
    geom = 1.0 / (1.0 - math.exp(-ctx.sc.dt_ms / ctx.sc.tau_syn_ms))
    del W, Wn
    W0 = sp.load_npz(CACHE / "adjacency.npz").tocsr()
    Wn0, _ = normalize(W0, ctx.sc.normalization)
    S0 = float(np.median(np.asarray(Wn0.sum(axis=1)).ravel()[np.diff(W0.indptr) > 0]))
    del W0, Wn0
    w_new = ctx.sc.w_scale * S0 / S
    print(f"anchor formula w_anchor(f) = 1000 / (geom * S * f * dt), geom {geom:.4g}: median signed row "
          f"sum S {S0:.6g} (original) -> {S:.6g} (modified); the calibrated w_scale "
          f"{ctx.sc.w_scale:.6g} scales by S0/S = {S0 / S:.6g} to {w_new:.6g}")
    c = get_or_run(store, ctx, c0["k"], c0["policy"], c0["seed"],
                   targets_frac=c0["targets_frac"] if c0["policy"] == "concentrated" else None,
                   force=force, w_scale=w_new, tag="_recal")
    store["stages"]["4"] = {"applies": True, "failing_cid": c0["cid"], "k": c0["k"],
                            "S_original": S0, "S_modified": S, "w_scale_original": ctx.sc.w_scale,
                            "w_scale_recalibrated": w_new, "recalibrated_cid": c["cid"]}
    save_store(store)
    banner("STAGE 4 CARRIES TWO CHANGED VARIABLES AT ONCE (graph AND w_scale). It is context for the "
           "main result, not part of it.")
    return [c]


# =============================================================================
# report
# =============================================================================

def report(store, ctx):
    bc = ctx.bc
    conds = sorted(store["conditions"].values(), key=lambda c: (c["policy"], c["targets_frac"] or 1.0,
                                                               c["k"], c["seed"] or 0))
    totals = assert_distinct_spike_totals(store)
    assert_no_key_collisions(store)
    header("PROVENANCE - the matrix actually loaded and the run it produced")
    print(f"  every condition below re-checked: its simulation cache carries the condition's own "
          f"descriptor (k, policy, targets_frac, seed, duration_ms, w_scale, g_inh, normalization, "
          f"adjacency path and hash) and is re-run on any mismatch; no two conditions share a cache "
          f"path; no two different k share a spike total ({len(totals)} distinct totals over "
          f"{len(conds)} conditions).")
    print(f"  {'k':>7} {'policy':>13} {'seed':>4} {'adjacency file':>34} {'sha256[:12]':>13} "
          f"{'total spikes':>13}")
    for c in conds:
        pol = c["policy"] + (f" f{c['targets_frac']:g}" if c["policy"] == "concentrated" else "")
        print(f"  {c['k']:>7,} {pol:>13} {str(c['seed']):>4} {Path(c['adjacency']).name:>34} "
              f"{str(c.get('adjacency_sha12')):>13} {c.get('total_spikes', 0):>13,}")

    header("RESULTS - hop-2 descending layer (the primary readout), MI vs the joint circular-shift null")
    print(f"  k / {ctx.interlayer_edges:,} = k as a multiple of the edges the hop-1 -> hop-2-descending "
          f"block ALREADY has ({ctx.interlayer_edges:,} of {ctx.interlayer_positions:,} positions, "
          f"{ctx.interlayer_edges / ctx.interlayer_positions:.2%} filled).")
    print(f"  {'k':>7} {'k/inter':>8} {'policy':>13} {'seed':>4} {'sha':>13} {'units':>5} {'MI':>8} "
          f"{'null mean':>9} {'null sd':>8} {'z':>7} {'pct':>4} {'cross':>5} | {'rate':>6} {'act':>6} "
          f"{'per-act':>7} {'sens':>6} {'motor':>6} {'regime':>6}")
    for c in conds:
        r, m = c["readout"], c["regime"]
        zf = "n/a" if r["z"] is None else f"{r['z']:.2f}"
        pol = c["policy"] + (f" f{c['targets_frac']:g}" if c["policy"] == "concentrated" else "")
        print(f"  {c['k']:>7,} {c['k'] / ctx.interlayer_edges:>8.3g} {pol:>13} {str(c['seed']):>4} "
              f"{str(c.get('adjacency_sha12')):>13} {r['n_units']:>5} {r['mi']:>8.4g} "
              f"{r['null_mean']:>9.4g} {r['null_sd']:>8.3g} {zf:>7} {r['percentile']:>3.0f}% "
              f"{'YES' if r['crossed'] else 'no':>5} | {m['rate_hz']:>6.3g} {m['active_frac']:>5.1%} "
              f"{m['rate_per_active_hz']:>7.3g} {m['sensory_rate_hz']:>6.3g} {m['motor_rate_hz']:>6.3g} "
              f"{'FAIL' if c['regime_fail'] else 'pass':>6}")

    header("CONTROLS")
    print(f"  {'k':>7} {'policy':>13} {'seed':>4} | positive control ({bc.positive_control_layer}) "
          f"| negative control ({bc.negative_control_layer})")
    for c in conds:
        p, n = c["positive_control"], c["negative_control"]
        zp = "n/a" if p["z"] is None else f"{p['z']:.2f}"
        zn = "n/a" if n["z"] is None else f"{n['z']:.2f}"
        pol = c["policy"] + (f" f{c['targets_frac']:g}" if c["policy"] == "concentrated" else "")
        print(f"  {c['k']:>7,} {pol:>13} {str(c['seed']):>4} | MI {p['mi']:.4g} z {zp:>7} "
              f"above-null {str(p['crossed']):>5} ({p['units_above_own_null_max']}/{p['n_units']} units) "
              f"| MI {n['mi']:.4g} z {zn:>7} above-null {str(n['crossed']):>5} "
              f"({n['units_above_own_null_max']}/{n['n_units']} units)")

    header("ROW SCALING OF THE TARGETS (sqrt_in divides row i by sqrt(sum_j |W[i,j]|))")
    print(f"  original: median row |W| sum over the {ctx.targets.size} targets "
          f"{np.median(ctx.row_abs_targets_before):g}, median in-degree "
          f"{np.median(ctx.in_degree_targets_before):g}")
    for c in conds:
        if c["manifest"] is None:
            continue
        r = c["manifest"]["row_abs_sum_targets"]
        d = c["manifest"]["in_degree_change"]
        pol = c["policy"] + (f" f{c['targets_frac']:g}" if c["policy"] == "concentrated" else "")
        print(f"  k {c['k']:>7,} {pol:>13} seed {c['seed']}: row |W| sum median {r['median_before']:g} -> "
              f"{r['median_after']:g} ({r['median_after'] / r['median_before'] - 1:+.2%}), existing input "
              f"shrunk by x{(r['sqrt_in_scale_median_after'] / r['sqrt_in_scale_median_before']):.4g}; "
              f"in-degree median {d['in_degree_median_before']:g} -> {d['in_degree_median_after']:g}, "
              f"{d['targets_touched']}/{ctx.targets.size} targets touched, +{d['added_per_target_median']:g} "
              f"edges/target (max +{d['added_per_target_max']})")

    header("ANSWER")
    spread = [c for c in conds if c["policy"] == "spread" and not c["regime_fail"]]
    crossing = sorted([c for c in spread if c["readout"]["crossed"]], key=lambda c: c["k"])
    lines = []
    if crossing:
        k = crossing[0]["k"]
        same_k = [c for c in spread if c["k"] == k]
        n_cross = sum(c["readout"]["crossed"] for c in same_k)
        zs = ", ".join("n/a" if c["readout"]["z"] is None else f"{c['readout']['z']:.2f}" for c in same_k)
        beat = ", ".join(f"{c['readout']['percentile'] / 100 * ctx.tc.n_shuffles:.0f}/{ctx.tc.n_shuffles}"
                         for c in same_k)
        lines.append(f"The hop-2 descending layer leaves the null at k = {k:,} added synapses "
                     f"({n_cross}/{len(same_k)} seeds), z = {zs}, beating {beat} shuffles.")
        lines.append(f"HEADLINE: that is {k / ctx.interlayer_edges:.3g}x the {ctx.interlayer_edges:,} "
                     f"edges the hop-1 -> hop-2-descending block already has - the inter-layer wiring "
                     f"has to be rewritten {k / ctx.interlayer_edges:.3g} times over.")
        lines.append(f"It is also {k / ctx.nnz_original:.3%} of the {ctx.nnz_original:,} edges in the "
                     f"whole matrix, and {k / ctx.targets.size:.2f} added edges per hop-2 descending "
                     f"neuron.")
    else:
        in_regime = [c["k"] for c in spread]
        big = max(in_regime) if in_regime else 0
        lines.append("NO tested k crossed: the hop-2 descending layer stayed inside the circular-shift "
                     "null at every k that kept the network in its usable regime.")
        lines.append(f"HEADLINE: the largest in-regime k tested, {big:,}, is already "
                     f"{big / ctx.interlayer_edges:.3g}x the {ctx.interlayer_edges:,} edges the "
                     f"hop-1 -> hop-2-descending block has - that much rewriting of the inter-layer "
                     f"wiring still does not move the layer out of the null.")
        lines.append(f"It is also {big / ctx.nnz_original:.3%} of the {ctx.nnz_original:,} edges in the "
                     f"whole matrix, and {big / ctx.targets.size:.2f} added edges per hop-2 descending "
                     f"neuron.")
        failed = sorted([c for c in conds if c["regime_fail"]], key=lambda c: c["k"])
        if failed:
            lines.append(f"The network left its usable regime at k = {failed[0]['k']:,} "
                         f"({'; '.join(failed[0]['regime_fail'])}).")
    pos_ok = all(c["positive_control"]["crossed"] for c in conds)
    neg = [c for c in conds if c["negative_control"]["crossed"]]
    lines.append(f"Positive control (motor at hop 1, 56 neurons): above the null in "
                 f"{sum(c['positive_control']['crossed'] for c in conds)}/{len(conds)} conditions"
                 + ("." if pos_ok else " - NOT throughout, which weakens every comparison."))
    lines.append(f"Negative control (motor at hop 3, 209 neurons): above the null in {len(neg)}/{len(conds)} "
                 f"conditions.")
    for ln in lines:
        print("  " + ln)

    header("KNOWN LIMITATIONS")
    lim = [
        (f"5 of the {ctx.sources.size} hop-1 neurons were excluded from the presynaptic draw, so edges "
         f"were sampled from {ctx.sources.size - 5} sources, not {ctx.sources.size}. They are matrix "
         f"indices 41, 216, 256, 740, 1041 (types 5-HTPMPV03 x2, DNg30, OA-VUMa4, AVLP610). Their "
         f"presynaptic column in the signed matrix is empty, so the sign rule - inherit the sign of the "
         f"column's first stored value - has nothing to inherit."),
        ("The cause is NOT the >= 2 synapse threshold: in cache/edges_raw.parquet these five carry "
         "1,271 to 10,144 outgoing edges with weights up to 209. Their consensusNt in "
         "cache/neurons.parquet is serotonin (x2), serotonin, octopamine and dopamine - all listed in "
         "DataConfig.nt_modulatory, which the matrix build deliberately silences (sign = 0, edges "
         "diverted to cache/modulatory_edges.parquet) for all 541 modulatory neurons in the dataset."),
        ("So consensusNt does give these five a transmitter, but not a SIGN: assigning them +1 or -1 "
         "would not be recovering information already in the build, it would be reversing the project's "
         "modulatory-exclusion decision for the whole connectome. That is a separate experiment, and it "
         "would have to change every modulatory neuron, not only these five."),
        (f"Effect on this measurement: the excluded five contribute 0 of the {ctx.interlayer_edges:,} "
         f"edges the block already has (empty columns), so the k / inter-layer ratio is unaffected; the "
         f"added edges are drawn from a source pool 0.75% smaller than the nominal hop-1 layer."),
        (f"duration_ms is {ctx.bc.duration_ms / 1e3:g} s, half of TransmissionConfig.duration_ms "
         f"({TransmissionConfig().duration_ms / 1e3:g} s), for runtime; the k = 0 condition measures "
         f"what that shortening costs."),
        ("DOSE AND THE INTERNAL CONTROL: at k = 100 only 94 of the 1,057 descending hop-2 targets "
         "received an added edge, so the 963 untouched ones served as a control group INSIDE the "
         "same simulation. At k = 10,000 every one of the 1,057 targets receives at least one edge "
         "(manifest: targets_touched 1,057/1,057, +1..24 each), so that within-simulation "
         "touched/untouched contrast does not exist at this dose - there are no untouched cells "
         "left to compare against."),
        ("That contrast is therefore REPLACED, not abandoned, by external control arms carrying an "
         "identical edge count, sign split and weight multiset placed elsewhere: arm C (displaced) "
         "moves both sources and targets to hop-3 non-motor neurons, and arm D (wrong_targets) "
         "keeps the real hop-1 sources and moves only the targets to non-descending hop-2 neurons. "
         "The k = 100 touched/untouched check WAS run and is reported in cache/branch_checks.json: "
         "it found cells that received no edges moving as far as cells that did, which is why the "
         "ladder's headline was withdrawn rather than refined."),
        ("A layer's statistic is the mean over its units with >= TransmissionConfig.min_spikes spikes, "
         "so the ANALYZED subset of a fixed layer grows with k even though layer MEMBERSHIP is fixed "
         "from the unmodified matrix; the units column of the results table reports it at every k."),
    ]
    for ln in lim:
        print("  * " + ln)
    store["answer"] = lines
    store["limitations"] = lim
    figure(store, ctx)
    save_store(store)


def figure(store, ctx):
    bc = ctx.bc
    conds = list(store["conditions"].values())
    fig, ax = plt.subplots(figsize=(8.5, 5))
    styles = {"spread": (CATEGORICAL[0], "o", "spread (uniform over all 1,057 targets)"),
              "concentrated": (CATEGORICAL[1], "s", "concentrated (subset of targets)")}
    for policy, (color, marker, label) in styles.items():
        pts = sorted([c for c in conds if c["policy"] == policy or (policy == "spread" and c["k"] == 0)],
                     key=lambda c: c["k"])
        if not pts:
            continue
        by_k = {}
        for c in pts:
            if c["readout"]["z"] is not None:
                by_k.setdefault(c["k"], []).append(c["readout"]["z"])
        if not by_k:
            continue
        ks = sorted(by_k)
        mean = [float(np.mean(by_k[k])) for k in ks]
        lo = [float(np.min(by_k[k])) for k in ks]
        hi = [float(np.max(by_k[k])) for k in ks]
        ax.errorbar(ks, mean, yerr=[np.array(mean) - lo, np.array(hi) - np.array(mean)], color=color,
                    marker=marker, ms=6, linewidth=1.8, capsize=3, label=label)
        for c in pts:
            if c["regime_fail"] and c["readout"]["z"] is not None:
                ax.plot([c["k"]], [c["readout"]["z"]], "x", color="#b00020", ms=11, mew=2,
                        label="_regime FAIL")
    ax.axhspan(-3, 3, color=GRID, alpha=0.9, zorder=0,
               label=f"null band (|z| < {ctx.tc.null_z_min:g})")
    ax.axhline(ctx.tc.null_z_min, color=TEXT_2, linestyle="--", linewidth=1.1,
               label=f"crossing threshold z = {ctx.tc.null_z_min:g}")
    ax.set_xscale("symlog", linthresh=100)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2, loc="upper left")
    ax.grid(True, color=GRID, linewidth=0.6, axis="y")
    style(ax, f"Hop-2 descending layer: MI z-score vs added synapses ({bc.duration_ms / 1e3:g} s, "
              f"{ctx.tc.n_shuffles} joint circular shifts)",
          "added synapses k (hop-1 -> hop-2 descending); 0 = unmodified connectome",
          "z of layer mean MI vs the null")
    save(fig, bc.figure)


# =============================================================================
# main
# =============================================================================

def main(stage, force=False):
    bc = BranchConfig()
    (ROOT / Paths().runs).mkdir(exist_ok=True)
    tee = Tee(ROOT / Paths().runs / bc.log)
    sys.stdout = tee
    try:
        print(f"\n{'=' * 100}\nbranch_sweep stage {stage} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 100}")
        header("SETUP")
        ctx = Context(bc)
        store = load_store()
        if stage in ("1", "all"):
            stage1(store, ctx, force)
        if stage in ("2", "all"):
            stage2(store, ctx, force)
        if stage in ("3", "all"):
            stage3(store, ctx, force)
        if stage in ("4", "all"):
            stage4(store, ctx, force)
        report(store, ctx)
    finally:
        sys.stdout = tee.out
        tee.f.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["1", "2", "3", "4", "all", "report"], default="1")
    ap.add_argument("--force", action="store_true", help="re-simulate even if a condition is cached")
    a = ap.parse_args()
    main(a.stage, a.force)
