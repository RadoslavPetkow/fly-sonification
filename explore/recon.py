"""Throwaway recon of the neuPrint dataset named in config.py.

Writes everything to explore/recon_report.txt and stdout. Every name used in a
filter below (fields, ROIs, classification values) is first discovered from the
server and asserted to exist.

    .venv/bin/python explore/recon.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import pandas as pd
from neuprint import Client, fetch_roi_hierarchy

from config import NEUPRINT_DATASET, NEUPRINT_SERVER, DataConfig

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 50)

REPORT = ROOT / "explore" / "recon_report.txt"
MECHANO_ROI_BASES = ("AMMC", "WED", "SAD", "AVLP")
CLASSIFICATION_FIELDS = ("superclass", "class", "subclass", "somaNeuromere")
NT_FIELDS = ("predictedNt", "consensusNt", "celltypePredictedNt")

_out = REPORT.open("w")


def emit(*parts):
    text = " ".join(str(p) for p in parts)
    print(text)
    _out.write(text + "\n")


def section(title):
    emit("")
    emit("=" * 78)
    emit(title)
    emit("=" * 78)


def query(c, cypher):
    df = c.fetch_custom(cypher, format="pandas")
    if df.empty:
        raise RuntimeError(f"query returned no rows:\n{cypher}")
    return df


def value_counts(c, field):
    df = query(c, f"MATCH (n:Neuron) RETURN n.`{field}` AS value, count(*) AS n ORDER BY n DESC")
    return df


def overlap_table(sets):
    names = list(sets)
    rows = []
    for a in names:
        rows.append([len(sets[a] & sets[b]) for b in names])
    return pd.DataFrame(rows, index=names, columns=names)


def main():
    c = Client(NEUPRINT_SERVER, dataset=NEUPRINT_DATASET)
    emit(f"server={NEUPRINT_SERVER}  dataset={NEUPRINT_DATASET}  neuprint server version={c.fetch_version()}")

    # ---------------------------------------------------------------- 1
    section("1. Datasets visible to this token")
    datasets = c.fetch_datasets()
    for name, meta in datasets.items():
        emit(f"  {name:22s} uuid={meta.get('uuid')}  last-mod={meta.get('last-mod')}")
    assert NEUPRINT_DATASET in datasets, f"{NEUPRINT_DATASET} not visible; have {list(datasets)}"
    all_rois = set(datasets[NEUPRINT_DATASET]["ROIs"])
    emit(f"  {NEUPRINT_DATASET}: {len(all_rois)} ROIs, "
         f"{len(datasets[NEUPRINT_DATASET]['superLevelROIs'])} superLevelROIs")

    # ---------------------------------------------------------------- 2
    section("2. :Neuron property keys")
    ex = query(c, "MATCH (n:Neuron) WHERE n.type IS NOT NULL AND n.class IS NOT NULL "
                  "RETURN n.bodyId AS bodyId, n.type AS type, keys(n) AS keys LIMIT 1")
    ex_keys = sorted(ex.loc[0, "keys"])
    emit(f"Example neuron bodyId={ex.loc[0, 'bodyId']} type={ex.loc[0, 'type']}: {len(ex_keys)} keys")
    emit("  non-ROI keys:", [k for k in ex_keys if k not in all_rois])
    emit("  ROI-flag keys:", [k for k in ex_keys if k in all_rois])

    emit("")
    emit("Union of non-ROI property keys over ALL :Neuron nodes (key, #neurons having it):")
    keys = query(c, "MATCH (n:Neuron) UNWIND keys(n) AS k RETURN k AS key, count(*) AS n ORDER BY n DESC")
    nonroi = keys[~keys.key.isin(all_rois)].reset_index(drop=True)
    emit(nonroi.to_string())
    present = set(nonroi.key)
    emit(f"  ({len(keys) - len(nonroi)} further keys are per-ROI boolean flags)")

    # ---------------------------------------------------------------- 3
    section("3. ROI hierarchy")
    meta = c.fetch_custom("MATCH (m:Meta) RETURN m.primaryRois AS p").loc[0, "p"]
    primary = set(meta)
    emit(fetch_roi_hierarchy(include_subprimary=False, mark_primary=True, format="text", client=c))
    # format="nx" is broken for this dataset (neuprint-python 0.6.3 hardcodes a
    # 'hemibrain' root), so walk the dict form ourselves.
    tree = fetch_roi_hierarchy(include_subprimary=True, mark_primary=False, format="dict", client=c)
    # Some ROIs (e.g. IB) sit under more than one parent, so keep every path.
    paths, children = {}, {}

    def walk(d, trail):
        for name, sub in d.items():
            paths.setdefault(name, []).append(trail + [name])
            children.setdefault(name, set()).update(sub)
            walk(sub, trail + [name])

    walk(tree, [])
    for base in MECHANO_ROI_BASES:
        matches = sorted(r for r in paths if re.fullmatch(rf"{base}(\(.*\))?", r))
        emit(f"\n{base}: exact spellings in hierarchy = {matches}")
        assert matches, f"no ROI named like {base}"
        for r in matches:
            emit(f"  {r!r:14s} primary={r in primary}")
            for path in paths[r]:
                emit(f"  {'':14s} path: {' > '.join(path)}")
            if children[r]:
                emit(f"  {'':14s} children: {sorted(children[r])}")
    emit("\nAll ROI names (any level) containing AMMC/WED/SAD/AVLP:",
         sorted(r for r in all_rois if re.search("|".join(MECHANO_ROI_BASES), r)))
    emit(f"config.DataConfig.sensory_roi = {DataConfig.sensory_roi!r}; "
         f"exists as ROI name: {DataConfig.sensory_roi in all_rois}")

    # ---------------------------------------------------------------- 4
    section("4. Neuron-level classification fields")
    for f in CLASSIFICATION_FIELDS:
        if f not in present:
            emit(f"\n{f}: NOT PRESENT in this dataset")
            continue
        emit(f"\n{f}:")
        emit(value_counts(c, f).to_string())
    for f in ("status",):
        emit(f"\n{f} (for reference re. neuron counts):")
        emit(value_counts(c, f).to_string())

    # ---------------------------------------------------------------- 5
    section("5. Candidate sensory populations")
    ammc_rois = sorted(r for r in all_rois if re.fullmatch(r"AMMC(\(.*\))?", r))
    assert ammc_rois, "no AMMC ROI"

    jo = query(c, "MATCH (n:Neuron) WHERE n.type =~ '^JO-.*' "
                  "RETURN n.bodyId AS bodyId, n.type AS type, n.superclass AS superclass, "
                  "n.class AS class, n.subclass AS subclass, n.entryNerve AS entryNerve, n.pre AS pre")
    set_a = set(jo.bodyId)

    sc_vals = value_counts(c, "superclass")
    sensory_sc = sorted(v for v in sc_vals.value.dropna() if "sensory" in v)
    cl_vals = value_counts(c, "class")
    sensory_cl = sorted(v for v in cl_vals.value.dropna() if "sensory" in v)
    emit(f"superclass values containing 'sensory': {sensory_sc}")
    emit(f"class      values containing 'sensory': {sensory_cl}")
    b1 = query(c, f"MATCH (n:Neuron) WHERE n.superclass IN {json.dumps(sensory_sc)} RETURN n.bodyId AS bodyId")
    b2 = query(c, f"MATCH (n:Neuron) WHERE n.class IN {json.dumps(sensory_cl)} RETURN n.bodyId AS bodyId")
    set_b1, set_b2 = set(b1.bodyId), set(b2.bodyId)

    flag = " OR ".join(f"n.`{r}`" for r in ammc_rois)
    touch = query(c, f"MATCH (n:Neuron) WHERE {flag} RETURN n.bodyId AS bodyId, n.type AS type, "
                     f"n.superclass AS superclass, n.roiInfo AS roiInfo")
    ammc_pre = {}
    for bid, info in zip(touch.bodyId, touch.roiInfo):
        info = json.loads(info)
        ammc_pre[bid] = sum(info.get(r, {}).get("pre", 0) for r in ammc_rois)
    touch["ammc_pre"] = touch.bodyId.map(ammc_pre)
    set_c = set(touch.loc[touch.ammc_pre > 0, "bodyId"])
    emit(f"AMMC ROIs used for (c): {ammc_rois}; neurons with any synapse flag there: {len(touch)}")

    sets = {
        "a_type^JO-": set_a,
        "b1_superclass~sensory": set_b1,
        "b2_class~sensory": set_b2,
        "c_pre_in_AMMC": set_c,
    }
    emit("\nCounts:")
    for k, v in sets.items():
        emit(f"  {k:24s} {len(v):7d}")
    emit("\nPairwise overlap (diagonal = set size):")
    emit(overlap_table(sets).to_string())
    emit(f"\n  a & b1 & c: {len(set_a & set_b1 & set_c)}   "
         f"a not in c: {len(set_a - set_c)}   a not in b1: {len(set_a - set_b1)}")

    emit("\n(a) per exact type:")
    emit(jo.groupby("type").size().to_string())
    jo["family"] = jo.type.str.extract(r"^(JO-[A-F])", expand=False).fillna(jo.type)
    emit("\n(a) per family (JO-<letter>, else full type):")
    emit(jo.groupby("family").size().to_string())
    emit("\n(a) family x subclass annotation:")
    emit(pd.crosstab(jo.family, jo.subclass.fillna("<null>"), margins=True).to_string())
    emit(f"\n(a) superclass: {dict(Counter(jo.superclass.fillna('<null>')))}")
    emit(f"(a) class:      {dict(Counter(jo['class'].fillna('<null>')))}")
    emit(f"(a) entryNerve: {dict(Counter(jo.entryNerve.fillna('<null>')))}")
    emit(f"(a) JO neurons with pre in AMMC: {len(set_a & set_c)} / {len(set_a)}")
    for pref in DataConfig.auditory_subtypes:
        exact = int((jo.type == pref).sum())
        prefixed = int(jo.type.str.startswith(pref).sum())
        emit(f"config auditory_subtypes {pref!r}: exact type matches={exact}, prefix matches={prefixed}")

    emit("\n(c) neurons with pre in AMMC, by superclass:")
    cpre = touch[touch.ammc_pre > 0]
    emit(cpre.superclass.fillna("<null>").value_counts().to_string())
    emit("\n(c) top 25 types by AMMC presynaptic site count:")
    emit(cpre.groupby(cpre.type.fillna("<null>")).ammc_pre.agg(["size", "sum"])
         .sort_values("sum", ascending=False).head(25).to_string())

    # ---------------------------------------------------------------- 6
    section("6. Candidate descending neurons")
    dn_sc = sorted(v for v in sc_vals.value.dropna() if "descending" in v)
    emit(f"superclass values containing 'descending': {dn_sc}")
    by_ann = query(c, f"MATCH (n:Neuron) WHERE n.superclass IN {json.dumps(dn_sc)} "
                      "RETURN n.bodyId AS bodyId, n.type AS type, n.superclass AS superclass")
    by_re = query(c, "MATCH (n:Neuron) WHERE n.type =~ '^DN.*' "
                     "RETURN n.bodyId AS bodyId, n.type AS type, n.superclass AS superclass")
    emit(f"by superclass annotation: {len(by_ann)}  "
         f"{dict(by_ann.superclass.value_counts())}")
    emit(f"by type regex ^DN:        {len(by_re)}  "
         f"superclasses: {dict(by_re.superclass.fillna('<null>').value_counts())}")
    ann_ids, re_ids = set(by_ann.bodyId), set(by_re.bodyId)
    emit(f"intersection: {len(ann_ids & re_ids)}   agree exactly: {ann_ids == re_ids}")

    only_ann = by_ann[~by_ann.bodyId.isin(re_ids)]
    emit(f"\nannotated descending but type !~ ^DN: {len(only_ann)}")
    emit("  by superclass:", dict(only_ann.superclass.value_counts()))
    emit("  types:")
    emit(only_ann.groupby([only_ann.superclass, only_ann.type.fillna("<null>")]).size()
         .rename("n").to_string())

    only_re = by_re[~by_re.bodyId.isin(ann_ids)]
    emit(f"\ntype ~ ^DN but not annotated descending: {len(only_re)}")
    emit(only_re.groupby([only_re.superclass.fillna("<null>"), only_re.type]).size()
         .rename("n").to_string() if len(only_re) else "  (none)")

    # ---------------------------------------------------------------- 7
    section("7. Neurotransmitter prediction fields")
    for f in NT_FIELDS:
        assert f in present, f"{f} not a neuron property"
        vc = value_counts(c, f)
        emit(f"\n{f}:")
        emit(vc.to_string())
        emit(f"  neurons with no {f}: {int(vc.loc[vc.value.isna(), 'n'].sum())}")
    nt_vals = set(value_counts(c, "predictedNt").value.dropna())
    cfg_nt = set(DataConfig.nt_sign) | set(DataConfig.nt_modulatory)
    emit(f"\nconfig NT names not among predictedNt values: {sorted(cfg_nt - nt_vals)}")
    emit(f"predictedNt values not named in config:         {sorted(nt_vals - cfg_nt)}")
    agree = query(c, "MATCH (n:Neuron) WHERE n.predictedNt IS NOT NULL AND n.consensusNt IS NOT NULL "
                     "RETURN n.predictedNt = n.consensusNt AS same, count(*) AS n")
    emit("predictedNt == consensusNt:")
    emit(agree.to_string())

    # ---------------------------------------------------------------- 8
    section("8. Totals")
    tot = query(c, "MATCH (n:Neuron) RETURN count(n) AS neurons, sum(n.pre) AS pre, sum(n.post) AS post")
    emit(tot.to_string())
    m = query(c, "MATCH (m:Meta) RETURN m.totalPreCount AS totalPreCount, m.totalPostCount AS totalPostCount")
    emit(m.to_string())
    traced = query(c, "MATCH (n:Neuron) WHERE n.status = 'Traced' RETURN count(n) AS traced")
    emit(traced.to_string())

    _out.close()
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
