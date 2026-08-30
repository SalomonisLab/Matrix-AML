#!/usr/bin/env python3
"""Per-cell-state violins for EVERY modality, not just RNA.

The evidence page's violin panel was built from the RNA pseudobulk matrix alone, so it could only draw
a molecule that exists as a gene symbol. That left ADT, Lipid and Metabolite entirely un-plottable and
GRN nearly so, because those features are not RNA species -- ADT markers are antibody names (GPR56 is
the gene ADGRG1, CD73 is NT5E), Lipid markers are species such as "TG O-54:1", and Metabolite markers
are unannotated peaks named "Unknown0 932". The page rendered twelve disabled buttons per modality,
which read as missing data. The data is not missing: each modality has its own pseudobulk matrix, and
it is the matrix its classifier was trained on.

This builds the same {molecule: {cell_state: {mutant:[...], control:[...]}}} structure from each
modality's OWN matrix, in that modality's own units.

TWO THINGS IT RECORDS THAT THE RNA VIOLINS NEVER HAD TO:

  units       every modality is on a different scale, so the y-axis label must come from the data, not
              be hardcoded to "CP10k+log1p" as the RNA panel does.
  fidelity    ADT / GRN / Lipid / Metabolite are IMPUTED FROM RNA. `feature_fidelity` gives each
              feature's held-out Spearman against its measured counterpart, so a reader can weigh an
              imputed molecule instead of taking it at face value. A platform-level result already
              says these blocks add nothing beyond a matched random control, so showing the per-feature
              fidelity next to the violin is the minimum honest framing.

  bsub -q test -W 4:00 -M 48000 -o mv.log /usr/local/anaconda3-2020/bin/python build_modality_violins.py
  -> gui/evidence_modalities.json
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

from amlmm.context import build_context, Config
from amlmm import pseudobulk_io as pio
from amlmm import discovery as DSC

EVPATH = os.path.join(HERE, "..", "gui", "evidence.json")
OUTPATH = os.path.join(HERE, "..", "gui", "evidence_modalities.json")

MODALITIES = ["ADT", "GRN", "Lipid", "Metabolite"]      # RNA already has violins in evidence.json
UNITS = {"ADT": "imputed ADT (CLR)", "GRN": "imputed regulon activity",
         "Lipid": "imputed lipid abundance (log)", "Metabolite": "imputed metabolite intensity (log)"}
MAX_STATES = 10                                          # keep the payload comparable to the RNA panel


def main():
    t0 = time.time()
    EV = json.load(open(EVPATH, encoding="utf-8"))["drivers"]
    ctx = build_context(Config())
    pb = ctx.tables["pseudobulks"]

    have = [m for m in MODALITIES if m in getattr(ctx, "_modality_paths", {})]
    print("modalities available: %s (of %s)" % (have, MODALITIES), flush=True)
    if not have:
        print("no imputed modality matrices in this context -- nothing to build")
        return

    fid = {}
    for m in have:
        try:
            f = pio.feature_fidelity(ctx, m)
            # NaN must not reach the JSON: json.dump writes a bare NaN token, which Python reads back
            # happily and every browser rejects as invalid JSON. The first build shipped that and the
            # page silently fell back to its RNA-only path via the fetch .catch().
            fid[m] = {} if f is None else {str(k): round(float(v), 4) for k, v in f.items()
                                           if v == v and np.isfinite(float(v))}
            print("  %-11s fidelity for %d features" % (m, len(fid[m])), flush=True)
        except Exception as ex:
            fid[m] = {}
            print("  %-11s fidelity unavailable: %s" % (m, str(ex)[:70]), flush=True)

    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "units": UNITS,
           "note": ("Per-cell-state values from each modality's OWN pseudobulk matrix -- the matrix its "
                    "classifier was trained on. All four are IMPUTED FROM RNA, so these are model "
                    "estimates rather than measurements; `fidelity` is each feature's held-out Spearman "
                    "against its measured counterpart."),
           "fidelity": fid, "drivers": {}}

    for driver, dd in EV.items():
        markers = dd.get("markers") or {}
        states = list((dd.get("violins") or {}).values())
        states = list(states[0].keys())[:MAX_STATES] if states else []
        if not states:
            continue
        # Mutant / control split from the same label source the classifiers used. evidence.json keys
        # are bare driver names ('NPM1', 'FLT3-ITD', 'inv(16)_CBFB-MYH11') while labels_for_field wants
        # the mutation-matrix column, which is prefixed mut_/cyto_. The first run returned zero drivers
        # for exactly this reason, so try the plausible spellings rather than assume one.
        lab = None
        base = str(driver)
        cands = [base, "mut_" + base, "cyto_" + base,
                 "mut_" + base.replace("-", "_"), "cyto_" + base.replace("-", "_"),
                 base.split("_")[0], "cyto_" + base.split("_")[0]]
        seen = set()
        for c in cands:
            if c in seen:
                continue
            seen.add(c)
            try:
                l = DSC.labels_for_field(ctx, c)
            except Exception:
                continue
            if l is not None and len(l.dropna()):
                lab, used_field = l, c
                break
        if lab is None:
            print("  %-26s no label series under any of %s -- skipped" % (driver, sorted(seen)), flush=True)
            continue
        pos = set(lab[lab.astype(str).isin(["1", "1.0", "present", "True", "true"])].index.astype(str))
        neg = set(lab[lab.astype(str).isin(["0", "0.0", "absent", "False", "false"])].index.astype(str))
        if len(pos) < 2 or len(neg) < 2:
            print("  %-26s too few labelled samples (%d pos / %d neg)" % (driver, len(pos), len(neg)),
                  flush=True)
            continue

        dout = {}
        for m in have:
            mols = [x for x in (markers.get(m) or [])]
            if not mols:
                continue
            per_mol = {}
            for st in states:
                ids = pio.cellstate_pseudobulks(ctx, st)
                if not ids:
                    continue
                X = pio.pseudobulk_modality_matrix(ctx, m, ids)
                if X.shape[0] == 0:
                    continue
                key = pb.loc[list(X.index), "sample_key"].astype(str)
                cols = {c: i for i, c in enumerate(X.columns)}
                V = X.values
                for mol in mols:
                    ci = cols.get(mol)
                    if ci is None:
                        continue
                    mv, cv = [], []
                    for i, sk in enumerate(key.values):
                        v = float(V[i, ci])
                        if not np.isfinite(v):
                            continue
                        if sk in pos:
                            mv.append(round(v, 4))
                        elif sk in neg:
                            cv.append(round(v, 4))
                    if mv or cv:
                        per_mol.setdefault(mol, {})[st] = {"mutant": mv, "control": cv}
            if per_mol:
                dout[m] = per_mol
        if dout:
            out["drivers"][driver] = dout
            print("  %-26s via %-18s %s" % (driver, used_field, {k: len(v) for k, v in dout.items()}), flush=True)

    json.dump(out, open(OUTPATH, "w"), indent=1, allow_nan=False)   # fail loudly, never emit NaN
    n_mol = sum(len(v) for d in out["drivers"].values() for v in d.values())
    print("\nwrote %s -- %d drivers, %d (modality x molecule) violin sets (%.0fs)"
          % (OUTPATH, len(out["drivers"]), n_mol, time.time() - t0))


if __name__ == "__main__":
    main()
