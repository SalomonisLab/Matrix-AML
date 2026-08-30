#!/usr/bin/env python3
"""Freeze the whole MOSAIC-AML web UI into a static, self-contained bundle you can just hand to people.

The live GUI is a set of single-file HTML pages talking to a small read-only JSON API. That makes a
static export straightforward: bake every API response into the bundle and swap `fetch` for a lookup.

Two constraints drive the design:

  * **It must work by double-clicking `index.html`, with no server.** Browsers block `fetch()` against
    `file://` URLs, so the data cannot be shipped as `.json` files the pages fetch. It is shipped as
    `.js` files assigned to a global instead -- `<script src>` is not subject to that restriction --
    and a shim intercepts `fetch` and serves from the global.
  * **Nothing in it may look interactive when it is not.** Capabilities report `ingest: false` so the
    upload button never appears, any non-GET request returns an explicit "this is a static export"
    error rather than failing silently, and a banner says so on every page.

Images, PDFs and TSVs stay as ordinary files (`<img src>` and download links work fine from `file://`);
only the JSON is inlined.

  python build_static_site.py [--out DIR] [--no-zip]
      -> deliverables/mosaic_static/          the browsable bundle
      -> deliverables/MOSAIC-AML_static_site.zip
"""
import os, sys, json, glob, time, shutil, zipfile, argparse, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GUI = os.path.join(ROOT, "gui")
RUNS = os.path.join(ROOT, "runs")
DELIV = os.path.join(ROOT, "deliverables")

PAGES = {"mosaic_board.html": "index.html", "validation.html": "validation.html",
         "rx_validation.html": "rx_validation.html", "therapy.html": "therapy.html",
         "survival_validation.html": "survival_validation.html",
         "calibration.html": "calibration.html", "evidence.html": "evidence.html",
         "cebpa_evidence.html": "cebpa_evidence.html"}
# JSON assets the pages fetch by name from the gui/ directory
GUI_JSON = ["evidence.json", "evidence_samples.json", "evidence_modalities.json", "mutation_frequency.json", "reliability.json",
            "validation_stats.json", "cellstate_localization.json", "vaf_by_mutation.json",
            "cebpa_violin_data.json", "bulk_bakeoff_results.json"]
# deliverables served under /val/ that are fetched as JSON (as opposed to linked for download)
VAL_JSON = ["drug_model_validation.json", "drug_model_card.json", "state_response_validation.json",
            "feature_attribution.json", "calibration_dca.json", "production_fused_model.json",
            "modality_breakdown_current.json", "pooled_heldout_eval.json",
            "validation_tcga_laml.json", "survival_model_card.json"]
ASSET_EXT = (".png", ".pdf", ".tsv", ".md", ".xlsx")


def api_key(path, params=None):
    """The same normalised key the shim computes, so both sides agree without a lookup table."""
    if params:
        q = "&".join("%s=%s" % (k, params[k]) for k in sorted(params))
        return "%s?%s" % (path, q)
    return path


def prune_drug_report(r):
    """Drop what the page never reads and the file already says twice.

    `drug_response.evidence.all` and `molecular_mechanism.evidence.all` repeat, for all 118 inhibitors,
    rows that `per_drug` already carries -- about 65 kB per report, times 30 reports.
    """
    for a in r.get("agents") or []:
        ev = a.get("evidence")
        if isinstance(ev, dict) and "all" in ev and "top" in ev:
            ev.pop("all", None)
    return r


def js_blob(name, mapping, out_dir):
    """Write one data shard as a plain assignment, minified."""
    p = os.path.join(out_dir, "data", name)
    with open(p, "w", encoding="utf-8") as f:
        f.write("window.__MOSAIC_STATIC__=Object.assign(window.__MOSAIC_STATIC__||{},")
        json.dump(mapping, f, separators=(",", ":"), default=str)
        f.write(");\n")
    return os.path.getsize(p)


SHIM = r"""/* MOSAIC-AML static export -- serve the frozen API out of a global instead of the network. */
(function () {
  var D = window.__MOSAIC_STATIC__ || {};
  /* Candidate keys for one request URL. The build rewrites "/val/x.png" to "val/x.png" so that
     <img> and download links resolve relative to the bundle; that same rewrite also lands inside
     fetch() calls, so a request may arrive as "val/x.json" while the baked key is "/val/x.json".
     Rather than trying to rewrite only some occurrences, we just try both shapes. */
  function keys(u) {
    var s = String(u), i = s.indexOf('?');
    var p = i < 0 ? s : s.slice(0, i), tail = '';
    if (i >= 0) {
      var q = new URLSearchParams(s.slice(i + 1)), parts = [];
      q.forEach(function (v, k) { parts.push([k, v]); });
      parts.sort(function (a, b) { return a[0] < b[0] ? -1 : 1; });
      tail = '?' + parts.map(function (kv) { return kv[0] + '=' + kv[1]; }).join('&');
    }
    var base = p.split('/').pop();
    var cands = [];
    if (p.charAt(0) === '/') cands.push(p);
    else { cands.push('/' + p); cands.push('/' + base); }
    cands.push('/val/' + base);
    cands.push('/' + base);
    return cands.map(function (c) { return c + tail; });
  }
  function res(status, body) {
    var txt = typeof body === 'string' ? body : JSON.stringify(body);
    return {
      ok: status >= 200 && status < 300, status: status, statusText: '' + status,
      json: function () { return Promise.resolve(typeof body === 'string' ? JSON.parse(body) : body); },
      text: function () { return Promise.resolve(txt); }
    };
  }
  var realFetch = window.fetch ? window.fetch.bind(window) : null;
  window.fetch = function (u, opts) {
    var m = (opts && opts.method ? opts.method : 'GET').toUpperCase();
    if (m !== 'GET')
      return Promise.resolve(res(405, {
        error: 'This is a static export of MOSAIC-AML. Uploading a patient and re-running the ' +
               'pipeline need the live server.'
      }));
    var ks = keys(u);
    for (var i = 0; i < ks.length; i++)
      if (Object.prototype.hasOwnProperty.call(D, ks[i])) return Promise.resolve(res(200, D[ks[i]]));
    // not baked in: let a real request through if this is being served over http, else say so plainly
    if (realFetch && location.protocol.indexOf('http') === 0) return realFetch(u, opts);
    return Promise.resolve(res(404, { error: 'not included in the static export: ' + ks[0] }));
  };

  // Safety net for any asset URL a page builds at runtime with a leading slash.
  document.addEventListener('error', function (e) {
    var el = e.target;
    if (el && el.tagName === 'IMG' && el.getAttribute('src') && el.getAttribute('src').charAt(0) === '/'
        && !el.dataset.rebased) {
      el.dataset.rebased = '1';
      el.setAttribute('src', el.getAttribute('src').replace(/^\/+/, ''));
    }
  }, true);

  document.addEventListener('DOMContentLoaded', function () {
    var b = document.createElement('div');
    b.id = 'static-export-banner';
    b.innerHTML = '<b>Static export</b> · a frozen, read-only snapshot of the MOSAIC-AML board' +
      ' — ' + (window.__MOSAIC_META__ || {}).generated +
      '. Everything you can click is real data; uploading a patient and re-running the pipeline need' +
      ' the live server.';
    b.setAttribute('style',
      'font:12.5px/1.5 system-ui,"Segoe UI",sans-serif;background:#f2efe6;color:#5f5b4e;' +
      'border-bottom:1px solid #e4e3dc;padding:7px 18px;letter-spacing:.01em');
    document.body.insertBefore(b, document.body.firstChild);
  });
})();
"""

README = """MOSAIC-AML — static export
==========================

A frozen, read-only copy of the MOSAIC-AML decision board and its validation pages.
No install, no server, no internet.

  ->  open  index.html  in any browser (double-click it)

What's in it
------------
  index.html          the decision board: {n_runs} specimens, per-mutation calls, therapy
                      hypotheses, tests to run, and the COMPASS-AML drug card
  therapy.html        COMPASS-AML per-patient drug prioritisation ({n_drug} specimens),
                      ranked per clinical tier; click a row for the underlying evidence
  rx_validation.html  how the drug layer was validated, with every figure
  validation.html     how the mutation caller was validated
  survival_validation.html
                      how the survival layer was validated, including the frozen
                      transfer to TCGA-LAML and the limits it establishes
  calibration.html    calibration and decision-curve detail
  evidence.html       per-mutation evidence view (classifiers, markers, where the signal lives)
  val/                the figures, PDFs and tables the pages link to
  data/               the frozen API responses ({mb:.0f} MB)

What is deliberately switched off
---------------------------------
Adding a patient, re-running the pipeline and anything else that writes need the live
server. The buttons are hidden and any attempt returns an explicit message rather than
failing quietly. Everything you can read is the real thing.

Two standing caveats carried over from the live system
------------------------------------------------------
  * Mutation calls marked "predicted" are inferred from expression, not sequenced.
  * COMPASS-AML predicts EX-VIVO sensitivity from the BeatAML2 functional screen. That is a
    prioritisation signal for trial matching or laboratory validation — not an estimate of
    clinical benefit, and not a treatment recommendation.

Generated {generated} from {root}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DELIV, "mosaic_static"))
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    out = os.path.abspath(a.out)

    # gui_server resolves its runs dir from argv at import time
    sys.argv = ["gui_server.py", RUNS]
    sys.path.insert(0, GUI)
    import gui_server as GS

    if os.path.isdir(out):
        shutil.rmtree(out)
    for sub in ("", "data", "val", os.path.join("val", "figures")):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    # ---------------------------------------------------------------- data ----
    runs = GS.scan_runs()
    core = {api_key("/api/runs"): {"runs_dir": "(static export)", "runs": runs},
            api_key("/api/capabilities"): {"ingest": False, "lsf": False, "static_export": True,
                                           "inbox": None, "inbox_exists": False, "python": None},
            api_key("/api/jobs"): {"jobs": []},
            api_key("/api/samples"): {"samples": [], "dir": None}}
    for name in GUI_JSON:
        p = os.path.join(GUI, name)
        if os.path.exists(p):
            core["/" + name] = json.load(open(p, encoding="utf-8"))
    for name in VAL_JSON:
        p = os.path.join(DELIV, name)
        if os.path.exists(p):
            core["/val/" + name] = json.load(open(p, encoding="utf-8"))

    reports, drugs = {}, {}
    for r in runs:
        run = r["run"]
        rep = GS.report_for(run)
        if rep is not None:
            reports[api_key("/api/report", {"run": run})] = rep
        dp = os.path.join(RUNS, run, "drug_report.json")
        if os.path.exists(dp):
            drugs[api_key("/api/drug_report", {"run": run})] = prune_drug_report(
                json.load(open(dp, encoding="utf-8")))

    meta = {"generated": time.strftime("%d %B %Y"), "n_runs": len(reports), "n_drug": len(drugs)}
    sizes = {
        "core.js": js_blob("core.js", core, out),
        "reports.js": js_blob("reports.js", reports, out),
        "drugs.js": js_blob("drugs.js", drugs, out),
    }
    with open(os.path.join(out, "data", "meta.js"), "w", encoding="utf-8") as f:
        f.write("window.__MOSAIC_META__=%s;\n" % json.dumps(meta))
    with open(os.path.join(out, "data", "static-shim.js"), "w", encoding="utf-8") as f:
        f.write(SHIM)

    # -------------------------------------------------------------- assets ----
    # Plotly is vendored rather than pulled from a CDN: the whole point of the export is that it works
    # on a laptop with no internet, and three of the pages are nothing but charts without it.
    n_assets = 0
    vend = os.path.join(GUI, "vendor")
    if os.path.isdir(vend):
        os.makedirs(os.path.join(out, "vendor"), exist_ok=True)
        for p in glob.glob(os.path.join(vend, "*.js")):
            shutil.copy2(p, os.path.join(out, "vendor", os.path.basename(p)))
            n_assets += 1
    for src_dir, dst_dir in ((DELIV, os.path.join(out, "val")),
                             (os.path.join(DELIV, "figures"), os.path.join(out, "val", "figures"))):
        for p in glob.glob(os.path.join(src_dir, "*")):
            if os.path.isfile(p) and p.lower().endswith(ASSET_EXT):
                shutil.copy2(p, os.path.join(dst_dir, os.path.basename(p)))
                n_assets += 1

    # --------------------------------------------------------------- pages ----
    head = ('<script src="data/meta.js"></script>'
            '<script src="data/core.js"></script>'
            '<script src="data/reports.js"></script>'
            '<script src="data/drugs.js"></script>'
            '<script src="data/static-shim.js"></script>')
    link_map = {'href="/"': 'href="index.html"', "href='/'": "href='index.html'"}
    for src, dst in PAGES.items():
        p = os.path.join(GUI, src)
        if not os.path.exists(p):
            continue
        s = open(p, encoding="utf-8").read()
        s = s.replace("/val/", "val/")                       # figures, PDFs and TSVs sit under val/
        for page in PAGES:
            s = s.replace('"/' + page, '"' + PAGES[page]).replace("'/" + page, "'" + PAGES[page])
        for k, v in link_map.items():
            s = s.replace(k, v)
        if "</head>" in s:
            s = s.replace("</head>", head + "</head>", 1)
        else:                                                # no explicit head close: put it first
            s = s.replace("<body>", "<body>" + head, 1)
        open(os.path.join(out, dst), "w", encoding="utf-8").write(s)

    mb = sum(sizes.values()) / 1e6
    open(os.path.join(out, "README.txt"), "w", encoding="utf-8").write(
        README.format(mb=mb, root=ROOT, **meta))

    print("static bundle -> %s" % out)
    print("  pages   %d" % len(PAGES))
    print("  data    %s" % "  ".join("%s %.1f MB" % (k, v / 1e6) for k, v in sizes.items()))
    print("  runs    %d reports, %d with a drug report" % (len(reports), len(drugs)))
    print("  assets  %d files" % n_assets)

    if not a.no_zip:
        z = os.path.join(DELIV, "MOSAIC-AML_static_site.zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for dirpath, _, files in os.walk(out):
                for fn in files:
                    full = os.path.join(dirpath, fn)
                    zf.write(full, os.path.join("MOSAIC-AML_static",
                                                os.path.relpath(full, out)))
        print("  zip     %s  (%.1f MB)" % (z, os.path.getsize(z) / 1e6))
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
