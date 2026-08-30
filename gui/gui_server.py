#!/usr/bin/env python3
"""MOSAIC-AML decision-board GUI server (stdlib only — no pip installs).

Serves the single-file front-end `mosaic_board.html` plus a tiny JSON API that
auto-discovers per-patient reports under a `runs/` tree:

    python gui_server.py [runs_dir] [port]

  runs_dir  directory to scan for */patient_report*.json   (default: ../runs)
  port      TCP port on 127.0.0.1                           (default: 8765)

Endpoints
  GET  /                      -> the board HTML
  GET  /api/runs              -> roster summary rows (one per run with a report)
  GET  /api/report?run=NAME   -> the full patient_report.json (+ ledger evidence hash)
  GET  /api/capabilities      -> {ingest, lsf, inbox, python} — what this host can do
  GET  /api/jobs              -> in-flight + finished ingests (from each run's status.json)
  GET  /api/samples           -> candidate inputs in the inbox dir
  POST /api/ingest            -> {name, sample, mutations?, dataset?} : dispatch
                                 ingest_patient.py for a NEW scRNA sample (bsub on the
                                 cluster, detached locally) -> a new run/report

Binds to localhost only. Viewing is read-only; ingest dispatches the pipeline's own
`ingest_patient.py` (argv list, no shell) for an uploaded sample. Run this ON the cluster
(login node) so ingest jobs reach LSF + the atlas/bundles, and reach it via
`ssh -L 8765:127.0.0.1:8765 ...`.
"""
import json
import os
import re
import shutil
import socketserver
import subprocess
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs


# stdlib-only + works on the stock RHEL python3 (3.6.8) that's on every cluster node —
# http.server.ThreadingHTTPServer only exists on 3.7+, so define the same thing here.
class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True

HERE = Path(__file__).resolve().parent
HTML_PATH = HERE / "mosaic_board.html"
DEFAULT_RUNS = HERE.parent / "runs"
REPORT_GLOB = "patient_report.json"   # canonical pipeline output (ignore *_test stubs)

# --- ingest (upload a new patient) wiring ---------------------------------------------------
# The board can dispatch `ingest_patient.py` (scRNA -> composition -> arbiter -> report). On the
# cluster this submits an LSF job (compute node); with no `bsub` it runs detached locally.
PIPELINE_DIR = Path(os.environ.get("AMLMM_PIPELINE") or (HERE.parent / "pipeline")).resolve()
INGEST_SCRIPT = PIPELINE_DIR / "ingest_patient.py"
INBOX_DIR = Path(os.environ.get("AMLMM_INBOX") or (HERE.parent / "inbox")).resolve()
LSF_MEM = os.environ.get("AMLMM_LSF_MEM", "8000")
# Queue for dispatched ingests. Left at the site default, but exposed so a busy shared queue (a long
# unrelated job ahead of you) does not silently stall a patient upload: AMLMM_LSF_QUEUE=<queue>.
LSF_QUEUE = os.environ.get("AMLMM_LSF_QUEUE", "normal")
USE_LSF = (shutil.which("bsub") is not None) and os.environ.get("AMLMM_NO_LSF") != "1"
# The SERVER is stdlib-only and runs under ANY python3 (incl. the stock 3.6.8). The INGEST JOB
# needs the analysis python (anndata/scanpy/sklearn). Under LSF the job runs on a COMPUTE node,
# so default to the cluster analysis python there; locally fall back to whatever runs the server.
CLUSTER_PY = "/usr/local/anaconda3-2020/bin/python"
PYTHON = os.environ.get("AMLMM_PYTHON") or (CLUSTER_PY if USE_LSF else sys.executable)


def _runs_dir() -> Path:
    # forgiving arg order: any non-integer arg is the runs dir; a bare integer is the port.
    for a in sys.argv[1:]:
        if not a.isdigit():
            return Path(a).expanduser().resolve()
    return DEFAULT_RUNS.resolve()


def _port() -> int:
    for a in sys.argv[1:]:
        if a.isdigit():
            return int(a)
    return 8765


RUNS_DIR = _runs_dir()


def _load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 — surface a readable stub, never crash the scan
        return {"_error": f"{type(e).__name__}: {e}"}


def _report_file(run: str) -> "Path | None":
    """Resolve a run name to its report file, refusing anything outside RUNS_DIR."""
    d = (RUNS_DIR / run).resolve()
    try:
        d.relative_to(RUNS_DIR)              # path-traversal guard
    except ValueError:
        return None
    if not d.is_dir():
        return None
    hits = sorted(d.glob(REPORT_GLOB))
    return hits[0] if hits else None


# Only real reports belong on the board — the held-out sealed test, the Trumpp validation cohort, and
# genuine uploads. Everything else under runs/ (probe/regression/discovery/diagnostic scratch dirs) is
# hidden so the roster is the actual patient set, not development placeholders. Set AMLMM_SHOW_ALL=1 to
# override (show every dir with a report).
BOARD_GROUPS = [("predict_", "Held-out validation (sealed test)"),
                ("trumpp_", "Trumpp/Waclawiczek cohort"),
                ("gse_", "GSE281087 (single-cell external · n=15)"),
                ("leucegene_", "Leucegene (bulk external · n=367)"),
                ("ingest_", "Uploaded patients")]
SHOW_ALL = os.environ.get("AMLMM_SHOW_ALL") == "1"


def _board_group(name: str):
    for pre, label in BOARD_GROUPS:
        if name.startswith(pre):
            return label
    return "Other" if SHOW_ALL else None


def scan_runs() -> "list[dict]":
    """One summary row per REAL patient/validation run dir (placeholders filtered out)."""
    out = []
    if not RUNS_DIR.is_dir():
        return out
    for d in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        group = _board_group(d.name)
        if group is None:                       # a dev/scratch dir — keep it off the board
            continue
        hits = sorted(d.glob(REPORT_GLOB))
        if not hits:
            continue
        rep = _load_json(hits[0])
        # Hide reports still carrying pre-bulk-caller calls. A few samples (Colorado::AML-05/-06,
        # Milan::PT01__D14) are no longer in the atlas, so rescore_all_bulk could not recompute their
        # bulk-equivalent and they kept the old 26-mutation sc predictions — showing them next to
        # 58-category reports is misleading. Set AMLMM_SHOW_ALL=1 to see them anyway.
        # `predictor` is the multimodal trainer's stamp, `mutation_caller` the bulk caller's — a report
        # carrying EITHER is current; only ones with neither are pre-rescore leftovers.
        if not SHOW_ALL and isinstance(rep, dict) and rep.get("mutation_predictions") \
                and not rep.get("mutation_caller") and not rep.get("predictor"):
            continue
        con = rep.get("consensus", {}) if isinstance(rep, dict) else {}
        preds = rep.get("mutation_predictions") if isinstance(rep, dict) else None
        # GENE-LEVEL validation (gene_validation.py): a gene's variant categories are aggregated to one
        # gene-level call (present iff any variant is called present) vs the gene's known status. This
        # scores ~40 genes+cyto units, not just the ~17 categories whose exact variant is known.
        # The combined ratio is dominated by confirmed-absent genes the model calls absent by default
        # (one Leucegene sample scored 26/27 while missing its only real driver), so the positives are
        # forwarded separately and the roster leads with drivers recovered.
        n_correct = n_labeled = None
        n_pos_labeled = n_pos_correct = n_neg_labeled = n_neg_correct = None
        neg_scope = None
        vg = rep.get("validation_gene") if isinstance(rep, dict) else None
        if isinstance(vg, dict) and vg.get("n_labeled"):
            n_labeled = vg["n_labeled"]; n_correct = vg["n_correct"]
            n_pos_labeled = vg.get("n_pos_labeled"); n_pos_correct = vg.get("n_pos_correct")
            n_neg_labeled = vg.get("n_neg_labeled"); n_neg_correct = vg.get("n_neg_correct")
            neg_scope = vg.get("negative_scope")
        elif isinstance(preds, list):           # fallback: variant-level, for reports without a gene pass
            labeled = [p for p in preds if p.get("true_label")]
            n_labeled = len(labeled) or None
            n_correct = sum(1 for p in labeled if p.get("true_label") == p.get("call")) if labeled else None
        out.append({
            "run": d.name,
            "group": group,
            # so the drug page can offer only the runs it can actually show
            "has_drug_report": (d / "drug_report.json").is_file(),
            "sample_key": rep.get("sample_key"),
            "annotation": rep.get("annotation"),
            "dataset": rep.get("dataset"),
            "specimen_class": rep.get("specimen_class") if isinstance(rep, dict) else None,
            "validation": rep.get("validation") if isinstance(rep, dict) else None,
            "n_drivers": (len(preds) if isinstance(preds, list) else None),
            "n_labeled": n_labeled, "n_correct": n_correct,
            "n_pos_labeled": n_pos_labeled, "n_pos_correct": n_pos_correct,
            "n_neg_labeled": n_neg_labeled, "n_neg_correct": n_neg_correct,
            "negative_scope": neg_scope,
            "leading_hypothesis": con.get("leading_hypothesis"),
            "overall_confidence": con.get("overall_confidence"),
            "leading_confirmed_by_genetics": con.get("leading_confirmed_by_genetics"),
            "has_deliberation": bool(rep.get("deliberation")) if isinstance(rep, dict) else False,
            "knowledge_version": con.get("knowledge_version"),
            "error": rep.get("_error") if isinstance(rep, dict) else None,
        })
    return out


def report_for(run: str) -> "dict | None":
    f = _report_file(run)
    if f is None:
        return None
    rep = _load_json(f)
    # merge the reproducibility hash from the sibling ledger, if any
    led = f.parent / "ledger.json"
    if isinstance(rep, dict) and led.is_file():
        lj = _load_json(led)
        if isinstance(lj, dict) and lj.get("deterministic_evidence_hash"):
            rep.setdefault("_ledger", {})["deterministic_evidence_hash"] = \
                lj["deterministic_evidence_hash"]
            rep["_ledger"]["stop_reason"] = lj.get("stop_reason")
            rep["_ledger"]["rounds_run"] = lj.get("rounds_run")
    if isinstance(rep, dict):
        rep["_run"] = run
    return rep


def _slug(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_") or "patient"
    return f"ingest_{s}"


def _set_status(run_dir: Path, state, step, message=""):
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "status.json").write_text(
            json.dumps({"state": state, "step": step, "message": message, "ts": time.time()}),
            encoding="utf-8")
    except Exception:
        pass


def _spawn_explainer(run_id, run_dir):
    """After a dispatched ingest lands, add the AGENT's therapy explanations.

    The pipeline runs via bsub on a COMPUTE node, and compute nodes are firewalled from the LLM gateway
    (port 4000 answers only on the login node) — so the report would otherwise show deterministic [RULES]
    text. THIS server runs on the login node, so a watcher spawned from here can reach the gateway.
    Best-effort: if it fails, the report still has its deterministic explanation."""
    w = PIPELINE_DIR / "await_and_explain.py"
    if not w.is_file():
        return
    try:
        subprocess.Popen([PYTHON, str(w), str(run_dir), run_id],
                         stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                         cwd=str(PIPELINE_DIR), start_new_session=True)
    except Exception:  # noqa: BLE001
        pass


def dispatch_ingest(name, sample, mutations="", dataset="uploaded", kind="scrna",
                    bulk_ref="beataml", bulk_scale="auto"):
    """Kick off ingest_patient.py for a new sample (single-cell or bulk RNA). Returns {run_id, job_id, mode}
    or {error}. argv is a list (no shell) so user-supplied fields cannot inject commands."""
    if not INGEST_SCRIPT.is_file():
        return {"error": f"ingest script not found: {INGEST_SCRIPT}"}
    # Resolve the sample the caller gave us. /api/samples hands back full paths, but a user typing into
    # the box (or a script posting the label it just listed) will send a bare filename — which the child
    # process cannot find, since its working directory is not the inbox. Resolve against the inbox and
    # fail HERE with a readable message rather than launching a job that dies on a file-open error.
    sample = str(sample).strip()
    if sample and not os.path.exists(sample):
        cand = INBOX_DIR / sample
        if cand.exists():
            sample = str(cand)
        else:
            return {"error": f"sample not found: {sample!r} (also looked in the inbox, {INBOX_DIR})"}
    run_id = _slug(name)
    run_dir = RUNS_DIR / run_id
    _set_status(run_dir, "queued", "submitting", name)
    if kind == "bulk":                                    # bulk RNA -> variant-level mutation panel (no atlas load)
        _br = bulk_ref if bulk_ref in ("beataml", "leucegene", "sc") else "beataml"
        _bs = bulk_scale if bulk_scale in ("auto", "linear", "log2", "log1p") else "auto"
        cmd = [PYTHON, str(INGEST_SCRIPT), "--bulk", sample, "--name", name,
               "--bulk-ref", _br, "--bulk-scale", _bs,
               "--dataset", dataset, "--run-id", run_id, "--out-root", str(RUNS_DIR)]
    else:
        cmd = [PYTHON, str(INGEST_SCRIPT), "--sample", sample, "--name", name,
               "--dataset", dataset, "--run-id", run_id, "--out-root", str(RUNS_DIR)]
    if mutations.strip():
        cmd += ["--mutations", mutations.strip()]
    log = str(run_dir / "_job.log")
    try:
        if USE_LSF:
            mem = "4000" if kind == "bulk" else LSF_MEM      # bulk skips the atlas load -> lighter job
            bsub = ["bsub", "-q", LSF_QUEUE, "-J", f"aml_{run_id}", "-M", mem,
                    "-R", f"rusage[mem={mem}]", "-o", log, "-e", log] + cmd
            p = subprocess.run(bsub, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True, timeout=90)
            if p.returncode != 0:
                _set_status(run_dir, "error", "submit", (p.stderr or p.stdout or "bsub failed")[:300])
                return {"error": (p.stderr or p.stdout or "bsub failed")[:300], "run_id": run_id}
            m = re.search(r"Job <(\d+)>", p.stdout or "")
            _spawn_explainer(run_id, run_dir)
            return {"run_id": run_id, "job_id": (m.group(1) if m else None), "mode": "lsf"}
        # local fallback: detached background process
        fh = open(log, "w", encoding="utf-8")
        subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(PIPELINE_DIR))
        return {"run_id": run_id, "job_id": None, "mode": "local"}
    except Exception as e:  # noqa: BLE001
        _set_status(run_dir, "error", "submit", f"{type(e).__name__}: {e}")
        return {"error": f"{type(e).__name__}: {e}", "run_id": run_id}


def list_jobs() -> "list[dict]":
    """Every run dir that has a status.json (in-flight or finished ingests)."""
    out = []
    if not RUNS_DIR.is_dir():
        return out
    for d in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        sp = d / "status.json"
        if not sp.is_file():
            continue
        st = _load_json(sp)
        out.append({"run": d.name, "state": st.get("state"), "step": st.get("step"),
                    "message": st.get("message"), "ts": st.get("ts"),
                    "has_report": (d / REPORT_GLOB).is_file()})
    return out


def list_samples() -> "list[dict]":
    """Candidate inputs dropped into the inbox dir: single-cell (10x dirs / .h5ad / 10x .h5) and
    bulk RNA expression tables (.tsv / .csv / .txt)."""
    out = []
    if INBOX_DIR.is_dir():
        for p in sorted(INBOX_DIR.iterdir()):
            if p.is_dir() and (p / "filtered_feature_bc_matrix.h5").is_file():
                out.append({"path": str(p), "label": p.name, "kind": "10x dir", "input": "scrna"})
            elif p.suffix.lower() in (".h5ad", ".h5"):
                out.append({"path": str(p), "label": p.name, "kind": p.suffix.lower().lstrip("."), "input": "scrna"})
            elif p.suffix.lower() in (".tsv", ".csv", ".txt"):
                out.append({"path": str(p), "label": p.name, "kind": "bulk " + p.suffix.lower().lstrip("."), "input": "bulk"})
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "MosaicAMLBoard/1.0"

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        path, qs = u.path, parse_qs(u.query)
        if path in ("/", "/index.html", "/mosaic_board.html"):
            if not HTML_PATH.is_file():
                return self._send(500, b"mosaic_board.html not found next to gui_server.py",
                                  "text/plain; charset=utf-8")
            return self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
        if path in ("/evidence.html", "/evidence.json", "/evidence_samples.json", "/evidence_modalities.json",
                    "/mutation_frequency.json", "/reliability.json", "/validation.html", "/validation_stats.json",
                    "/calibration.html", "/cellstate_localization.json", "/vaf_by_mutation.json",
                    "/therapy.html", "/rx_validation.html", "/survival_validation.html",
                    "/cebpa_evidence.html", "/cebpa_violin_data.json", "/bulk_bakeoff_results.json"):
            fp = HERE / path.lstrip("/")
            if not fp.is_file():
                return self._send(404, path.encode() + b" not found", "text/plain; charset=utf-8")
            ctype = "text/html; charset=utf-8" if path.endswith(".html") else "application/json; charset=utf-8"
            return self._send(200, fp.read_bytes(), ctype)
        if path.startswith("/vendor/"):                 # third-party JS vendored so the pages work offline
            name = os.path.basename(path[len("/vendor/"):])
            fp = HERE / "vendor" / name
            if not name.endswith(".js") or not fp.is_file():
                return self._send(404, path.encode() + b" not found", "text/plain; charset=utf-8")
            return self._send(200, fp.read_bytes(), "application/javascript; charset=utf-8")
        if path.startswith("/val/"):                    # validation assets from ../deliverables (figures, tables)
            rel = path[len("/val/"):]
            # one optional whitelisted subdirectory (deliverables/figures/), basename only otherwise --
            # the basename strip is the path-traversal guard, so the subdir must be an exact match
            sub = ""
            if "/" in rel:
                head, rel = rel.split("/", 1)
                if head in ("figures", "tables"):
                    sub = head
            name = os.path.basename(rel)
            ext = os.path.splitext(name)[1].lower()
            ctype = {".png": "image/png", ".pdf": "application/pdf", ".tsv": "text/tab-separated-values; charset=utf-8",
                     ".md": "text/markdown; charset=utf-8", ".json": "application/json; charset=utf-8"}.get(ext)
            fp = (HERE.parent / "deliverables" / sub / name) if sub else (HERE.parent / "deliverables" / name)
            if ctype is None or not fp.is_file():
                return self._send(404, path.encode() + b" not found", "text/plain; charset=utf-8")
            return self._send(200, fp.read_bytes(), ctype)
        if path == "/api/runs":
            return self._json({"runs_dir": str(RUNS_DIR), "runs": scan_runs()})
        if path == "/api/report":
            run = (qs.get("run") or [""])[0]
            rep = report_for(run) if run else None
            if rep is None:
                return self._json({"error": f"no report for run {run!r}"}, code=404)
            return self._json(rep)
        if path == "/api/drug_report":
            # COMPASS-AML per-patient prioritisation, written next to the patient report by predict_drugs.py
            run = (qs.get("run") or [""])[0]
            f = _report_file(run)
            dr = (f.parent / "drug_report.json") if f is not None else None
            if dr is None or not dr.is_file():
                return self._json({"error": f"no drug report for run {run!r}",
                                   "hint": "run pipeline/predict_drugs.py for this sample"}, code=404)
            return self._json(_load_json(dr))
        if path == "/api/capabilities":
            return self._json({"ingest": INGEST_SCRIPT.is_file(), "lsf": USE_LSF,
                               "inbox": str(INBOX_DIR), "inbox_exists": INBOX_DIR.is_dir(),
                               "python": PYTHON})
        if path == "/api/jobs":
            return self._json({"jobs": list_jobs()})
        if path == "/api/samples":
            return self._json({"inbox": str(INBOX_DIR), "samples": list_samples()})
        return self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self):  # noqa: N802
        u = urlparse(self.path)
        if u.path != "/api/ingest":
            return self._send(404, b"not found", "text/plain; charset=utf-8")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:  # noqa: BLE001
            return self._json({"error": f"bad request: {e}"}, code=400)
        name = str(body.get("name") or "").strip()
        sample = str(body.get("sample") or "").strip()
        if not name or not sample:
            return self._json({"error": "name and sample are required"}, code=400)
        kind = "bulk" if str(body.get("kind") or "").lower() in ("bulk", "bulk_rna", "bulkrna") else "scrna"
        res = dispatch_ingest(name, sample, str(body.get("mutations") or ""),
                              str(body.get("dataset") or "uploaded"), kind=kind,
                              bulk_ref=str(body.get("bulk_ref") or "beataml"),
                              bulk_scale=str(body.get("bulk_scale") or "auto"))
        return self._json(res, code=200 if res.get("run_id") and not res.get("error") else 400)

    do_HEAD = do_GET

    def log_message(self, fmt, *args):   # quieter console
        if "/api/" in (args[0] if args else ""):
            sys.stderr.write("  %s\n" % (fmt % args))


def main():
    port = _port()
    if not RUNS_DIR.is_dir():
        print(f"[warn] runs dir not found: {RUNS_DIR}\n"
              f"       pass one:  python gui_server.py /path/to/runs", file=sys.stderr)
    n = len(scan_runs())
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"MOSAIC-AML board -> {url}")
    print(f"  runs dir : {RUNS_DIR}")
    print(f"  reports  : {n} found")
    if INGEST_SCRIPT.is_file():
        print(f"  ingest   : enabled ({'LSF/bsub' if USE_LSF else 'local detached'}) "
              f"via {INGEST_SCRIPT.name}; inbox {INBOX_DIR}")
    else:
        print(f"  ingest   : disabled (no {INGEST_SCRIPT})")
    print("  Ctrl-C to stop.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
        httpd.server_close()


if __name__ == "__main__":
    main()
