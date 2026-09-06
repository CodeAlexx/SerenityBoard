#!/usr/bin/env python3
"""LoRA analytics parity: native /api/lora/* vs serenityboard.lora_analytics (torch oracle).

usage: lora_parity.py <serenityboardd> <lora_a.safetensors> [<lora_b.safetensors>]
"""
import json, math, os, subprocess, sys, tempfile, time, urllib.request, uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from serenityboard.lora_analytics import analyze_lora_file, compare_lora_files, diagnose, summary_stats

binary, path_a = sys.argv[1], sys.argv[2]
path_b = sys.argv[3] if len(sys.argv) > 3 else None
PORT = 6199
REL = 2e-4


def close(x, y, where):
    if isinstance(x, dict):
        assert isinstance(y, dict) and sorted(x) == sorted(y), f"{where}: keys {sorted(x)[:5]} vs {sorted(y)[:5]}"
        return sum(close(x[k], y[k], f"{where}.{k}") for k in x)
    if isinstance(x, list):
        assert isinstance(y, list) and len(x) == len(y), f"{where}: length {len(x)} vs {len(y)}"
        return sum(close(a, b, f"{where}[{i}]") for i, (a, b) in enumerate(zip(x, y)))
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if isinstance(x, float) and math.isinf(x):
            assert isinstance(y, float) and math.isinf(y) and (x > 0) == (y > 0), f"{where}: {x} vs {y}"
            return 0
        # diff_*_pct = 100 * (vb - va) / |va| amplifies float32-vs-double noise by
        # cancellation; bound it in percentage points instead (2 * 100 * REL).
        tol = 200 * REL if where.rsplit(".", 1)[-1].startswith("diff_") else REL * max(abs(x), abs(y), 1e-6)
        if abs(x - y) > tol:
            return 1 if not _report(where, x, y) else 1
        return 0
    assert x == y, f"{where}: {x!r} vs {y!r}"
    return 0


def _report(where, x, y):
    print(f"MISMATCH {where}: py={x!r} native={y!r} rel={abs(x - y) / max(abs(x), 1e-12):.3e}")
    return True


def post(url, body, headers):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{url}", data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def multipart(files):
    boundary = uuid.uuid4().hex
    body = b""
    for field, (fname, data) in files.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{fname}\"\r\n"
                 f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}


logdir = tempfile.mkdtemp(prefix="sb_lora_parity_")
proc = subprocess.Popen([binary, "serve", "--logdir", logdir, "--port", str(PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.5)
mismatches = 0
try:
    t0 = time.time()
    m_a = analyze_lora_file(path_a)
    py = {"layers": m_a, "summary": summary_stats(m_a), "diagnostics": diagnose(m_a), "num_layers": len(m_a)}
    t_py = time.time() - t0
    t0 = time.time()
    status, nat = post("/api/lora/analyze", json.dumps({"path": path_a}).encode(), {"Content-Type": "application/json"})
    t_nat = time.time() - t0
    assert status == 200, nat
    mismatches += close(py, nat, "analyze")
    print(f"analyze: {len(m_a)} layers, py {t_py:.1f}s (torch), native {t_nat:.1f}s, diagnostics={len(nat['diagnostics'])}")

    # error envelopes
    s, e = post("/api/lora/analyze", json.dumps({"path": "/nonexistent.safetensors"}).encode(), {"Content-Type": "application/json"})
    assert s == 404 and e["error"]["message"] == "File not found: /nonexistent.safetensors", (s, e)
    s, e = post("/api/lora/analyze", json.dumps({"path": binary}).encode(), {"Content-Type": "application/json"})
    assert s == 400 and e["error"]["message"] == "Only .safetensors files are supported", (s, e)
    print("analyze: 404/400 envelopes match")

    # upload variant
    body, hdr = multipart({"file": (os.path.basename(path_a), open(path_a, "rb").read())})
    status, up = post("/api/lora/analyze-upload", body, hdr)
    assert status == 200, up
    assert up.pop("filename") == os.path.basename(path_a)
    mismatches += close(py, up, "analyze-upload")
    print("analyze-upload: identical to path analyze")

    if path_b:
        comp = compare_lora_files(path_a, path_b)
        m_b = analyze_lora_file(path_b)
        py_c = {"layers": comp, "summary_a": summary_stats(m_a), "summary_b": summary_stats(m_b),
                "diagnostics": diagnose(m_a) + diagnose(m_b), "num_layers": len(comp)}
        status, nat_c = post("/api/lora/compare", json.dumps({"path_a": path_a, "path_b": path_b}).encode(),
                             {"Content-Type": "application/json"})
        assert status == 200, nat_c
        mismatches += close(py_c, nat_c, "compare")
        print(f"compare: {len(comp)} layers checked (incl. diff_*_pct)")
        body, hdr = multipart({"file_a": ("a.safetensors", open(path_a, "rb").read()),
                               "file_b": ("b.safetensors", open(path_b, "rb").read())})
        status, up_c = post("/api/lora/compare-upload", body, hdr)
        assert status == 200, up_c
        assert up_c.pop("filename_a") == "a.safetensors" and up_c.pop("filename_b") == "b.safetensors"
        mismatches += close(py_c, up_c, "compare-upload")
        print("compare-upload: identical to path compare")
finally:
    proc.terminate()
    proc.wait()
print(f"LoRA parity: {mismatches} mismatches")
sys.exit(1 if mismatches else 0)
