#!/usr/bin/env python3
"""Route parity gate: Python server (oracle) vs native server on the SAME logdir.

Builds a logdir with the Python writer (all data classes), starts both servers,
fetches every route with several query variants from both, and compares status
codes + parsed JSON (floats within 1e-9 relative). Also checks the WebSocket
subscribe -> scalar push path on both servers with a stdlib RFC 6455 client.

usage: route_parity.py <native-serenityboardd> [--keep]
"""
from __future__ import annotations

import base64, json, math, os, socket, struct, subprocess, sys, tempfile, time, shutil, hashlib
import urllib.request, urllib.error

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np
from serenityboard import SummaryWriter

NATIVE = sys.argv[1]
KEEP = "--keep" in sys.argv
PY = sys.executable


def build_logdir(logdir: str) -> None:
    rng = np.random.default_rng(0)
    for name, seed_val, n in (("run_a", 1, 40), ("nested/run_b", 2, 25)):
        with SummaryWriter(logdir=logdir, run_name=name, hparams={"lr": 1e-3, "max_steps": n, "arch": "dit"},
                           system_metrics=False) as w:
            for s in range(n):
                w.add_scalar("loss/train", float(np.exp(-s / 10) + rng.normal(0, 0.01)), s)
                w.add_scalar("lr", 1e-3 * (1 - s / n), s)
                if s % 5 == 0:
                    w.add_histogram("weights/layer0", rng.normal(seed_val, 1, 500), s, bins=32)
                    w.add_text("notes", f"step {s} note", s)
                    w.add_trace(s, "forward", 10 + s * 0.1, {"k": s})
                    w.add_trace(s, "backward", 20 + s * 0.2)
                    w.add_eval("suite", f"case-{s % 3}", s, "psnr", 30 + s * 0.05)
                    w.add_pr_curve("pr/cls", rng.integers(0, 2, 64), rng.random(64), s, num_thresholds=11)
            img = (rng.random((3, 16, 16)) * 255).astype(np.uint8)
            w.add_image("samples/img", img, 5)
            w.add_image("samples/img", img, 10)
            w.add_audio("audio/clip", (np.sin(np.linspace(0, 200, 4000)) * 0.5).astype(np.float32), 5, sample_rate=8000)
            w.add_hparams({"lr": 1e-3, "max_steps": n, "arch": "dit"}, {"final_loss": 0.05 * seed_val})
            w.add_custom_scalars_layout({"categories": [{"title": "Loss", "charts": [{"title": "train", "tags": ["loss/.*"]}]}]})
            w.add_plugin_data("plug", "t", {"x": 1}, 1)
            w.add_mesh("mesh/cube", np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32),
                       colors=np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8),
                       faces=np.array([[[0, 1, 2]]], dtype=np.int32), config_dict={"a": 1}, global_step=3)
            w.add_embedding(rng.random((20, 4)).astype(np.float32), metadata=[f"p{i}" for i in range(20)], global_step=2,
                            tag="emb")


def get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=30) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def request(base, method, path, body=None):
    req = urllib.request.Request(base + path, data=body.encode() if body else None, method=method,
                                 headers={"Content-Type": "application/json"} if body else {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def approx_equal(a, b, path="$"):
    """Structural JSON equality with float tolerance; returns first mismatch or None."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return f"{path}: keys {sorted(set(a) ^ set(b))}"
        for k in a:
            m = approx_equal(a[k], b[k], f"{path}.{k}")
            if m:
                return m
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            m = approx_equal(x, y, f"{path}[{i}]")
            if m:
                return m
        return None
    if isinstance(a, bool) or isinstance(b, bool):
        return None if a == b else f"{path}: {a!r} != {b!r}"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b or (math.isnan(a) and math.isnan(b)):
            return None
        tol = 1e-9 * max(abs(a), abs(b), 1.0)
        return None if abs(a - b) <= tol else f"{path}: {a!r} != {b!r}"
    return None if a == b else f"{path}: {a!r} != {b!r}"


def ws_probe(host, port, run):
    """Minimal RFC6455 client: subscribe, then expect at least one scalar message within 3 s."""
    s = socket.create_connection((host, port), timeout=5)
    key = base64.b64encode(os.urandom(16)).decode()
    s.sendall((f"GET /ws/live HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    head = b""
    while b"\r\n\r\n" not in head:
        head += s.recv(4096)
    status = head.split(b"\r\n")[0]
    expect = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
    ok_accept = expect.encode() in head
    payload = json.dumps({"subscribe": {"runs": [run], "tags": ["loss/*"], "kinds": ["scalar", "trace", "eval"]}}).encode()
    mask = os.urandom(4)
    frame = bytes([0x81]) + (bytes([0x80 | len(payload)]) if len(payload) < 126 else bytes([0x80 | 126]) + struct.pack(">H", len(payload)))
    frame += mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    s.sendall(frame)
    s.settimeout(3.5)
    msgs = []
    buf = b""
    deadline = time.time() + 3.0
    try:
        while time.time() < deadline:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
            while len(buf) >= 2:
                b1 = buf[1] & 0x7F
                off = 2
                if b1 == 126:
                    if len(buf) < 4: break
                    ln = struct.unpack(">H", buf[2:4])[0]; off = 4
                elif b1 == 127:
                    if len(buf) < 10: break
                    ln = struct.unpack(">Q", buf[2:10])[0]; off = 10
                else:
                    ln = b1
                if len(buf) < off + ln: break
                msgs.append(json.loads(buf[off:off + ln].decode()))
                buf = buf[off + ln:]
    except socket.timeout:
        pass
    s.close()
    return status.decode(), ok_accept, msgs


def main():
    logdir = tempfile.mkdtemp(prefix="sb_parity_")
    build_logdir(logdir)
    py_port, nat_port = 46061, 46062
    py = subprocess.Popen([PY, "-c", f"from serenityboard.server.app import create_app; import uvicorn; uvicorn.run(create_app({logdir!r}), host='127.0.0.1', port={py_port}, log_level='warning')"],
                          cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    nat = subprocess.Popen([NATIVE, "serve", "--logdir", logdir, "--port", str(nat_port), "--host", "127.0.0.1"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    py_base, nat_base = f"http://127.0.0.1:{py_port}", f"http://127.0.0.1:{nat_port}"
    for base in (py_base, nat_base):
        for _ in range(100):
            try:
                get(base, "/api/runs"); break
            except Exception:
                time.sleep(0.1)
    time.sleep(0.5)
    runs = ["run_a", "nested__run_b"]
    routes = ["/api/runs", "/api/plugins",
              "/api/compare/scalars?tag=loss/train&runs=run_a,nested__run_b",
              "/api/compare/scalars?tag=loss/train&runs=run_a,nested__run_b&downsample=7&x_axis=relative",
              "/api/compare/scalars?tag=lr&runs=run_a,missing&x_axis=wall_time",
              "/api/compare/eval?suite=suite&runs=run_a,nested__run_b", "/api/compare/eval?suite=suite&runs=run_a&score=psnr",
              "/api/compare/hparams?runs=run_a,nested__run_b,missing",
              "/api/runs/missing/tags"]
    for run in runs:
        routes += [f"/api/runs/{run}/tags", f"/api/runs/{run}/metrics", f"/api/runs/{run}/hparams",
                   f"/api/runs/{run}/scalars?tag=loss/train&downsample=0", f"/api/runs/{run}/scalars?tag=loss/train",
                   f"/api/runs/{run}/scalars?tag=loss/train&downsample=7", f"/api/runs/{run}/scalars?tag=loss/train&downsample=3&x_axis=relative",
                   f"/api/runs/{run}/scalars?tag=lr&x_axis=wall_time", f"/api/runs/{run}/scalars?tag=nope",
                   f"/api/runs/{run}/scalars/last?tags=loss/train,lr,nope",
                   f"/api/runs/{run}/histograms?tag=weights/layer0", f"/api/runs/{run}/histograms?tag=weights/layer0&downsample=3",
                   f"/api/runs/{run}/distributions?tag=weights/layer0", f"/api/runs/{run}/text?tag=notes", f"/api/runs/{run}/text?tag=notes&limit=2",
                   f"/api/runs/{run}/traces", f"/api/runs/{run}/traces?step_from=5&step_to=15", f"/api/runs/{run}/eval?suite=suite",
                   f"/api/runs/{run}/eval?suite=suite&step=5", f"/api/runs/{run}/artifacts?tag=samples/img", f"/api/runs/{run}/artifacts?tag=samples/img&kind=image",
                   f"/api/runs/{run}/artifacts?tag=samples/img&kind=video", f"/api/runs/{run}/images?tag=samples/img",
                   f"/api/runs/{run}/pr-curves?tag=pr/cls", f"/api/runs/{run}/pr-curves?tag=pr/cls&downsample=2",
                   f"/api/runs/{run}/audio?tag=audio/clip", f"/api/runs/{run}/custom-scalars/layout",
                   f"/api/runs/{run}/custom-scalars/data?tags=loss/.*,^lr$", f"/api/runs/{run}/graphs",
                   f"/api/runs/{run}/meshes", f"/api/runs/{run}/meshes?tag=mesh/cube", f"/api/runs/{run}/meshes?tag=mesh/cube&step=3",
                   f"/api/runs/{run}/meshes?tag=mesh/cube&step=99", f"/api/runs/{run}/embeddings", f"/api/runs/{run}/embeddings?tag=emb&step=2",
                   f"/api/runs/{run}/embeddings?tag=emb&step=7", f"/api/runs/{run}/notes",
                   f"/api/runs/{run}/export?format=json&tags=loss/train,lr", f"/api/runs/{run}/export?format=json&x_axis=relative",
                   f"/api/runs/{run}/blob/not-a-key", f"/api/runs/{run}/blob/0123456789abcdef.png"]
    failures = 0
    ignore_keys = {"last_activity", "start_time", "wall_time", "updated_at"}  # servers are compared on the same db, so these match; kept for safety

    def strip(x):
        return x
    for route in routes:
        sp, bp, hp = get(py_base, route)
        sn, bn, hn = get(nat_base, route)
        if sp != sn:
            failures += 1; print(f"STATUS MISMATCH {route}: py={sp} native={sn} {bn[:160]!r}"); continue
        ctype_p = hp.get("content-type", hp.get("Content-Type", "")).split(";")[0]
        if ctype_p == "application/json":
            try:
                jp, jn = json.loads(bp), json.loads(bn)
            except Exception as e:
                failures += 1; print(f"JSON PARSE {route}: {e} native={bn[:200]!r}"); continue
            m = approx_equal(jp, jn)
            if m:
                failures += 1; print(f"BODY MISMATCH {route}: {m}")
            else:
                print(f"ok {sp} {route}")
        else:
            print(f"ok {sp} {route} ({ctype_p}, {len(bn)} bytes)")
    # blob bytes identical
    imgs = json.loads(get(py_base, "/api/runs/run_a/artifacts?tag=samples/img")[1])
    key = imgs[0]["blob_key"]
    bp, bn = get(py_base, f"/api/runs/run_a/blob/{key}")[1], get(nat_base, f"/api/runs/run_a/blob/{key}")[1]
    if bp == bn and get(nat_base, f"/api/runs/run_a/blob/{key}")[2].get("Content-Type", "").startswith("image/png"):
        print(f"ok blob bytes identical ({len(bn)} bytes, image/png)")
    else:
        failures += 1; print("BLOB MISMATCH")
    # csv export byte-compare (after normalizing line endings)
    cp = get(py_base, "/api/runs/run_a/export?format=csv&tags=loss/train,lr")[1].replace(b"\r\n", b"\n")
    cn = get(nat_base, "/api/runs/run_a/export?format=csv&tags=loss/train,lr")[1].replace(b"\r\n", b"\n")
    if cp == cn:
        print("ok csv export identical")
    else:
        failures += 1; print("CSV MISMATCH:\n  py :", cp[:200], "\n  nat:", cn[:200])
    # notes round trip on native, read back by python (same db)
    request(nat_base, "PUT", "/api/runs/run_a/notes", json.dumps({"note": "hello from native"}))
    time.sleep(0.2)
    np_note = json.loads(get(py_base, "/api/runs/run_a/notes")[1])
    if np_note["note"] == "hello from native":
        print("ok notes written by native, read by python")
    else:
        failures += 1; print("NOTES MISMATCH", np_note)
    # 404 envelope shape
    st, body = get(nat_base, "/api/runs/missing/tags")[:2]
    env = json.loads(body)
    if st == 404 and env == {"error": {"code": "not_found", "message": "Run not found", "details": {}}}:
        print("ok 404 envelope")
    else:
        failures += 1; print("ENVELOPE MISMATCH", st, env)
    # WebSocket: subscribe, then append scalars with the python writer (resume), expect pushes on both servers
    results = {}
    for name, port in (("python", py_port), ("native", nat_port)):
        # append new points while subscribed
        import threading
        def appender():
            time.sleep(0.8)
            with SummaryWriter(logdir=logdir, run_name="run_a", resume_step=39, system_metrics=False) as w:
                for s in range(40, 46):
                    w.add_scalar("loss/train", 0.5, s)
                    w.add_trace(s, "forward", 1.0)
                    w.add_eval("suite", "case-0", s, "psnr", 1.0)
                w.flush()
        t = threading.Thread(target=appender); t.start()
        status, ok_accept, msgs = ws_probe("127.0.0.1", port, "run_a")
        t.join()
        kinds = sorted({m.get("type") for m in msgs})
        scalar_pts = sum(len(m.get("points", [])) for m in msgs if m.get("type") == "scalar")
        results[name] = (status, ok_accept, kinds, scalar_pts)
        print(f"ws {name}: {status} accept_ok={ok_accept} kinds={kinds} scalar_points={scalar_pts}")
        time.sleep(0.5)
    if not (results["native"][1] and "101" in results["native"][0] and "scalar" in results["native"][2]):
        failures += 1; print("WS FAIL native")
    py.terminate(); nat.terminate()
    py.wait(timeout=10); nat.wait(timeout=10)
    if KEEP:
        print("logdir kept:", logdir)
    else:
        shutil.rmtree(logdir, ignore_errors=True)
    print("PARITY FAILURES:", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
