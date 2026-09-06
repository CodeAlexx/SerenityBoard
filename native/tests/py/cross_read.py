"""Cross-language gate: the Python package reads a run written by the native writer.
usage: cross_read.py <logdir-with-run1>"""
import io, json, os, sys, wave
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from serenityboard.server.run_manager import RunWatcher
from serenityboard.server.data_provider import RunDataProvider
from PIL import Image

logdir = sys.argv[1]
w = RunWatcher(logdir); w.scan_once()
runs = {r["name"]: r for r in w.get_runs()}
assert "run1" in runs, runs.keys()
r = runs["run1"]
assert r["status"] in ("complete", "completed"), r
assert r["hparams"] == {"lr": 1e-4}, r["hparams"]  # add_hparams replaced the constructor hparams
assert r["last_step"] == 11, r  # run1 went through the resume test: purge >10, then step 11
print("runs ok:", {k: (v["status"], v["last_step"]) for k, v in runs.items()})
p = RunDataProvider(os.path.join(logdir, "run1", "board.db"))
tags = p.get_tags()
assert tags["scalars"] == ["loss/train", "lr/a", "lr/b"], tags["scalars"]
assert tags["tensors"] == ["weights"] and tags["artifacts"] == ["img"] and tags["text_events"] == ["notes"]
assert tags["audio"] == ["clip"] and tags["pr_curves"] == ["pr"] and tags["eval_suites"] == ["suite"]
assert tags["trace_events"] == ["forward"]
print("tags ok")
sc = p.read_scalars_downsampled("loss/train", 0)
assert len(sc) == 12 and sc[0][0] == 0 and sc[11][2] == 0.5 and abs(sc[10][2] - 1/11) < 1e-12, sc
h = p.read_histograms("weights", 100)
assert len(h) == 1 and len(h[0]["bins"]) == 8
counts, edges = np.histogram(np.array([0.1, 0.2, 0.3, 0.4, 5.0]), bins=8)
got = np.array(h[0]["bins"])
assert np.array_equal(got[:, 2], counts.astype(float)), (got[:, 2], counts)
assert np.allclose(got[:, 0], edges[:-1]) and np.allclose(got[:, 1], edges[1:]), (got, edges)
print("histogram bins == np.histogram:", counts.tolist())
d = p.read_distributions("weights", 100)
assert len(d) == 1 and len(d[0]["percentiles"]) == 9
imgs = p.read_artifacts("img", 100)
assert len(imgs) == 1 and imgs[0]["width"] == 2 and imgs[0]["height"] == 2
info = p.get_blob_info(imgs[0]["blob_key"])
png = Image.open(os.path.join(logdir, "run1", "blobs", imgs[0]["blob_key"]))
assert png.size == (2, 2) and png.mode == "RGB" and png.getpixel((0, 0)) == (1, 2, 3), (png.size, png.mode, png.getpixel((0,0)))
print("image blob decodes:", png.size, png.mode, info)
pr = p.read_pr_curves("pr", 50)
assert len(pr) == 1 and pr[0]["num_thresholds"] == 5 and len(pr[0]["precision"]) == 5, pr
au = p.read_audio("clip", 50)
assert len(au) == 1 and au[0]["sample_rate"] == 8000 and au[0]["num_channels"] == 1, au
assert au[0]["waveform"] is not None and au[0]["peak_db"] is not None, au[0]
with wave.open(os.path.join(logdir, "run1", "blobs", au[0]["blob_key"])) as wf:
    assert wf.getnchannels() == 1 and wf.getframerate() == 8000 and wf.getnframes() == 4
print("audio analysis ok: peak_db=%.2f" % au[0]["peak_db"])
tx = p.read_text("notes"); assert tx[0]["value"] == "hello"
tr = p.read_trace_events(); assert tr[0]["phase"] == "forward" and tr[0]["duration_ms"] == 12.5 and tr[0]["details"] == {"k": 1}
ev = p.read_eval_results("suite"); assert ev[0]["score_value"] == 0.75 and ev[0]["case_id"] == "case-1"
hp = p.get_hparams(); assert hp["metrics"] == {"final_loss": 0.05}, hp
last = p.read_scalars_last(["loss/train"]); assert last["loss/train"]["step"] == 11
info = p.get_run_info(); assert info["active_session_id"] and info["status"] == "complete", info
print("ALL CROSS-READ CHECKS PASS")
