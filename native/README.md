# SerenityBoard native backend (C++)

A C++ port of the SerenityBoard backend for use with the Diffusion Compiler:
the same `board.db` schema, the same HTTP/WebSocket API, the same JS frontend —
no Python at runtime. The Python package in `../serenityboard` stays the oracle;
`scripts/route_parity.py` diffs both servers on one logdir.

```
native/
  include/serenityboard/   public headers (+ serenityboard.h, a C ABI for C/C++ tools)
  src/core/                sbcore: schema v2 + migrations, blob store, PNG (zlib) / WAV,
                           NumPy-compatible histogram + PR-curve packing, reservoirs,
                           session guard, async SummaryWriter, system metrics, C API
  src/server/              sbserver: own HTTP/1.1 + WebSocket server, data provider,
                           run manager, live updates (1 s poll), all /api routes, static UI
  src/ingest/              Diffusion Compiler adapter (step lines, --report JSON, difbench /
                           difregress / difprobe / runtime-trace documents, training-report
                           JSON lines, losses.diftensor)
  tools/                   serenityboardd (serve), sbingest
  tests/                   sb_core_tests (C++), tests/py/cross_read.py (Python reads C++ db)
  scripts/                 route_parity.py (oracle diff), ui_drive.mjs + ui_check*.js (headless UI)
  third_party/             sqlite 3.46 amalgamation (public domain), nlohmann/json 3.11.3 (MIT)
```

## Build

```bash
cmake -S native -B native/build -G Ninja -DCMAKE_BUILD_TYPE=Release   # C++20, -Werror
ninja -C native/build
native/build/sb_core_tests
```

## Serve

```bash
native/build/serenityboardd serve --logdir /path/to/runs --port 6006 [--host 0.0.0.0] [--frontend DIR]
```

The frontend is found automatically at `../serenityboard/frontend` relative to the
binary's repo. Runs are discovered exactly as the Python server does (any directory
with `board.db`, depth <= 4, nested names joined with `__`).

## Log from C/C++ (the compiler's tools)

```c
#include "serenityboard/serenityboard.h"
sb_writer *w = sb_writer_open("/runs", "krea2-lora-01", "{\"lr\":1e-4}", -1, 1);
sb_add_scalar(w, "loss/train", loss, step);
sb_add_trace(w, step, "forward", forward_ms, NULL);
sb_writer_close(w);
```

Link `libsbcore.a` (+ `libsb_sqlite.a`, zlib, pthread). C++ callers use
`sb::SummaryWriter` directly (`include/serenityboard/summary_writer.hpp`).

## Ingest what the compiler already writes

```bash
sbingest --logdir /runs --run h3-t2va-1 steps:runner.log            # H3_STEP / FLUX2_NATIVE_STEP / KREA2_NATIVE_STEP / KREA2_STEP
sbingest --logdir /runs --run krea2-1 report:sampler-report.json    # difkrea2sample / difflux2sample reports
sbingest --logdir /runs --run bench-1 report:difbench.json          # benchmark / regression-report / device-probe / runtime-trace documents
sbingest --logdir /runs --run train-1 jsonl:training-report.jsonl   # TrainingStepReport lines (exp branch) or DIF_TRACE_FILE sinks
sbingest --logdir /runs --run train-1 losses:losses.diftensor       # diftrain F32 loss vector
```

## Contract fidelity

- Schema: v2 DDL verbatim, V1→V2→V3→V4 migrations, WAL/synchronous NORMAL/busy_timeout 5000,
  JSON-encoded metadata, `run_notes` created lazily by the server.
- Writer: single writer thread, INSERT OR REPLACE batches, 0.1/0.5/2.0 s retries on
  busy-class errors then a sticky error with the Python message, per-tag reservoirs (seed 0,
  always keep last) drained on flush/close, session guard (resume purge + orphan blob removal,
  crashed/complete), rank gate (`SB_RANK`/`RANK`/`LOCAL_RANK`), system-metrics thread.
- Server: every route in `serenityboard/server/routes/*` and `app.py`, the error envelope,
  no-cache headers, CSV/JSON export, distributions basis points, audio waveform/spectrogram/
  peak/rms (NumPy float32 pairwise mean reproduced), meshes/embeddings single-object form,
  notes upsert, `/ws/live` subscribe/scalar/trace/eval/session_changed with fnmatch globs.

## Not ported (returns the Python fallback behaviour)

- `/api/lora/*` → 503 "LoRA analytics not available" (needs safetensors + SVD; planned).
- Ops routers (`/api/tables|registry|sweeps|automations`) → 404 (opt-in in Python too).
- Writer: `add_video` (mp4/gif encoding), `add_graph(nn.Module)` tracing (use
  `add_graph_json`), embedding sprite sheets, TensorBoard-compat class (C++ callers use
  the native API).
- Reservoir sampling uses `std::mt19937`, so the *sampled subset* can differ from the
  Python writer for identical streams; capacities and semantics are identical.
