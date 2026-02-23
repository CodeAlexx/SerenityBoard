"""High-level logging API for SerenityBoard."""
from __future__ import annotations

import io
import json
import logging
import os
import queue
import threading
import time
import uuid

import numpy as np
from PIL import Image

from serenityboard.writer.async_writer import WriterError, WriteItem, _WriterThread
from serenityboard.writer.blob_storage import BlobStorage
from serenityboard.writer.reservoir import Reservoir
from serenityboard.writer.system_metrics import SystemMetricsCollector

__all__ = ["SummaryWriter", "WriterError"]

logger = logging.getLogger(__name__)

# Default reservoir sizes per data class.
# 0 = unbounded (lossless): every item goes straight to the queue.
_DEFAULT_RESERVOIR = {
    "scalars": 0,
    "text_events": 0,
    "trace_events": 0,
    "tensors": 500,
    "artifacts": 100,
    "audio": 100,
    "plugin_data": 500,
    "pr_curves": 100,
    "graphs": 10,
    "embeddings": 20,
    "meshes": 50,
}

_VALID_IMAGE_FORMATS = {"CHW", "HWC", "HW"}
_VALID_BATCH_IMAGE_FORMATS = {"NCHW", "NHWC", "NHW"}

# Sprite sheet size limits for add_embedding().
_MAX_SPRITE_COUNT = 10_000
_SPRITE_WARN_THRESHOLD = 5_000
_MAX_SPRITE_SHEET_DIM = 4096


class SummaryWriter:
    """Write training metrics, images, and text to a SerenityBoard run directory.

    Only rank 0 writes; all other ranks become silent no-ops.
    """

    def __init__(
        self,
        logdir: str,
        run_name: str | None = None,
        hparams: dict | None = None,
        resume_step: int | None = None,
        max_queue_size: int = 1000,
        flush_interval_secs: float = 2.0,
        flush_interval_items: int = 100,  # not used in V1 (timer-based only)
        max_retries_on_disk_error: int = 3,
        reservoir_config: dict | None = None,
        rank: int | None = None,
        system_metrics: bool = True,
    ) -> None:
        # Rank detection: explicit arg > SB_RANK > RANK > LOCAL_RANK > 0
        if rank is None:
            for var in ("SB_RANK", "RANK", "LOCAL_RANK"):
                val = os.environ.get(var)
                if val is not None:
                    rank = int(val)
                    break
            else:
                rank = 0
        self._rank = rank

        if rank != 0:
            logger.warning(
                "SerenityBoard: rank=%d, logging disabled. Only rank 0 writes. "
                "Set rank=0 explicitly if this is incorrect.",
                rank,
            )
            self._noop = True
            return

        self._noop = False
        self._closed = False

        # Run directory setup
        if run_name is None:
            from datetime import datetime, timezone

            run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._run_name = run_name
        self._run_dir = os.path.join(logdir, run_name)
        os.makedirs(self._run_dir, exist_ok=True)

        self._db_path = os.path.join(self._run_dir, "board.db")
        self._session_id = str(uuid.uuid4())

        # Reservoir setup
        reservoir_sizes = dict(_DEFAULT_RESERVOIR)
        if reservoir_config:
            reservoir_sizes.update(reservoir_config)
        self._reservoirs = {
            name: Reservoir(max_size=size)
            for name, size in reservoir_sizes.items()
        }

        # Blob storage
        self._blob_storage = BlobStorage(os.path.join(self._run_dir, "blobs"))

        # Queue and writer thread
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._flush_event = threading.Event()
        self._ready_event = threading.Event()

        self._writer = _WriterThread(
            db_path=self._db_path,
            session_id=self._session_id,
            resume_step=resume_step,
            q=self._queue,
            flush_event=self._flush_event,
            flush_interval=flush_interval_secs,
            blob_storage=self._blob_storage,
            ready_event=self._ready_event,
        )
        self._writer.MAX_RETRIES = max_retries_on_disk_error
        self._writer.start()
        self._ready_event.wait()

        # Propagate init errors (e.g. ValueError from SessionGuard)
        if self._writer.init_error is not None:
            raise self._writer.init_error

        # Write hparams to metadata if provided
        if hparams:
            self._queue.put(
                WriteItem(
                    table="metadata",
                    params=("hparams", json.dumps(hparams)),
                    step=None,
                )
            )

        # Write run metadata
        self._queue.put(
            WriteItem(
                table="metadata",
                params=("run_name", json.dumps(run_name)),
                step=None,
            )
        )
        self._queue.put(
            WriteItem(
                table="metadata",
                params=("start_time", json.dumps(time.time())),
                step=None,
            )
        )
        self._queue.put(
            WriteItem(
                table="metadata",
                params=("schema_version", json.dumps("2")),
                step=None,
            )
        )

        # System / GPU metrics collector
        self._sys_metrics: SystemMetricsCollector | None = None
        if system_metrics:
            self._sys_metrics = SystemMetricsCollector(writer=self)
            self._sys_metrics.start()

    # ── reservoir helpers ─────────────────────────────────────────────

    def _enqueue(self, data_class: str, tag: str, item: WriteItem) -> None:
        """Route item through reservoir (lossy) or directly to queue (lossless)."""
        reservoir = self._reservoirs.get(data_class)
        if reservoir is not None and data_class in {"tensors", "artifacts", "audio", "plugin_data", "pr_curves", "graphs", "embeddings", "meshes"}:
            # Lossy mode: add to reservoir, flushed later
            reservoir.add(tag, item)
        else:
            # Lossless mode: enqueue directly
            self._queue.put(item)

    def _drain_reservoirs(self) -> None:
        """Drain all reservoir contents into the write queue."""
        for name, reservoir in self._reservoirs.items():
            if name in {"scalars", "text_events", "trace_events"}:
                continue
            for key in reservoir.keys:
                items = reservoir.drain_items(key)
                for item in items:
                    self._queue.put(item)

    # ── error propagation ─────────────────────────────────────────────

    def _check_error(self) -> None:
        if self._noop:
            return
        err = self._writer.sticky_error
        if err is not None:
            raise err

    # ── public API ────────────────────────────────────────────────────

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        if self._noop:
            return
        self._check_error()
        wt = time.time()
        item = WriteItem(
            table="scalars", params=(tag, step, wt, float(value)), step=step
        )
        self._enqueue("scalars", tag, item)

    def add_scalars(
        self, main_tag: str, values: dict[str, float], step: int
    ) -> None:
        """Log multiple scalars under ``main_tag/sub_tag``."""
        if self._noop:
            return
        for sub_tag, val in values.items():
            self.add_scalar(f"{main_tag}/{sub_tag}", val, step)

    def add_image(
        self, tag: str, img: object, step: int, dataformats: str = "CHW"
    ) -> None:
        """Log an image (numpy array or PIL Image)."""
        if self._noop:
            return
        self._check_error()

        if dataformats not in _VALID_IMAGE_FORMATS:
            raise ValueError(
                f"Unsupported dataformats: {dataformats!r}. Use 'CHW', 'HWC', or 'HW'."
            )

        img_array = np.array(img)
        if dataformats == "CHW":
            img_array = np.transpose(img_array, (1, 2, 0))  # CHW -> HWC
        # HWC and HW are already in the right layout

        # float [0, 1] -> uint8
        if img_array.dtype in (np.float32, np.float64, np.float16):
            img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)

        # Build PIL image (avoid deprecated mode= kwarg removed in Pillow 13)
        if img_array.ndim == 2:
            pil_img = Image.fromarray(img_array.astype("uint8"))
        elif img_array.shape[2] == 1:
            pil_img = Image.fromarray(img_array[:, :, 0].astype("uint8"))
        elif img_array.shape[2] in (3, 4):
            pil_img = Image.fromarray(img_array.astype("uint8"))
        else:
            raise ValueError(f"Unsupported channel count: {img_array.shape[2]}")

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        blob_key = self._blob_storage.store(png_bytes, "png")
        w, h = pil_img.size

        wt = time.time()
        item = WriteItem(
            table="artifacts",
            params=(tag, step, 0, wt, "image", "image/png", blob_key, w, h, "{}"),
            step=step,
        )
        self._enqueue("artifacts", tag, item)

    def add_histogram(
        self, tag: str, values: object, step: int, bins: int = 64
    ) -> None:
        """Log a histogram of values."""
        if self._noop:
            return
        self._check_error()

        arr = np.asarray(values, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        counts, bin_edges = np.histogram(arr, bins=bins)
        # Shape: (bins, 3) with columns [left_edge, right_edge, count]
        hist = np.stack(
            [bin_edges[:-1], bin_edges[1:], counts.astype(np.float64)], axis=1
        )
        dtype = "float64"
        shape_json = json.dumps(list(hist.shape))
        data_bytes = hist.tobytes()

        wt = time.time()
        item = WriteItem(
            table="tensors",
            params=(tag, step, wt, dtype, shape_json, data_bytes),
            step=step,
        )
        self._enqueue("tensors", tag, item)

    def add_images(
        self, tag: str, imgs: object, step: int, dataformats: str = "NCHW"
    ) -> None:
        """Log a batch of images."""
        if self._noop:
            return
        self._check_error()

        if dataformats not in _VALID_BATCH_IMAGE_FORMATS:
            raise ValueError(
                f"Unsupported dataformats: {dataformats!r}. "
                f"Use one of {sorted(_VALID_BATCH_IMAGE_FORMATS)}."
            )

        arr = np.asarray(imgs)
        # Map batch format to single-image format
        single_fmt = dataformats[1:]  # NCHW->CHW, NHWC->HWC, NHW->HW
        for i in range(arr.shape[0]):
            self.add_image(f"{tag}/{i}", arr[i], step, dataformats=single_fmt)

    def add_text(self, tag: str, text: str, step: int) -> None:
        """Log a text string."""
        if self._noop:
            return
        self._check_error()
        wt = time.time()
        item = WriteItem(table="text_events", params=(tag, step, wt, text), step=step)
        self._enqueue("text_events", tag, item)

    def add_hparams(self, hparam_dict: dict, metric_dict: dict) -> None:
        """Log hyper-parameters and their associated metrics."""
        if self._noop:
            return
        self._check_error()
        self._queue.put(
            WriteItem(
                table="metadata",
                params=("hparams", json.dumps(hparam_dict)),
                step=None,
            )
        )
        wt = time.time()
        for metric_tag, value in metric_dict.items():
            self._queue.put(
                WriteItem(
                    table="hparam_metrics",
                    params=(metric_tag, float(value), None, wt),
                    step=None,
                )
            )

    def add_trace(
        self, step: int, phase: str, duration_ms: float, details: dict | None = None
    ) -> None:
        """Log a trace event (e.g. forward, backward, offload_wait)."""
        if self._noop:
            return
        self._check_error()
        wt = time.time()
        details_json = json.dumps(details) if details else "{}"
        item = WriteItem(
            table="trace_events",
            params=(step, wt, phase, float(duration_ms), details_json),
            step=step,
        )
        self._enqueue("trace_events", tag=f"trace/{phase}", item=item)

    def add_eval(
        self,
        suite_name: str,
        case_id: str,
        step: int,
        score_name: str,
        score_value: float,
        artifact: str | None = None,
        details: dict | None = None,
    ) -> None:
        """Log an evaluation result."""
        if self._noop:
            return
        self._check_error()
        wt = time.time()
        details_json = json.dumps(details) if details else "{}"
        item = WriteItem(
            table="eval_results",
            params=(suite_name, case_id, step, wt, score_name, float(score_value), artifact, details_json),
            step=step,
        )
        self._queue.put(item)

    def add_custom_scalars_layout(self, layout: dict) -> None:
        """Store a custom scalars layout configuration.

        Layout format:
        {"categories": [{"title": "Losses", "charts": [{"title": "Train vs Val", "tags": ["loss/train", "loss/val"]}]}]}
        """
        if self._noop:
            return
        self._check_error()
        layout_json = json.dumps(layout)
        item = WriteItem(
            table="custom_scalar_layouts",
            params=("default", layout_json),
            step=None,
        )
        self._queue.put(item)

    def add_pr_curve(
        self, tag: str, labels: object, predictions: object, step: int,
        num_thresholds: int = 201, class_index: int = 0,
    ) -> None:
        """Log a precision-recall curve computed from labels and predictions."""
        if self._noop:
            return
        self._check_error()

        labels_arr = np.asarray(labels, dtype=np.float64).ravel()
        preds_arr = np.asarray(predictions, dtype=np.float64).ravel()
        if labels_arr.shape != preds_arr.shape:
            raise ValueError("labels and predictions must have the same length")

        thresholds = np.linspace(0.0, 1.0, num_thresholds)
        tp = np.zeros(num_thresholds, dtype=np.float64)
        fp = np.zeros(num_thresholds, dtype=np.float64)
        tn = np.zeros(num_thresholds, dtype=np.float64)
        fn = np.zeros(num_thresholds, dtype=np.float64)
        precision = np.zeros(num_thresholds, dtype=np.float64)
        recall = np.zeros(num_thresholds, dtype=np.float64)

        for i, thresh in enumerate(thresholds):
            predicted_pos = preds_arr >= thresh
            actual_pos = labels_arr >= 0.5
            tp[i] = np.sum(predicted_pos & actual_pos)
            fp[i] = np.sum(predicted_pos & ~actual_pos)
            tn[i] = np.sum(~predicted_pos & ~actual_pos)
            fn[i] = np.sum(~predicted_pos & actual_pos)
            denom_p = tp[i] + fp[i]
            precision[i] = tp[i] / denom_p if denom_p > 0 else 1.0
            denom_r = tp[i] + fn[i]
            recall[i] = tp[i] / denom_r if denom_r > 0 else 0.0

        # Pack as [6, num_thresholds] float64 array
        data = np.stack([tp, fp, tn, fn, precision, recall], axis=0)
        data_bytes = data.tobytes()

        wt = time.time()
        item = WriteItem(
            table="pr_curves",
            params=(tag, step, class_index, wt, num_thresholds, data_bytes),
            step=step,
        )
        self._enqueue("pr_curves", tag, item)

    def add_audio(
        self, tag: str, audio_data: object, step: int,
        sample_rate: int = 44100, num_channels: int = 1,
    ) -> None:
        """Log an audio clip (numpy array or raw WAV bytes)."""
        if self._noop:
            return
        self._check_error()

        import wave as _wave

        arr = np.asarray(audio_data)
        if arr.dtype in (np.float32, np.float64, np.float16):
            arr = np.clip(arr, -1.0, 1.0)
            arr = (arr * 32767).astype(np.int16)
        elif arr.dtype != np.int16:
            arr = arr.astype(np.int16)

        if arr.ndim == 1:
            actual_channels = 1
        elif arr.ndim == 2:
            actual_channels = arr.shape[1]
            arr = arr.ravel()
        else:
            raise ValueError(f"Audio array must be 1D or 2D, got shape {arr.shape}")

        buf = io.BytesIO()
        with _wave.open(buf, 'wb') as wf:
            wf.setnchannels(actual_channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(arr.tobytes())

        wav_bytes = buf.getvalue()
        duration_ms = (len(arr) / actual_channels / sample_rate) * 1000.0

        blob_key = self._blob_storage.store(wav_bytes, "wav")
        wt = time.time()
        item = WriteItem(
            table="audio",
            params=(tag, step, 0, wt, blob_key, sample_rate, actual_channels, duration_ms, "audio/wav", ""),
            step=step,
        )
        self._enqueue("audio", tag, item)

    def add_graph(
        self,
        model: object,
        input_to_model: object | None = None,
        verbose: bool = False,
        use_strict_trace: bool = True,
        tag: str = "default",
        step: int = 0,
    ) -> None:
        """Log a PyTorch model computation graph.

        Parameters
        ----------
        model:
            A ``torch.nn.Module`` to trace.
        input_to_model:
            Example input tensor(s) for ``torch.jit.trace``.  Required unless
            the model has already been scripted.
        verbose:
            If True, include additional node attributes in the graph data.
        use_strict_trace:
            Passed to ``torch.jit.trace(..., strict=...)``.
        tag:
            Tag to identify the graph (default ``"default"``).
        step:
            Global step value (default ``0``).
        """
        if self._noop:
            return
        self._check_error()

        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError(
                "PyTorch is required for add_graph(). "
                "Install it with: pip install torch"
            )

        if not isinstance(model, nn.Module):
            raise TypeError(
                f"model must be a torch.nn.Module, got {type(model).__name__}"
            )

        # Trace the model to capture the computation graph
        try:
            with torch.no_grad():
                if input_to_model is not None:
                    traced = torch.jit.trace(
                        model, input_to_model, strict=use_strict_trace
                    )
                else:
                    # Try scripting if no input is provided
                    traced = torch.jit.script(model)
        except Exception as exc:
            raise ValueError(
                f"Failed to trace/script the model: {exc}. "
                "Provide input_to_model for torch.jit.trace, or ensure "
                "the model is scriptable."
            ) from exc

        # Extract graph structure from the traced model.
        # Use inlined_graph because it carries scopeName() on each node
        # (e.g. "__module.block1/__module.block1.fc"), which we need to
        # build the collapsible hierarchy in the frontend.
        # traced.graph (non-inlined) only has prim::GetAttr/CallMethod with
        # empty scopeName(), so it cannot provide module hierarchy info.
        graph = traced.inlined_graph

        # Build a mapping from JIT scope names to module class names.
        # named_modules() yields ('layer1', Conv2d(...)), ('layer1.bn', BatchNorm2d(...)), etc.
        module_map: dict[str, str] = {}
        try:
            for name, mod in model.named_modules():
                module_map[name] = type(mod).__name__
        except Exception:
            pass  # Some models may not support named_modules

        nodes = []
        for node in graph.nodes():
            node_info: dict = {
                "name": node.kind(),
                "op": node.kind(),
                "scope": node.scopeName() if hasattr(node, "scopeName") else "",
                "inputs": [inp.debugName() for inp in node.inputs()],
                "outputs": [out.debugName() for out in node.outputs()],
            }
            # Extract output shapes and dtypes when available
            output_shapes = []
            output_dtypes = []
            for out in node.outputs():
                t = out.type()
                try:
                    if hasattr(t, "sizes") and t.sizes() is not None:
                        output_shapes.append(list(t.sizes()))
                    else:
                        output_shapes.append(None)
                except Exception:
                    output_shapes.append(None)
                try:
                    if hasattr(t, "dtype"):
                        output_dtypes.append(str(t.dtype()))
                    else:
                        output_dtypes.append(None)
                except Exception:
                    output_dtypes.append(None)
            node_info["shapes"] = output_shapes
            node_info["dtypes"] = output_dtypes

            # Include attribute key-value pairs
            attrs = {}
            for attr_name in node.attributeNames():
                try:
                    attrs[attr_name] = str(node[attr_name])
                except Exception:
                    attrs[attr_name] = "<unavailable>"
            node_info["attributes"] = attrs

            nodes.append(node_info)

        # Build graph inputs/outputs from the graph signature
        graph_inputs = []
        for inp in graph.inputs():
            info: dict = {"name": inp.debugName()}
            t = inp.type()
            try:
                if hasattr(t, "sizes") and t.sizes() is not None:
                    info["shape"] = list(t.sizes())
            except Exception:
                pass
            try:
                if hasattr(t, "dtype"):
                    info["dtype"] = str(t.dtype())
            except Exception:
                pass
            graph_inputs.append(info)

        graph_outputs = []
        for out in graph.outputs():
            info = {"name": out.debugName()}
            t = out.type()
            try:
                if hasattr(t, "sizes") and t.sizes() is not None:
                    info["shape"] = list(t.sizes())
            except Exception:
                pass
            try:
                if hasattr(t, "dtype"):
                    info["dtype"] = str(t.dtype())
            except Exception:
                pass
            graph_outputs.append(info)

        # Collect unique scope prefixes so the frontend can build collapsible groups.
        scopes: set[str] = set()
        for n in nodes:
            scope = n.get("scope", "")
            if scope:
                parts = scope.split("/")
                for i in range(1, len(parts) + 1):
                    scopes.add("/".join(parts[:i]))

        graph_data = {
            "nodes": nodes,
            "inputs": graph_inputs,
            "outputs": graph_outputs,
            "model_name": type(model).__name__,
            "scopes": sorted(scopes),
            "module_map": module_map,
        }
        graph_json = json.dumps(graph_data)
        graph_blob_key = self._blob_storage.store(graph_json.encode("utf-8"), "json")

        wt = time.time()
        item = WriteItem(
            table="graphs",
            params=(tag, step, wt, graph_blob_key),
            step=step,
        )
        self._enqueue("graphs", tag, item)

    def add_embedding(
        self,
        mat: object,
        metadata: object | None = None,
        label_img: object | None = None,
        global_step: int | None = None,
        tag: str = "default",
        metadata_header: list[str] | None = None,
    ) -> None:
        """Log an embedding matrix for projector visualization.

        Parameters
        ----------
        mat:
            Embedding matrix of shape (N, D) as numpy array or torch tensor.
            N is the number of points, D is the dimensionality.
        metadata:
            Per-point labels. A list of length N (single column) or a list of
            lists for multi-column metadata.
        label_img:
            Optional sprite images for each point, shape (N, C, H, W) as
            numpy array or torch tensor.
        global_step:
            Step number (defaults to 0).
        tag:
            Tag name for the embedding.
        metadata_header:
            Column headers for multi-column metadata.
        """
        if self._noop:
            return
        self._check_error()

        step = global_step if global_step is not None else 0

        # Convert torch tensor to numpy if needed
        try:
            import torch
            if isinstance(mat, torch.Tensor):
                mat = mat.detach().cpu().numpy()
            if label_img is not None and isinstance(label_img, torch.Tensor):
                label_img = label_img.detach().cpu().numpy()
        except ImportError:
            pass  # torch not available, assume numpy

        mat_arr = np.asarray(mat, dtype=np.float32)
        if mat_arr.ndim != 2:
            raise ValueError(
                f"Embedding matrix must be 2D (N x D), got shape {mat_arr.shape}"
            )
        n, d = mat_arr.shape

        # Validate metadata length
        if metadata is not None:
            if len(metadata) != n:
                raise ValueError(
                    f"metadata length ({len(metadata)}) must match number of "
                    f"embedding points ({n})"
                )

        # Convert metadata to a plain list for JSON serialization
        if metadata is not None:
            if hasattr(metadata, 'tolist'):  # numpy array or torch tensor
                metadata = metadata.tolist()
            elif not isinstance(metadata, list):
                metadata = list(metadata)

        # --- Size cap: subsample embeddings, metadata, AND sprites together ---
        if n > _MAX_SPRITE_COUNT:
            indices = np.linspace(0, n - 1, _MAX_SPRITE_COUNT, dtype=int)
            mat_arr = mat_arr[indices]
            if metadata is not None:
                metadata = [metadata[i] for i in indices]
            if label_img is not None:
                label_img = np.asarray(label_img)[indices]
            logger.warning(
                "add_embedding: subsampled from %d to %d points "
                "(embedding matrix, metadata, and sprites all subsampled)",
                n,
                _MAX_SPRITE_COUNT,
            )
            n = _MAX_SPRITE_COUNT
        elif n > _SPRITE_WARN_THRESHOLD:
            logger.warning(
                "add_embedding: %d points requested; sprite sheets "
                "above %d may be very large",
                n,
                _SPRITE_WARN_THRESHOLD,
            )

        # Serialize embedding matrix as float32 binary blob
        mat_bytes = mat_arr.tobytes()
        blob_key = self._blob_storage.store(mat_bytes, "emb")

        # Serialize metadata as JSON
        metadata_json = json.dumps(metadata if metadata is not None else [])
        header_json = json.dumps(metadata_header if metadata_header is not None else [])

        # Handle sprite images
        sprite_blob_key = None
        sprite_h = None
        sprite_w = None
        if label_img is not None:
            img_arr = np.asarray(label_img)
            if img_arr.ndim != 4:
                raise ValueError(
                    f"label_img must be 4D (N, C, H, W), got shape {img_arr.shape}"
                )
            if img_arr.shape[0] != n:
                raise ValueError(
                    f"label_img batch size ({img_arr.shape[0]}) must match "
                    f"number of embedding points ({n})"
                )
            # img_arr is (N, C, H, W) - convert each to HWC for PIL
            _, c, h, w = img_arr.shape
            sprite_h = h
            sprite_w = w

            # float [0, 1] -> uint8
            if img_arr.dtype in (np.float32, np.float64, np.float16):
                img_arr = (np.clip(img_arr, 0, 1) * 255).astype(np.uint8)

            # Build sprite sheet: tile images into a grid
            import math
            sprite_n = img_arr.shape[0]
            if sprite_n > 0:
                cols = int(math.ceil(math.sqrt(sprite_n)))
                rows = int(math.ceil(sprite_n / cols))

                sheet_width = cols * w
                sheet_height = rows * h
                if sheet_width > _MAX_SPRITE_SHEET_DIM or sheet_height > _MAX_SPRITE_SHEET_DIM:
                    logger.warning(
                        "add_embedding: sprite sheet dimensions %dx%d exceed "
                        "%d; consider smaller sprites or fewer points",
                        sheet_width,
                        sheet_height,
                        _MAX_SPRITE_SHEET_DIM,
                    )
                # Transpose from NCHW to NHWC
                img_arr = np.transpose(img_arr, (0, 2, 3, 1))  # (N, H, W, C)

                if c == 1:
                    sprite_sheet = np.zeros((rows * h, cols * w), dtype=np.uint8)
                    for i in range(sprite_n):
                        r, col_idx = divmod(i, cols)
                        sprite_sheet[r * h:(r + 1) * h, col_idx * w:(col_idx + 1) * w] = img_arr[i, :, :, 0]
                    pil_sprite = Image.fromarray(sprite_sheet, mode="L")
                elif c == 3:
                    sprite_sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
                    for i in range(sprite_n):
                        r, col_idx = divmod(i, cols)
                        sprite_sheet[r * h:(r + 1) * h, col_idx * w:(col_idx + 1) * w, :] = img_arr[i]
                    pil_sprite = Image.fromarray(sprite_sheet, mode="RGB")
                elif c == 4:
                    sprite_sheet = np.zeros((rows * h, cols * w, 4), dtype=np.uint8)
                    for i in range(sprite_n):
                        r, col_idx = divmod(i, cols)
                        sprite_sheet[r * h:(r + 1) * h, col_idx * w:(col_idx + 1) * w, :] = img_arr[i]
                    pil_sprite = Image.fromarray(sprite_sheet, mode="RGBA")
                else:
                    raise ValueError(f"Unsupported channel count for sprites: {c}")

                buf = io.BytesIO()
                pil_sprite.save(buf, format="PNG")
                sprite_bytes = buf.getvalue()
                sprite_blob_key = self._blob_storage.store(sprite_bytes, "png")

        wt = time.time()
        item = WriteItem(
            table="embeddings",
            params=(
                tag, step, wt, n, d, blob_key,
                metadata_json, header_json,
                sprite_blob_key, sprite_h, sprite_w,
            ),
            step=step,
        )
        self._enqueue("embeddings", tag, item)

    def add_pr_curve_raw(
        self, tag: str, true_positive_counts: object,
        false_positive_counts: object, true_negative_counts: object,
        false_negative_counts: object, precision: object, recall: object,
        step: int, class_index: int = 0,
    ) -> None:
        """Log a pre-computed precision-recall curve."""
        if self._noop:
            return
        self._check_error()

        tp = np.asarray(true_positive_counts, dtype=np.float64).ravel()
        fp = np.asarray(false_positive_counts, dtype=np.float64).ravel()
        tn = np.asarray(true_negative_counts, dtype=np.float64).ravel()
        fn = np.asarray(false_negative_counts, dtype=np.float64).ravel()
        prec = np.asarray(precision, dtype=np.float64).ravel()
        rec = np.asarray(recall, dtype=np.float64).ravel()

        num_thresholds = len(tp)
        data = np.stack([tp, fp, tn, fn, prec, rec], axis=0)
        data_bytes = data.tobytes()

        wt = time.time()
        item = WriteItem(
            table="pr_curves",
            params=(tag, step, class_index, wt, num_thresholds, data_bytes),
            step=step,
        )
        self._enqueue("pr_curves", tag, item)

    def add_plugin_data(
        self, plugin_name: str, tag: str, data: dict, step: int
    ) -> None:
        """Log arbitrary plugin data as JSON."""
        if self._noop:
            return
        self._check_error()
        wt = time.time()
        item = WriteItem(
            table="plugin_data",
            params=(plugin_name, tag, step, wt, json.dumps(data)),
            step=step,
        )
        self._enqueue("plugin_data", tag=f"{plugin_name}/{tag}", item=item)

    def add_mesh(
        self,
        tag: str,
        vertices: object,
        colors: object | None = None,
        faces: object | None = None,
        config_dict: dict | None = None,
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        """Log a 3D mesh with vertices, optional colors, and optional faces.

        Parameters
        ----------
        tag:
            Data identifier for the mesh.
        vertices:
            Vertex positions as numpy array or torch tensor of shape (N, 3) or
            (1, N, 3).  Batch dim is squeezed if B=1; B>1 is not supported.
        colors:
            Optional per-vertex RGB colors, shape (N, 3) or (1, N, 3), values
            0-255 (uint8).
        faces:
            Optional triangle face indices, shape (F, 3) or (1, F, 3), integer
            dtype.
        config_dict:
            Optional dict with camera/material configuration, stored as JSON.
        global_step:
            Step number.  Required.
        walltime:
            Override wall clock time (default: ``time.time()``).
        """
        if self._noop:
            return
        self._check_error()

        if global_step is None:
            global_step = 0

        # Convert torch tensors to numpy if needed
        try:
            import torch
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.detach().cpu().numpy()
            if colors is not None and isinstance(colors, torch.Tensor):
                colors = colors.detach().cpu().numpy()
            if faces is not None and isinstance(faces, torch.Tensor):
                faces = faces.detach().cpu().numpy()
        except ImportError:
            pass  # torch not available, assume numpy

        vertices_arr = np.asarray(vertices)

        # Squeeze batch dimension: (1, N, 3) -> (N, 3)
        if vertices_arr.ndim == 3:
            if vertices_arr.shape[0] > 1:
                import warnings
                warnings.warn(
                    f"Batched meshes (B={vertices_arr.shape[0]}) received; "
                    f"only the first item will be stored.",
                    stacklevel=2,
                )
            vertices_arr = vertices_arr[0]
        if vertices_arr.ndim != 2 or vertices_arr.shape[1] != 3:
            raise ValueError(
                f"vertices must have shape (N, 3), got {vertices_arr.shape}"
            )
        vertices_arr = vertices_arr.astype(np.float32)
        num_vertices = vertices_arr.shape[0]

        # Process colors
        colors_blob_key = None
        has_colors = 0
        if colors is not None:
            colors_arr = np.asarray(colors)
            if colors_arr.ndim == 3:
                if colors_arr.shape[0] > 1:
                    import warnings
                    warnings.warn(
                        f"Batched colors (B={colors_arr.shape[0]}) received; "
                        f"only the first item will be stored.",
                        stacklevel=2,
                    )
                colors_arr = colors_arr[0]
            if colors_arr.ndim != 2 or colors_arr.shape[1] != 3:
                raise ValueError(
                    f"colors must have shape (N, 3), got {colors_arr.shape}"
                )
            if colors_arr.shape[0] != num_vertices:
                raise ValueError(
                    f"colors vertex count ({colors_arr.shape[0]}) does not "
                    f"match vertices ({num_vertices})"
                )
            if colors_arr.dtype in (np.float32, np.float64, np.float16):
                colors_arr = (np.clip(colors_arr, 0, 1) * 255).astype(np.uint8)
            else:
                colors_arr = colors_arr.astype(np.uint8)
            colors_blob_key = self._blob_storage.store(colors_arr.tobytes(), "bin")
            has_colors = 1

        # Process faces
        faces_blob_key = None
        has_faces = 0
        if faces is not None:
            faces_arr = np.asarray(faces)
            if faces_arr.ndim == 3:
                if faces_arr.shape[0] > 1:
                    import warnings
                    warnings.warn(
                        f"Batched faces (B={faces_arr.shape[0]}) received; "
                        f"only the first item will be stored.",
                        stacklevel=2,
                    )
                faces_arr = faces_arr[0]
            if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
                raise ValueError(
                    f"faces must have shape (F, 3), got {faces_arr.shape}"
                )
            faces_arr = faces_arr.astype(np.int32)
            faces_blob_key = self._blob_storage.store(faces_arr.tobytes(), "bin")
            has_faces = 1
        num_faces = faces_arr.shape[0] if faces is not None else 0

        # Store vertices blob
        vertices_blob_key = self._blob_storage.store(vertices_arr.tobytes(), "bin")

        # Serialize config
        config_json = json.dumps(config_dict) if config_dict is not None else None

        wt = walltime if walltime is not None else time.time()
        step = global_step
        item = WriteItem(
            table="meshes",
            params=(
                tag, step, wt, num_vertices, has_faces, has_colors, num_faces,
                vertices_blob_key, faces_blob_key, colors_blob_key, config_json,
            ),
            step=step,
        )
        self._enqueue("meshes", tag, item)

    # ── lifecycle ─────────────────────────────────────────────────────

    def flush(self) -> None:
        """Block until all queued items have been committed."""
        if self._noop:
            return
        self._check_error()
        # Guard against deadlock: if writer thread is dead, queue.join() hangs
        if not self._writer.is_alive():
            self._check_error()
            return
        # Drain reservoir contents into queue before waiting
        self._drain_reservoirs()
        self._queue.join()
        self._check_error()

    def close(self) -> None:
        """Flush, mark the session complete, and shut down the writer thread."""
        if self._noop:
            return
        if self._closed:
            return
        self._closed = True
        try:
            # Stop system metrics collector first so its final writes are queued
            if self._sys_metrics is not None:
                self._sys_metrics.stop()
                self._sys_metrics = None
            # Drain reservoir contents before shutting down
            self._drain_reservoirs()
            # Drain pending items (may raise WriterError)
            if self._writer.is_alive():
                self._queue.join()
            # Mark session complete ON THE WRITER THREAD via queue sentinel
            if self._writer.is_alive():
                self._writer.request_mark_complete()
                self._queue.join()  # wait for mark_complete to execute
        finally:
            # Always send shutdown and join, even on error
            if self._writer.is_alive():
                self._queue.put(None)
                self._writer.join(timeout=10)
        self._check_error()

    def __enter__(self) -> SummaryWriter:
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        if exc_type is not None:
            # Don't mask the original exception with a WriterError
            try:
                self.close()
            except WriterError:
                logger.error("WriterError during close (original exception preserved)")
        else:
            self.close()
        return False
