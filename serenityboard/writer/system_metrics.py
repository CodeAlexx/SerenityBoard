"""Background thread that polls GPU and system metrics into a SummaryWriter."""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serenityboard.writer.summary_writer import SummaryWriter

__all__ = ["SystemMetricsCollector"]

logger = logging.getLogger(__name__)

_MIN_INTERVAL = 5.0  # GPU queries have overhead; never poll faster than this


# ---------------------------------------------------------------------------
# GPU backends (pynvml -> nvidia-smi subprocess -> None)
# ---------------------------------------------------------------------------

class _GpuBackend:
    """Abstract interface for GPU metric collection."""

    def query(self, gpu_index: int) -> dict[str, float] | None:
        """Return dict with utilization, temperature, memory_used, memory_total
        (all floats) or *None* if the query fails."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Release resources held by the backend."""


class _PynvmlBackend(_GpuBackend):
    """GPU metrics via the pynvml C bindings (fastest)."""

    def __init__(self) -> None:
        import pynvml  # noqa: F811

        pynvml.nvmlInit()
        self._nvml = pynvml

    def query(self, gpu_index: int) -> dict[str, float] | None:
        try:
            handle = self._nvml.nvmlDeviceGetHandleByIndex(gpu_index)
            util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
            temp = self._nvml.nvmlDeviceGetTemperature(
                handle, self._nvml.NVML_TEMPERATURE_GPU
            )
            mem = self._nvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                "gpu/utilization": float(util.gpu),
                "gpu/temperature": float(temp),
                "gpu/memory_used": mem.used / (1024 ** 3),
                "gpu/memory_total": mem.total / (1024 ** 3),
            }
        except Exception:
            return None

    def shutdown(self) -> None:
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass


class _NvidiaSmiBackend(_GpuBackend):
    """GPU metrics via ``nvidia-smi`` subprocess (fallback)."""

    def query(self, gpu_index: int) -> dict[str, float] | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={gpu_index}",
                    "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) != 4:
                return None
            util, temp, mem_used_mib, mem_total_mib = (float(p) for p in parts)
            return {
                "gpu/utilization": util,
                "gpu/temperature": temp,
                "gpu/memory_used": mem_used_mib / 1024,   # MiB -> GiB
                "gpu/memory_total": mem_total_mib / 1024,
            }
        except Exception:
            return None

    def shutdown(self) -> None:
        pass


def _create_gpu_backend() -> _GpuBackend | None:
    """Try pynvml first, then nvidia-smi, then give up."""
    try:
        return _PynvmlBackend()
    except Exception:
        pass

    # Quick check: is nvidia-smi on PATH?
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return _NvidiaSmiBackend()
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# System (CPU / RAM) backends (psutil -> /proc fallback)
# ---------------------------------------------------------------------------

def _query_system_psutil() -> dict[str, float]:
    import psutil

    vm = psutil.virtual_memory()
    return {
        "system/cpu_percent": psutil.cpu_percent(interval=None),
        "system/ram_used_gb": vm.used / (1024 ** 3),
        "system/ram_total_gb": vm.total / (1024 ** 3),
    }


def _query_system_proc() -> dict[str, float]:
    """Fallback: read /proc/meminfo + /proc/loadavg for Linux."""
    metrics: dict[str, float] = {}

    # CPU: approximate utilisation from 1-min loadavg / logical CPUs
    try:
        load1 = os.getloadavg()[0]
        ncpu = os.cpu_count() or 1
        metrics["system/cpu_percent"] = min(load1 / ncpu * 100.0, 100.0)
    except (OSError, AttributeError):
        pass

    # RAM from /proc/meminfo
    try:
        meminfo: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    meminfo[key] = int(parts[1])  # value in kB
        total_kb = meminfo.get("MemTotal", 0)
        avail_kb = meminfo.get("MemAvailable", 0)
        metrics["system/ram_total_gb"] = total_kb / (1024 ** 2)
        metrics["system/ram_used_gb"] = (total_kb - avail_kb) / (1024 ** 2)
    except Exception:
        pass

    return metrics


def _create_system_query():
    """Return a callable that produces system metrics dict."""
    try:
        import psutil  # noqa: F401

        # Warm up cpu_percent (first call always returns 0.0)
        psutil.cpu_percent(interval=None)
        return _query_system_psutil
    except ImportError:
        return _query_system_proc


# ---------------------------------------------------------------------------
# Public collector
# ---------------------------------------------------------------------------

class SystemMetricsCollector:
    """Daemon thread that periodically logs system/GPU metrics via a SummaryWriter.

    Parameters
    ----------
    writer:
        The ``SummaryWriter`` instance to call ``add_scalar`` on.
    interval_seconds:
        Seconds between polls.  Clamped to >= 5.
    gpu_index:
        CUDA device index to query (default 0).
    """

    def __init__(
        self,
        writer: SummaryWriter,
        interval_seconds: float = 10.0,
        gpu_index: int = 0,
    ) -> None:
        self._writer = writer
        self._interval = max(interval_seconds, _MIN_INTERVAL)
        self._gpu_index = gpu_index

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Backends (created lazily on the polling thread so import errors
        # don't crash the caller's thread)
        self._gpu_backend: _GpuBackend | None = None
        self._system_query = _create_system_query
        self._backends_initialized = False

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Start the polling daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="sb-system-metrics", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 2)
            self._thread = None
        if self._gpu_backend is not None:
            self._gpu_backend.shutdown()
            self._gpu_backend = None

    # -- internals ----------------------------------------------------------

    def _init_backends(self) -> None:
        """One-time backend initialisation (runs on the polling thread)."""
        if self._backends_initialized:
            return
        self._backends_initialized = True
        try:
            self._gpu_backend = _create_gpu_backend()
        except Exception:
            self._gpu_backend = None
        try:
            self._system_query = _create_system_query()
        except Exception:
            self._system_query = _query_system_proc

    def _run(self) -> None:
        """Main loop executed on the daemon thread."""
        self._init_backends()
        step = 0
        while not self._stop_event.is_set():
            try:
                self._poll_once(step)
            except Exception:
                logger.debug("system metrics poll error", exc_info=True)
            step += 1
            # Sleep in small increments so we react to stop quickly
            self._stop_event.wait(timeout=self._interval)

    def _poll_once(self, step: int) -> None:
        """Collect all available metrics and log them as scalars."""
        # GPU metrics
        if self._gpu_backend is not None:
            gpu = self._gpu_backend.query(self._gpu_index)
            if gpu:
                for tag, value in gpu.items():
                    try:
                        self._writer.add_scalar(tag, value, step=step)
                    except Exception:
                        logger.debug(
                            "failed to log GPU metric %s", tag, exc_info=True
                        )

        # System metrics
        try:
            sys_metrics = self._system_query()
        except Exception:
            sys_metrics = {}
        for tag, value in sys_metrics.items():
            try:
                self._writer.add_scalar(tag, value, step=step)
            except Exception:
                logger.debug(
                    "failed to log system metric %s", tag, exc_info=True
                )
