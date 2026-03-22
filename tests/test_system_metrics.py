"""Tests for SystemMetricsCollector and GPU backends."""
from __future__ import annotations

import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from serenityboard.writer.system_metrics import (
    SystemMetricsCollector,
    _NvidiaSmiBackend,
    _PynvmlBackend,
    _create_gpu_backend,
    _create_system_query,
    _query_system_proc,
)


# ---------------------------------------------------------------------------
# System metrics
# ---------------------------------------------------------------------------

class TestSystemMetrics:
    """CPU and RAM metrics are logged as scalars."""

    def test_cpu_and_ram_logged(self) -> None:
        """SystemMetricsCollector logs system/cpu_percent and system/ram_* scalars."""
        writer = MagicMock()
        collector = SystemMetricsCollector(writer, interval_seconds=5.0, gpu_index=0)

        # Patch out GPU backend so only system metrics fire
        collector._gpu_backend = None
        collector._backends_initialized = True
        collector._system_query = _create_system_query()

        collector._poll_once(step=0)

        # Check that add_scalar was called with system tags
        tags_logged = {call.args[0] for call in writer.add_scalar.call_args_list}
        assert "system/cpu_percent" in tags_logged
        assert "system/ram_used_gb" in tags_logged
        assert "system/ram_total_gb" in tags_logged

    def test_system_metrics_values_reasonable(self) -> None:
        """CPU % >= 0, RAM > 0."""
        writer = MagicMock()
        collector = SystemMetricsCollector(writer, interval_seconds=5.0)
        collector._gpu_backend = None
        collector._backends_initialized = True
        collector._system_query = _create_system_query()

        collector._poll_once(step=0)

        values = {}
        for call in writer.add_scalar.call_args_list:
            values[call.args[0]] = call.args[1]

        assert values["system/cpu_percent"] >= 0.0
        assert values["system/ram_used_gb"] > 0.0
        assert values["system/ram_total_gb"] > 0.0


# ---------------------------------------------------------------------------
# GPU backend: pynvml
# ---------------------------------------------------------------------------

class TestPynvmlBackend:
    """When pynvml is available, GPU metrics are logged."""

    def test_gpu_metrics_via_pynvml(self) -> None:
        """Mock pynvml and verify GPU scalars are logged."""
        writer = MagicMock()
        collector = SystemMetricsCollector(writer, interval_seconds=5.0, gpu_index=0)
        collector._backends_initialized = True
        collector._system_query = lambda: {}  # skip system metrics

        # Create a mock GPU backend
        mock_backend = MagicMock()
        mock_backend.query.return_value = {
            "gpu/utilization": 75.0,
            "gpu/temperature": 65.0,
            "gpu/memory_used": 8.5,
            "gpu/memory_total": 24.0,
        }
        collector._gpu_backend = mock_backend

        collector._poll_once(step=0)

        tags_logged = {call.args[0] for call in writer.add_scalar.call_args_list}
        assert "gpu/utilization" in tags_logged
        assert "gpu/temperature" in tags_logged
        assert "gpu/memory_used" in tags_logged
        assert "gpu/memory_total" in tags_logged

        # Verify values
        values = {call.args[0]: call.args[1] for call in writer.add_scalar.call_args_list}
        assert values["gpu/utilization"] == 75.0
        assert values["gpu/memory_total"] == 24.0


# ---------------------------------------------------------------------------
# GPU backend: nvidia-smi fallback
# ---------------------------------------------------------------------------

class TestNvidiaSmiBackend:
    """When pynvml is not available, falls back to nvidia-smi subprocess."""

    def test_nvidia_smi_parsing(self) -> None:
        """Mock subprocess.run to simulate nvidia-smi output."""
        backend = _NvidiaSmiBackend()

        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "85, 72, 10240, 24576\n"

        with patch("subprocess.run", return_value=fake_result):
            result = backend.query(gpu_index=0)

        assert result is not None
        assert result["gpu/utilization"] == 85.0
        assert result["gpu/temperature"] == 72.0
        assert abs(result["gpu/memory_used"] - 10240 / 1024) < 0.01
        assert abs(result["gpu/memory_total"] - 24576 / 1024) < 0.01

    def test_nvidia_smi_failure_returns_none(self) -> None:
        """If nvidia-smi fails, query returns None (no crash)."""
        backend = _NvidiaSmiBackend()

        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stdout = ""

        with patch("subprocess.run", return_value=fake_result):
            result = backend.query(gpu_index=0)

        assert result is None


# ---------------------------------------------------------------------------
# No GPU available
# ---------------------------------------------------------------------------

class TestNoGpuAvailable:
    """When neither pynvml nor nvidia-smi is available, GPU metrics are skipped."""

    def test_no_gpu_no_crash(self) -> None:
        """With no GPU backend, only system metrics are logged."""
        writer = MagicMock()
        collector = SystemMetricsCollector(writer, interval_seconds=5.0)
        collector._gpu_backend = None
        collector._backends_initialized = True
        collector._system_query = _create_system_query()

        collector._poll_once(step=0)

        tags_logged = {call.args[0] for call in writer.add_scalar.call_args_list}
        # GPU tags should NOT be present
        assert not any(t.startswith("gpu/") for t in tags_logged)
        # System tags should still be present
        assert "system/cpu_percent" in tags_logged

    def test_create_gpu_backend_returns_none(self) -> None:
        """_create_gpu_backend returns None when both backends fail."""
        with patch("serenityboard.writer.system_metrics._PynvmlBackend",
                    side_effect=ImportError("no pynvml")):
            with patch("subprocess.run", side_effect=FileNotFoundError("no nvidia-smi")):
                result = _create_gpu_backend()

        assert result is None


# ---------------------------------------------------------------------------
# Collector lifecycle
# ---------------------------------------------------------------------------

class TestCollectorLifecycle:
    """Start and stop the polling thread."""

    def test_start_and_stop(self) -> None:
        """Collector starts a daemon thread and stops cleanly."""
        writer = MagicMock()
        collector = SystemMetricsCollector(writer, interval_seconds=5.0)

        collector.start()
        assert collector._thread is not None
        assert collector._thread.is_alive()

        collector.stop()
        assert collector._thread is None

    def test_interval_clamped_to_minimum(self) -> None:
        """Interval is clamped to >= 5 seconds."""
        writer = MagicMock()
        collector = SystemMetricsCollector(writer, interval_seconds=1.0)
        assert collector._interval == 5.0

    def test_metrics_logged_at_interval(self) -> None:
        """Collector logs at least one metric within 2x the interval."""
        writer = MagicMock()
        collector = SystemMetricsCollector(writer, interval_seconds=5.0)
        collector._gpu_backend = None
        collector._backends_initialized = True
        collector._system_query = _create_system_query()

        collector.start()
        # Wait for at least one poll
        time.sleep(0.5)
        collector.stop()

        # At least one add_scalar call should have happened
        assert writer.add_scalar.call_count > 0


# ---------------------------------------------------------------------------
# /proc fallback
# ---------------------------------------------------------------------------

class TestProcFallback:
    """_query_system_proc reads from /proc on Linux."""

    def test_proc_returns_metrics(self) -> None:
        """On Linux, /proc fallback should return cpu and ram metrics."""
        import platform
        if platform.system() != "Linux":
            pytest.skip("Only runs on Linux")

        metrics = _query_system_proc()
        assert "system/cpu_percent" in metrics
        assert "system/ram_total_gb" in metrics
        assert "system/ram_used_gb" in metrics
        assert metrics["system/ram_total_gb"] > 0.0
