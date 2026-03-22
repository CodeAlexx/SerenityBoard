"""Tests for SerenityBoard FastAPI application endpoints."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import time
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from serenityboard.server.app import create_app
from serenityboard.server.data_provider import RunDataProvider
from serenityboard.writer.schema import create_tables, set_pragmas


# ---------------------------------------------------------------------------
# Thread-safety patch for TestClient
# ---------------------------------------------------------------------------
# Starlette's TestClient runs the ASGI app in a background thread, but
# RunDataProvider opens sqlite3 connections with the default
# check_same_thread=True.  We patch __init__ for tests only.

_orig_rdp_init = RunDataProvider.__init__


def _thread_safe_rdp_init(self, db_path: str) -> None:
    """Same as RunDataProvider.__init__ but with check_same_thread=False."""
    self._db_path = db_path
    self.run_dir = os.path.dirname(db_path)
    self._conn = sqlite3.connect(
        f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
    )
    self._conn.execute("PRAGMA query_only = ON")
    self._last_seen: dict[str, int] = {}
    self._known_session_id: str | None = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_run_db(
    parent_dir: str,
    run_name: str,
    *,
    status: str = "complete",
    start_time: float | None = None,
    num_scalars: int = 5,
) -> str:
    """Create ``parent_dir/run_name/board.db`` with schema, metadata, a
    session row, scalar points, and an artifact entry.

    Returns the absolute path to ``board.db``.
    """
    run_dir = os.path.join(parent_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    db_path = os.path.join(run_dir, "board.db")

    now = time.time()
    if start_time is None:
        start_time = now - 600

    conn = sqlite3.connect(db_path)
    set_pragmas(conn)
    create_tables(conn)

    session_id = f"session-{run_name}"

    with conn:
        for key, value in [
            ("active_session_id", json.dumps(session_id)),
            ("status", json.dumps(status)),
            ("run_name", json.dumps(run_name)),
            ("start_time", json.dumps(start_time)),
        ]:
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (key, value),
            )

        conn.execute(
            "INSERT INTO sessions (session_id, start_time, resume_step, status) "
            "VALUES (?, ?, NULL, ?)",
            (session_id, start_time, "running" if status == "running" else "complete"),
        )

        for step in range(num_scalars):
            conn.execute(
                "INSERT INTO scalars (tag, step, wall_time, value) VALUES (?, ?, ?, ?)",
                ("loss/train", step, start_time + step * 10, 1.0 - step * 0.1),
            )

        # Add an artifact
        conn.execute(
            "INSERT INTO artifacts "
            "(tag, step, seq_index, wall_time, kind, mime_type, blob_key, width, height, meta) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("samples", 0, 0, start_time + 50, "image", "image/png", f"{run_name}_img.png", 512, 512, "{}"),
        )

    conn.close()

    # Create blobs directory with dummy blob
    blobs_dir = os.path.join(run_dir, "blobs")
    os.makedirs(blobs_dir, exist_ok=True)
    with open(os.path.join(blobs_dir, f"{run_name}_img.png"), "wb") as f:
        f.write(b"FAKE_PNG_DATA")

    return db_path


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def app_client():
    """Create a temp logdir with 2 runs, build the app, yield TestClient.

    Patches RunDataProvider.__init__ so sqlite3 connections work across
    threads (TestClient runs ASGI handlers in a background thread).
    """
    logdir = tempfile.mkdtemp(prefix="sb_apptest_")
    try:
        _make_run_db(logdir, "run1")
        _make_run_db(logdir, "run2")

        with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
            app = create_app(logdir)
            client = TestClient(app, raise_server_exceptions=False)
            yield client, logdir
    finally:
        shutil.rmtree(logdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestListRuns:
    def test_list_runs(self, app_client):
        """GET /api/runs should return 2 runs with correct names."""
        client, _ = app_client
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        names = {r["name"] for r in data}
        assert names == {"run1", "run2"}


class TestGetTags:
    def test_get_tags(self, app_client):
        """GET /api/runs/{run}/tags should return scalars and artifacts keys."""
        client, _ = app_client
        resp = client.get("/api/runs/run1/tags")
        assert resp.status_code == 200
        data = resp.json()
        assert "scalars" in data
        assert "artifacts" in data
        assert "loss/train" in data["scalars"]
        assert "samples" in data["artifacts"]

    def test_get_tags_404(self, app_client):
        """GET /api/runs/nonexistent/tags should return 404."""
        client, _ = app_client
        resp = client.get("/api/runs/nonexistent/tags")
        assert resp.status_code == 404


class TestGetScalars:
    def test_get_scalars(self, app_client):
        """GET /api/runs/{run}/scalars?tag=loss/train should return array of
        [step, wall_time, value] triples."""
        client, _ = app_client
        resp = client.get("/api/runs/run1/scalars", params={"tag": "loss/train"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 5
        # Each element is [step, wall_time, value]
        first = data[0]
        assert len(first) == 3
        assert first[0] == 0  # step
        assert isinstance(first[1], float)  # wall_time
        assert isinstance(first[2], float)  # value

    def test_get_scalars_missing_tag(self, app_client):
        """GET /api/runs/{run}/scalars?tag=nope should return empty array."""
        client, _ = app_client
        resp = client.get("/api/runs/run1/scalars", params={"tag": "nope"})
        assert resp.status_code == 200
        data = resp.json()
        assert data == []


class TestDeleteRun:
    def test_delete_run(self, app_client):
        """DELETE /api/runs/{run} should return 200, then list shows 1 run."""
        client, _ = app_client
        resp = client.delete("/api/runs/run1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["deleted"] == "run1"

        # Verify only 1 run remains
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        names = {r["name"] for r in resp.json()}
        assert names == {"run2"}

    def test_delete_run_404(self, app_client):
        """DELETE /api/runs/nonexistent should return 404."""
        client, _ = app_client
        resp = client.delete("/api/runs/nonexistent")
        assert resp.status_code == 404


class TestGetArtifacts:
    def test_get_artifacts(self, app_client):
        """GET /api/runs/{run}/artifacts?tag=samples should return entries with blob_key."""
        client, _ = app_client
        resp = client.get("/api/runs/run1/artifacts", params={"tag": "samples"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        entry = data[0]
        assert "blob_key" in entry
        assert entry["blob_key"] == "run1_img.png"


class TestCompareScalars:
    def test_compare_scalars(self, app_client):
        """GET /api/compare/scalars?tag=loss/train&runs=run1,run2 should return
        data for both runs."""
        client, _ = app_client
        resp = client.get(
            "/api/compare/scalars",
            params={"tag": "loss/train", "runs": "run1,run2"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "run1" in data
        assert "run2" in data
        assert len(data["run1"]) == 5
        assert len(data["run2"]) == 5
        # Each entry is [step, wall_time, value]
        assert len(data["run1"][0]) == 3

    def test_compare_scalars_xaxis_wall_time(self, app_client):
        """x_axis=wall_time should use wall_time as the x value (first element)."""
        client, _ = app_client
        resp = client.get(
            "/api/compare/scalars",
            params={"tag": "loss/train", "runs": "run1,run2", "x_axis": "wall_time"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "run1" in data
        assert len(data["run1"]) == 5
        # First element should be a float timestamp (wall_time), not an integer step
        first_point = data["run1"][0]
        assert isinstance(first_point[0], float)
        # wall_time as x should equal wall_time (second element)
        assert first_point[0] == first_point[1]

    def test_compare_scalars_xaxis_relative(self, app_client):
        """x_axis=relative should make the first data point's x value 0 (or close to 0)."""
        client, _ = app_client
        resp = client.get(
            "/api/compare/scalars",
            params={"tag": "loss/train", "runs": "run1", "x_axis": "relative"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "run1" in data
        assert len(data["run1"]) == 5
        # First point's x value (relative time) should be 0
        first_point = data["run1"][0]
        assert first_point[0] == pytest.approx(0.0, abs=1e-6)
        # Subsequent points should have positive relative time
        second_point = data["run1"][1]
        assert second_point[0] > 0
