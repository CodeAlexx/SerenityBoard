"""Tests for RunDataProvider."""
from __future__ import annotations

import json
import sqlite3
import uuid

import pytest

from serenityboard.server.data_provider import RunDataProvider


class TestRunDataProvider:

    def _make_provider(self, sample_db: str) -> RunDataProvider:
        return RunDataProvider(sample_db)

    # -- tag discovery --

    def test_get_tags(self, sample_db: str) -> None:
        p = self._make_provider(sample_db)
        tags = p.get_tags()

        assert "loss/train" in tags["scalars"]
        assert "lr" in tags["scalars"]
        assert "samples" in tags["artifacts"]
        assert "weights/layer1" in tags["tensors"]
        assert "config" in tags["text_events"]
        p.close()

    # -- last activity --

    def test_get_last_activity(self, sample_db: str) -> None:
        p = self._make_provider(sample_db)
        wall_time, step = p.get_last_activity()

        assert wall_time is not None
        assert step is not None
        assert step == 19
        p.close()

    # -- scalar reads: downsampled --

    def test_read_scalars_downsampled_full(self, sample_db: str) -> None:
        """n=0 means full resolution -- all 20 points."""
        p = self._make_provider(sample_db)
        rows = p.read_scalars_downsampled("loss/train", n=0)

        assert len(rows) == 20
        # Check ordering
        steps = [r[0] for r in rows]
        assert steps == list(range(20))
        p.close()

    def test_read_scalars_downsampled_limited(self, sample_db: str) -> None:
        """n=5 should return at most 5 points, preserving first and last."""
        p = self._make_provider(sample_db)
        rows = p.read_scalars_downsampled("loss/train", n=5)

        assert 3 <= len(rows) <= 5
        steps = [r[0] for r in rows]
        # First and last must be preserved
        assert steps[0] == 0
        assert steps[-1] == 19
        p.close()

    def test_read_scalars_downsampled_empty_tag(self, sample_db: str) -> None:
        """Nonexistent tag should return empty list."""
        p = self._make_provider(sample_db)
        rows = p.read_scalars_downsampled("nonexistent/tag", n=10)

        assert rows == []
        p.close()

    # -- scalar reads: incremental --

    def test_read_scalars_incremental(self, sample_db: str) -> None:
        """First call returns data, second call returns empty (no new data)."""
        p = self._make_provider(sample_db)

        first = p.read_scalars_incremental("loss/train")
        assert len(first) == 20

        second = p.read_scalars_incremental("loss/train")
        assert len(second) == 0
        p.close()

    # -- artifact reads --

    def test_read_artifacts(self, sample_db: str) -> None:
        p = self._make_provider(sample_db)
        results = p.read_artifacts("samples")

        assert len(results) == 1
        assert results[0]["blob_key"] == "abc123.png"
        assert results[0]["kind"] == "image"
        assert results[0]["mime_type"] == "image/png"
        assert results[0]["width"] == 512
        assert results[0]["height"] == 512
        p.close()

    # -- histogram reads --

    def test_read_histograms(self, sample_db: str) -> None:
        p = self._make_provider(sample_db)
        results = p.read_histograms("weights/layer1")

        assert len(results) == 1
        entry = results[0]
        assert "step" in entry
        assert "wall_time" in entry
        assert "bins" in entry
        # bins should be a list of lists (10 x 3)
        assert len(entry["bins"]) == 10
        assert len(entry["bins"][0]) == 3
        p.close()

    # -- run info --

    def test_get_run_info(self, sample_db: str) -> None:
        p = self._make_provider(sample_db)
        info = p.get_run_info()

        assert "run_name" in info
        assert info["run_name"] == "sample_run"
        assert "status" in info
        assert info["status"] == "complete"
        assert "active_session_id" in info
        assert "start_time" in info
        assert "schema_version" in info
        p.close()

    # -- text reads --

    def test_read_text(self, sample_db: str) -> None:
        p = self._make_provider(sample_db)
        results = p.read_text("config")

        assert len(results) == 2
        assert results[0]["step"] == 0
        assert results[1]["step"] == 1
        # Verify the text content is there
        assert "lr" in results[0]["value"]
        p.close()

    # -- read_scalars_last --

    def test_read_scalars_last(self, sample_db: str) -> None:
        """read_scalars_last should return the last point for each requested tag."""
        p = self._make_provider(sample_db)
        try:
            result = p.read_scalars_last(["loss/train", "lr"])

            # Both tags present
            assert "loss/train" in result
            assert "lr" in result

            # Each value has required keys
            for tag in ("loss/train", "lr"):
                assert "step" in result[tag]
                assert "wall_time" in result[tag]
                assert "value" in result[tag]

            # loss/train has steps 0-19, last is 19
            assert result["loss/train"]["step"] == 19
            # lr has steps 0-9, last is 9
            assert result["lr"]["step"] == 9
        finally:
            p.close()

    # -- session reset --

    def test_check_session_reset(self, sample_db: str) -> None:
        """Changing active_session_id should reset incremental cursors."""
        p = self._make_provider(sample_db)
        try:
            # First call: get all 20 points
            first = p.read_scalars_incremental("loss/train")
            assert len(first) == 20

            # Second call: cursor at end, no new data
            second = p.read_scalars_incremental("loss/train")
            assert len(second) == 0

            # Directly update active_session_id via a separate write connection
            write_conn = sqlite3.connect(sample_db)
            new_session = str(uuid.uuid4())
            with write_conn:
                write_conn.execute(
                    "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                    ("active_session_id", json.dumps(new_session)),
                )
            write_conn.close()

            # Third call: session changed, cursors should reset, get all 20 again
            third = p.read_scalars_incremental("loss/train")
            assert len(third) == 20
        finally:
            p.close()

    # -- downsampled edge cases --

    def test_read_scalars_downsampled_single_point(self, sample_db: str) -> None:
        """When a tag has a single unique step, downsampled returns that point."""
        # Insert a unique tag with exactly one point
        write_conn = sqlite3.connect(sample_db)
        with write_conn:
            write_conn.execute(
                "INSERT INTO scalars (tag, step, wall_time, value) VALUES (?, ?, ?, ?)",
                ("single_point_tag", 42, 1000.0, 3.14),
            )
        write_conn.close()

        p = self._make_provider(sample_db)
        try:
            rows = p.read_scalars_downsampled("single_point_tag", n=5)
            assert len(rows) == 1
            assert rows[0][0] == 42
            assert rows[0][2] == pytest.approx(3.14)
        finally:
            p.close()

    def test_read_scalars_downsampled_n_clamped(self, sample_db: str) -> None:
        """Requesting n=1 should be clamped to at least 3 rows (max(n, 3))."""
        p = self._make_provider(sample_db)
        try:
            rows = p.read_scalars_downsampled("loss/train", n=1)
            # loss/train has 20 points, n clamped to 3, so at least 3 returned
            assert len(rows) >= 3
            # First and last preserved
            steps = [r[0] for r in rows]
            assert steps[0] == 0
            assert steps[-1] == 19
        finally:
            p.close()
