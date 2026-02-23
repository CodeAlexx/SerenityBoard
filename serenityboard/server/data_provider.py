"""Read-only data provider for a single run's board.db."""
from __future__ import annotations

import json
import logging
import os
import sqlite3

import numpy as np

__all__ = ["RunDataProvider"]

logger = logging.getLogger(__name__)


class RunDataProvider:
    """Reads from a single run's board.db (read-only connection).

    Maintains per-tag high-water marks for incremental reads.
    Detects session changes and resets state accordingly.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self.run_dir = os.path.dirname(db_path)
        self._conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self._conn.execute("PRAGMA query_only = ON")
        self._last_seen: dict[str, int] = {}
        self._known_session_id: str | None = None

    # ── session tracking ─────────────────────────────────────────────

    def _check_session(self) -> None:
        """Read active_session_id from metadata. If changed, reset all cursors."""
        row = self._conn.execute(
            "SELECT value FROM metadata WHERE key = 'active_session_id'"
        ).fetchone()
        if row is None:
            return
        current_session = json.loads(row[0])
        if current_session != self._known_session_id:
            if self._known_session_id is not None:
                logger.info(
                    "Session change detected (%s -> %s), resetting cursors.",
                    self._known_session_id,
                    current_session,
                )
            self._known_session_id = current_session
            self._last_seen.clear()

    # ── scalar reads ─────────────────────────────────────────────────

    def read_scalars_incremental(self, tag: str) -> list[tuple]:
        """Read new scalar points since last call for this tag."""
        self._check_session()
        last = self._last_seen.get(tag, -1)
        rows = self._conn.execute(
            "SELECT step, wall_time, value FROM scalars "
            "WHERE tag = ? AND step > ? ORDER BY step",
            (tag, last),
        ).fetchall()
        if rows:
            self._last_seen[tag] = rows[-1][0]
        return rows

    def read_scalars_downsampled(self, tag: str, n: int) -> list[tuple]:
        """Read up to *n* points, preserving first and last.

        Uses arithmetic step sampling with indexed seeks on the (tag, step)
        primary key. Each sample point is an O(log N) B-tree seek, giving
        O(n log N) total instead of O(N) for window-function approaches.
        """
        # Truth mode: n<=0 means return full-resolution series with no downsampling.
        if n <= 0:
            return self._conn.execute(
                "SELECT step, wall_time, value FROM scalars "
                "WHERE tag = ? ORDER BY step",
                (tag,),
            ).fetchall()
        n = max(n, 3)  # need at least 3 for first + interior + last
        # Get bounds via indexed lookups (O(log N) each)
        first_row = self._conn.execute(
            "SELECT step FROM scalars WHERE tag = ? ORDER BY step ASC LIMIT 1",
            (tag,),
        ).fetchone()
        if first_row is None:
            return []
        last_row = self._conn.execute(
            "SELECT step FROM scalars WHERE tag = ? ORDER BY step DESC LIMIT 1",
            (tag,),
        ).fetchone()
        min_step, max_step = first_row[0], last_row[0]
        step_range = max_step - min_step
        if step_range == 0:
            return self._conn.execute(
                "SELECT step, wall_time, value FROM scalars WHERE tag = ? AND step = ?",
                (tag, min_step),
            ).fetchall()
        # Check if total rows <= n (also fast via COUNT on indexed column)
        total = self._conn.execute(
            "SELECT COUNT(*) FROM scalars WHERE tag = ?", (tag,)
        ).fetchone()[0]
        if total <= n:
            return self._conn.execute(
                "SELECT step, wall_time, value FROM scalars "
                "WHERE tag = ? ORDER BY step",
                (tag,),
            ).fetchall()
        # Generate n evenly-spaced target step values across [min_step, max_step]
        # For each target, seek to the nearest actual step via indexed lookup.
        sample_steps: set[int] = {min_step, max_step}
        step_interval = step_range / (n - 1)
        for i in range(1, n - 1):
            target = min_step + int(i * step_interval)
            # Find nearest step >= target (B-tree seek)
            row = self._conn.execute(
                "SELECT step FROM scalars WHERE tag = ? AND step >= ? ORDER BY step ASC LIMIT 1",
                (tag, target),
            ).fetchone()
            if row:
                sample_steps.add(row[0])
        # Batch fetch full rows for the sampled steps
        sorted_steps = sorted(sample_steps)[:n]
        placeholders = ",".join("?" * len(sorted_steps))
        rows = self._conn.execute(
            f"SELECT step, wall_time, value FROM scalars "  # noqa: S608
            f"WHERE tag = ? AND step IN ({placeholders}) ORDER BY step",
            [tag] + sorted_steps,
        ).fetchall()
        return rows

    def read_scalars_last(self, tags: list[str]) -> dict:
        """Return the last scalar point for each tag. Fast path for status bar."""
        result: dict[str, dict] = {}
        for tag in tags:
            row = self._conn.execute(
                "SELECT step, wall_time, value FROM scalars "
                "WHERE tag = ? ORDER BY step DESC LIMIT 1",
                (tag,),
            ).fetchone()
            if row:
                result[tag] = {"step": row[0], "wall_time": row[1], "value": row[2]}
        return result

    # ── tag discovery ────────────────────────────────────────────────

    def get_tags(self) -> dict[str, list[str]]:
        """Return all tags grouped by table type, plus trace phases and eval suites."""
        result: dict[str, list[str]] = {}
        for table in ("scalars", "tensors", "artifacts", "text_events"):
            rows = self._conn.execute(
                f"SELECT DISTINCT tag FROM {table}"  # noqa: S608 – table name is from a fixed list
            ).fetchall()
            result[table] = sorted(r[0] for r in rows)
        # audio table may not exist in older databases
        try:
            rows = self._conn.execute("SELECT DISTINCT tag FROM audio").fetchall()
            result["audio"] = sorted(r[0] for r in rows)
        except Exception:
            result["audio"] = []
        # trace_events uses 'phase' instead of 'tag'
        rows = self._conn.execute(
            "SELECT DISTINCT phase FROM trace_events"
        ).fetchall()
        result["trace_events"] = sorted(r[0] for r in rows)
        # eval_results uses 'suite_name'
        rows = self._conn.execute(
            "SELECT DISTINCT suite_name FROM eval_results"
        ).fetchall()
        result["eval_suites"] = sorted(r[0] for r in rows)
        # PR curve tags
        try:
            pr_curve_tags = [r[0] for r in self._conn.execute(
                "SELECT DISTINCT tag FROM pr_curves"
            ).fetchall()]
            result["pr_curves"] = sorted(pr_curve_tags)
        except Exception:
            result["pr_curves"] = []
        # graph tags
        try:
            rows = self._conn.execute("SELECT DISTINCT tag FROM graphs").fetchall()
            result["graphs"] = sorted(r[0] for r in rows)
        except Exception:
            result["graphs"] = []
        # mesh tags
        try:
            rows = self._conn.execute("SELECT DISTINCT tag FROM meshes").fetchall()
            result["meshes"] = sorted(r[0] for r in rows)
        except Exception:
            result["meshes"] = []
        # embedding tags
        try:
            rows = self._conn.execute("SELECT DISTINCT tag FROM embeddings").fetchall()
            result["embeddings"] = sorted(r[0] for r in rows)
        except Exception:
            result["embeddings"] = []
        return result

    # ── metadata / hparams ───────────────────────────────────────────

    def get_run_info(self) -> dict:
        """Return run metadata (name, status, start_time, hparams)."""
        meta: dict[str, object] = {}
        for row in self._conn.execute("SELECT key, value FROM metadata").fetchall():
            meta[row[0]] = json.loads(row[1])
        return meta

    def get_last_activity(self) -> tuple[float | None, int | None]:
        """Return (last_wall_time, last_step) across scalars/tensors.

        Returns (None, None) if no data exists yet.
        """
        latest_wt = None
        latest_step = None
        for table in ("scalars", "tensors", "trace_events"):
            try:
                row = self._conn.execute(
                    f"SELECT MAX(wall_time), MAX(step) FROM {table}"  # noqa: S608
                ).fetchone()
                if row and row[0] is not None:
                    if latest_wt is None or row[0] > latest_wt:
                        latest_wt = row[0]
                    if row[1] is not None:
                        if latest_step is None or row[1] > latest_step:
                            latest_step = row[1]
            except sqlite3.OperationalError:
                pass
        return latest_wt, latest_step

    def get_hparams(self) -> dict:
        """Return hparams and metrics for this run."""
        meta = self.get_run_info()
        hparams = meta.get("hparams", {})
        metrics: dict[str, float] = {}
        for row in self._conn.execute(
            "SELECT metric_tag, value FROM hparam_metrics"
        ).fetchall():
            metrics[row[0]] = row[1]
        return {"hparams": hparams, "metrics": metrics}

    # ── histogram / tensor reads ─────────────────────────────────────

    def read_histograms(self, tag: str, downsample: int = 100) -> list[dict]:
        """Read histogram data for a tag from the tensors table."""
        rows = self._conn.execute(
            "SELECT step, wall_time, dtype, shape, data FROM tensors "
            "WHERE tag = ? ORDER BY step",
            (tag,),
        ).fetchall()
        if len(rows) > downsample:
            indices = set(range(0, len(rows), max(1, len(rows) // downsample)))
            indices.add(len(rows) - 1)  # always include last
            rows = [rows[i] for i in sorted(indices)[:downsample]]
        result = []
        for r in rows:
            step, wall_time, dtype, shape_json, data = r
            try:
                shape = tuple(json.loads(shape_json))
                arr = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)
            except (ValueError, TypeError) as e:
                logger.warning("Corrupt histogram at step %d: %s", step, e)
                continue
            # arr is (N, 3) with columns [left, right, count]
            bins = arr.tolist()
            result.append({"step": step, "wall_time": wall_time, "bins": bins})
        return result

    # ── distribution reads ───────────────────────────────────────────

    _BASIS_POINTS = [0, 668, 1587, 3085, 5000, 6915, 8413, 9332, 10000]

    def _compress_histogram(self, bins: np.ndarray) -> list[dict] | None:
        """Compress histogram bins into 9 percentile values using normal-CDF basis points.

        Parameters
        ----------
        bins:
            Numpy array of shape (N, 3) with columns [left_edge, right_edge, count].

        Returns
        -------
        List of {"bp": int, "value": float} dicts, or None if all counts are zero.
        """
        counts = bins[:, 2]
        total = counts.sum()
        if total == 0:
            return None
        cumulative = np.cumsum(counts) / total  # normalized to [0, 1]
        centers = (bins[:, 0] + bins[:, 1]) / 2

        percentiles = []
        for bp in self._BASIS_POINTS:
            target = bp / 10000.0
            if target <= 0:
                percentiles.append({"bp": bp, "value": float(bins[0, 0])})
            elif target >= 1:
                percentiles.append({"bp": bp, "value": float(bins[-1, 1])})
            else:
                idx = int(np.searchsorted(cumulative, target))
                if idx == 0:
                    percentiles.append({"bp": bp, "value": float(centers[0])})
                elif idx >= len(cumulative):
                    percentiles.append({"bp": bp, "value": float(centers[-1])})
                else:
                    frac = (target - cumulative[idx - 1]) / (
                        cumulative[idx] - cumulative[idx - 1] + 1e-12
                    )
                    value = centers[idx - 1] + frac * (centers[idx] - centers[idx - 1])
                    percentiles.append({"bp": bp, "value": float(value)})
        return percentiles

    def read_distributions(self, tag: str, downsample: int = 100) -> list[dict]:
        """Read histogram data and compress into percentile distributions for a tag."""
        rows = self._conn.execute(
            "SELECT step, wall_time, dtype, shape, data FROM tensors "
            "WHERE tag = ? ORDER BY step",
            (tag,),
        ).fetchall()
        if len(rows) > downsample:
            indices = set(range(0, len(rows), max(1, len(rows) // downsample)))
            indices.add(len(rows) - 1)  # always include last
            rows = [rows[i] for i in sorted(indices)[:downsample]]
        result = []
        for r in rows:
            step, wall_time, dtype, shape_json, data = r
            try:
                shape = tuple(json.loads(shape_json))
                arr = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)
            except (ValueError, TypeError) as e:
                logger.warning("Corrupt histogram at step %d: %s", step, e)
                continue
            # arr is (N, 3) with columns [left, right, count]
            percentiles = self._compress_histogram(arr)
            if percentiles is None:
                continue
            result.append({"step": step, "wall_time": wall_time, "percentiles": percentiles})
        return result

    # ── PR curve reads ──────────────────────────────────────────────

    def read_pr_curves(self, tag: str, downsample: int = 50) -> list[dict]:
        """Return precision-recall curve data for the given tag."""
        try:
            rows = self._conn.execute(
                "SELECT step, wall_time, class_index, num_thresholds, data "
                "FROM pr_curves WHERE tag = ? ORDER BY step",
                (tag,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []  # pr_curves table may not exist in older databases
        if downsample and len(rows) > downsample:
            step_size = max(1, len(rows) // downsample)
            rows = rows[::step_size]
        results = []
        for r in rows:
            step_val, wall_time, class_idx, num_thresh, data_bytes = r
            arr = np.frombuffer(data_bytes, dtype=np.float64).reshape(6, num_thresh)
            thresholds = np.linspace(0.0, 1.0, num_thresh)
            results.append({
                "step": step_val,
                "wall_time": wall_time,
                "class_index": class_idx,
                "num_thresholds": num_thresh,
                "precision": arr[4].tolist(),
                "recall": arr[5].tolist(),
                "thresholds": thresholds.tolist(),
            })
        return results

    # ── blob / image reads ───────────────────────────────────────────

    def read_images(self, tag: str, downsample: int = 100) -> list[dict]:
        """Read image artifact metadata for a tag."""
        rows = self._conn.execute(
            "SELECT step, wall_time, blob_key, width, height, kind, meta FROM artifacts "
            "WHERE tag = ? ORDER BY step",
            (tag,),
        ).fetchall()
        if len(rows) > downsample:
            indices = set(range(0, len(rows), max(1, len(rows) // downsample)))
            indices.add(len(rows) - 1)  # always include last
            rows = [rows[i] for i in sorted(indices)[:downsample]]
        return [
            {
                "step": r[0],
                "wall_time": r[1],
                "blob_key": r[2],
                "width": r[3],
                "height": r[4],
            }
            for r in rows
        ]

    def get_blob_info(self, blob_key: str) -> dict | None:
        """Look up a blob in the DB. Returns mime_type or None if not found."""
        row = self._conn.execute(
            "SELECT mime_type FROM artifacts WHERE blob_key = ? LIMIT 1",
            (blob_key,),
        ).fetchone()
        if row is not None:
            return {"mime_type": row[0]}
        # Also check the audio table (may not exist in older databases)
        try:
            row = self._conn.execute(
                "SELECT mime_type FROM audio WHERE blob_key = ? LIMIT 1",
                (blob_key,),
            ).fetchone()
            if row is not None:
                return {"mime_type": row[0]}
        except sqlite3.OperationalError:
            pass
        # Also check mesh blob keys (vertices, faces, colors)
        try:
            row = self._conn.execute(
                "SELECT tag FROM meshes WHERE vertices_blob_key = ? "
                "OR faces_blob_key = ? OR colors_blob_key = ? LIMIT 1",
                (blob_key, blob_key, blob_key),
            ).fetchone()
            if row is not None:
                return {"mime_type": "application/octet-stream"}
        except sqlite3.OperationalError:
            pass
        # Also check embedding blob keys (tensor matrix + sprite image)
        try:
            row = self._conn.execute(
                "SELECT tag FROM embeddings WHERE tensor_blob_key = ? LIMIT 1",
                (blob_key,),
            ).fetchone()
            if row is not None:
                return {"mime_type": "application/octet-stream"}
            row = self._conn.execute(
                "SELECT tag FROM embeddings WHERE sprite_blob_key = ? LIMIT 1",
                (blob_key,),
            ).fetchone()
            if row is not None:
                return {"mime_type": "image/png"}
        except sqlite3.OperationalError:
            pass
        # Also check graph blob keys
        try:
            row = self._conn.execute(
                "SELECT tag FROM graphs WHERE graph_blob_key = ? LIMIT 1",
                (blob_key,),
            ).fetchone()
            if row is not None:
                return {"mime_type": "application/json"}
        except sqlite3.OperationalError:
            pass
        return None

    # ── audio reads ─────────────────────────────────────────────────

    def read_audio(self, tag: str, downsample: int = 50) -> list[dict]:
        """Return audio metadata for the given tag."""
        try:
            rows = self._conn.execute(
                "SELECT step, wall_time, blob_key, sample_rate, num_channels, "
                "duration_ms, mime_type, label "
                "FROM audio WHERE tag = ? ORDER BY step",
                (tag,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []  # audio table may not exist in older databases
        if downsample and len(rows) > downsample:
            step = max(1, len(rows) // downsample)
            rows = rows[::step]
        return [
            {
                "step": r[0],
                "wall_time": r[1],
                "blob_key": r[2],
                "sample_rate": r[3],
                "num_channels": r[4],
                "duration_ms": r[5],
                "mime_type": r[6],
                "label": r[7],
            }
            for r in rows
        ]

    # ── mesh reads ────────────────────────────────────────────────────

    def read_meshes(self, tag: str | None = None, step: int | None = None, downsample: int = 50) -> list[dict]:
        """Read mesh data, with flexible query modes.

        Parameters
        ----------
        tag:
            If provided with step, return a single mesh entry.
            If provided without step, return list of available steps with mesh stats.
            If None, return all mesh entries (for tag listing).
        step:
            Specific step to retrieve (requires tag).
        downsample:
            Maximum number of entries to return.
        """
        try:
            if tag is not None and step is not None:
                # Specific mesh entry
                row = self._conn.execute(
                    "SELECT tag, step, wall_time, num_vertices, has_faces, has_colors, "
                    "num_faces, vertices_blob_key, faces_blob_key, colors_blob_key, config_json "
                    "FROM meshes WHERE tag = ? AND step = ?",
                    (tag, step),
                ).fetchone()
                if row is None:
                    return []
                config = None
                if row[10]:
                    try:
                        config = json.loads(row[10])
                    except (json.JSONDecodeError, TypeError):
                        config = None
                return [{
                    "tag": row[0],
                    "step": row[1],
                    "wall_time": row[2],
                    "num_vertices": row[3],
                    "has_faces": bool(row[4]),
                    "has_colors": bool(row[5]),
                    "num_faces": row[6],
                    "vertices_blob_key": row[7],
                    "faces_blob_key": row[8],
                    "colors_blob_key": row[9],
                    "config": config,
                }]
            elif tag is not None:
                # All steps for a given tag
                rows = self._conn.execute(
                    "SELECT tag, step, wall_time, num_vertices, has_faces, has_colors, "
                    "num_faces, vertices_blob_key, faces_blob_key, colors_blob_key, config_json "
                    "FROM meshes WHERE tag = ? ORDER BY step",
                    (tag,),
                ).fetchall()
            else:
                # All meshes
                rows = self._conn.execute(
                    "SELECT tag, step, wall_time, num_vertices, has_faces, has_colors, "
                    "num_faces, vertices_blob_key, faces_blob_key, colors_blob_key, config_json "
                    "FROM meshes ORDER BY tag, step",
                ).fetchall()
        except sqlite3.OperationalError:
            return []  # meshes table may not exist in older databases
        if downsample and len(rows) > downsample:
            step_size = max(1, len(rows) // downsample)
            rows = rows[::step_size]
        results = []
        for r in rows:
            config = None
            if r[10]:
                try:
                    config = json.loads(r[10])
                except (json.JSONDecodeError, TypeError):
                    config = None
            results.append({
                "tag": r[0],
                "step": r[1],
                "wall_time": r[2],
                "num_vertices": r[3],
                "has_faces": bool(r[4]),
                "has_colors": bool(r[5]),
                "num_faces": r[6],
                "vertices_blob_key": r[7],
                "faces_blob_key": r[8],
                "colors_blob_key": r[9],
                "config": config,
            })
        return results

    # ── embedding reads ──────────────────────────────────────────────

    def read_embeddings(self, tag: str | None = None, step: int | None = None, downsample: int = 20) -> list[dict]:
        """Read embedding data, with flexible query modes.

        Parameters
        ----------
        tag:
            If provided with step, return full embedding data for that specific entry.
            If provided without step, return list of available steps with metadata.
            If None, return all embedding entries (for tag listing).
        step:
            Specific step to retrieve (requires tag).
        downsample:
            Maximum number of entries to return.
        """
        try:
            if tag is not None and step is not None:
                # Specific embedding entry
                row = self._conn.execute(
                    "SELECT tag, step, wall_time, num_points, dimensions, "
                    "tensor_blob_key, metadata_json, metadata_header, "
                    "sprite_blob_key, sprite_single_h, sprite_single_w "
                    "FROM embeddings WHERE tag = ? AND step = ?",
                    (tag, step),
                ).fetchone()
                if row is None:
                    return []
                metadata = None
                if row[6]:
                    try:
                        metadata = json.loads(row[6])
                    except (json.JSONDecodeError, TypeError):
                        metadata = None
                header = None
                if row[7]:
                    try:
                        header = json.loads(row[7])
                    except (json.JSONDecodeError, TypeError):
                        header = None
                return [{
                    "tag": row[0],
                    "step": row[1],
                    "wall_time": row[2],
                    "num_points": row[3],
                    "dimensions": row[4],
                    "tensor_blob_key": row[5],
                    "metadata": metadata,
                    "metadata_header": header,
                    "sprite_blob_key": row[8],
                    "sprite_single_h": row[9],
                    "sprite_single_w": row[10],
                }]
            elif tag is not None:
                # All steps for a given tag
                rows = self._conn.execute(
                    "SELECT tag, step, wall_time, num_points, dimensions, "
                    "tensor_blob_key, metadata_json, metadata_header, "
                    "sprite_blob_key, sprite_single_h, sprite_single_w "
                    "FROM embeddings WHERE tag = ? ORDER BY step",
                    (tag,),
                ).fetchall()
            else:
                # All embeddings
                rows = self._conn.execute(
                    "SELECT tag, step, wall_time, num_points, dimensions, "
                    "tensor_blob_key, metadata_json, metadata_header, "
                    "sprite_blob_key, sprite_single_h, sprite_single_w "
                    "FROM embeddings ORDER BY tag, step",
                ).fetchall()
        except sqlite3.OperationalError:
            return []  # embeddings table may not exist in older databases
        if downsample and len(rows) > downsample:
            step_size = max(1, len(rows) // downsample)
            rows = rows[::step_size]
        results = []
        for r in rows:
            metadata = None
            if r[6]:
                try:
                    metadata = json.loads(r[6])
                except (json.JSONDecodeError, TypeError):
                    metadata = None
            header = None
            if r[7]:
                try:
                    header = json.loads(r[7])
                except (json.JSONDecodeError, TypeError):
                    header = None
            results.append({
                "tag": r[0],
                "step": r[1],
                "wall_time": r[2],
                "num_points": r[3],
                "dimensions": r[4],
                "tensor_blob_key": r[5],
                "metadata": metadata,
                "metadata_header": header,
                "sprite_blob_key": r[8],
                "sprite_single_h": r[9],
                "sprite_single_w": r[10],
            })
        return results

    # ── graph reads ──────────────────────────────────────────────────

    def read_graphs(self, tag: str | None = None, downsample: int = 10) -> list[dict]:
        """Read graph data, optionally filtered by tag.

        The ``graph_blob_key`` column may contain either a blob storage key
        (hex hash + extension, e.g. ``"a1b2c3d4e5f60a8b.json"``) or legacy
        inline JSON (starts with ``{``).  Both cases are handled transparently.

        Parameters
        ----------
        tag:
            If provided, filter to this specific graph tag.
            If None, return graphs for all tags.
        downsample:
            Maximum number of graph events to return.
        """
        try:
            if tag is not None:
                rows = self._conn.execute(
                    "SELECT tag, step, wall_time, graph_blob_key "
                    "FROM graphs WHERE tag = ? ORDER BY step",
                    (tag,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT tag, step, wall_time, graph_blob_key "
                    "FROM graphs ORDER BY tag, step",
                ).fetchall()
        except sqlite3.OperationalError:
            return []  # graphs table may not exist in older databases
        if downsample and len(rows) > downsample:
            step_size = max(1, len(rows) // downsample)
            rows = rows[::step_size]
        blobs_dir = os.path.join(self.run_dir, "blobs")
        results = []
        for r in rows:
            value = r[3]
            if not value:
                continue
            try:
                if value.startswith("{"):
                    # Legacy inline JSON
                    graph_data = json.loads(value)
                else:
                    # Blob key — load file from blobs directory
                    blob_path = os.path.join(blobs_dir, value)
                    with open(blob_path, "rb") as f:
                        graph_data = json.loads(f.read().decode("utf-8"))
            except (json.JSONDecodeError, TypeError, OSError) as exc:
                logger.warning(
                    "Cannot load graph data at step %d for tag %s: %s",
                    r[1], r[0], exc,
                )
                continue
            results.append({
                "tag": r[0],
                "step": r[1],
                "wall_time": r[2],
                "graph_data": graph_data,
            })
        return results

    # ── text reads ───────────────────────────────────────────────────

    def read_text(self, tag: str, limit: int | None = None) -> list[dict]:
        """Read text entries for a tag, with optional limit."""
        if limit is not None:
            rows = self._conn.execute(
                "SELECT step, wall_time, value FROM text_events WHERE tag = ? ORDER BY step LIMIT ?",
                (tag, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT step, wall_time, value FROM text_events WHERE tag = ? ORDER BY step",
                (tag,),
            ).fetchall()
        return [{"step": r[0], "wall_time": r[1], "value": r[2]} for r in rows]

    # ── artifact reads ──────────────────────────────────────────────

    def read_artifacts(self, tag: str, downsample: int = 100, kind: str | None = None) -> list[dict]:
        """Read artifact metadata for a tag, with optional kind filter."""
        if kind:
            rows = self._conn.execute(
                "SELECT step, wall_time, blob_key, mime_type, width, height, kind, meta "
                "FROM artifacts WHERE tag = ? AND kind = ? ORDER BY step",
                (tag, kind),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT step, wall_time, blob_key, mime_type, width, height, kind, meta "
                "FROM artifacts WHERE tag = ? ORDER BY step",
                (tag,),
            ).fetchall()
        if len(rows) > downsample:
            indices = set(range(0, len(rows), max(1, len(rows) // downsample)))
            indices.add(len(rows) - 1)
            rows = [rows[i] for i in sorted(indices)[:downsample]]
        return [
            {
                "step": r[0],
                "wall_time": r[1],
                "blob_key": r[2],
                "mime_type": r[3],
                "width": r[4],
                "height": r[5],
                "kind": r[6],
                "meta": json.loads(r[7]) if r[7] else {},
            }
            for r in rows
        ]

    # ── trace reads ──────────────────────────────────────────────────

    def read_trace_events(self, step_from: int | None = None, step_to: int | None = None) -> list[dict]:
        """Read trace events, optionally filtered by step range."""
        conditions = []
        params: list = []
        if step_from is not None:
            conditions.append("step >= ?")
            params.append(step_from)
        if step_to is not None:
            conditions.append("step <= ?")
            params.append(step_to)
        where = " AND ".join(conditions) if conditions else "1=1"
        rows = self._conn.execute(
            f"SELECT step, wall_time, phase, duration_ms, details FROM trace_events "  # noqa: S608
            f"WHERE {where} ORDER BY step, phase",
            params,
        ).fetchall()
        return [
            {
                "step": r[0],
                "phase": r[2],
                "duration_ms": r[3],
                "details": json.loads(r[4]) if r[4] else {},
            }
            for r in rows
        ]

    def read_trace_events_incremental(self) -> list[dict]:
        """Read new trace events since last call. Uses cursor tracking."""
        self._check_session()
        cursor_key = "__trace_events"
        last = self._last_seen.get(cursor_key, -1)
        rows = self._conn.execute(
            "SELECT step, wall_time, phase, duration_ms, details FROM trace_events "
            "WHERE step > ? ORDER BY step, phase",
            (last,),
        ).fetchall()
        if rows:
            self._last_seen[cursor_key] = rows[-1][0]
        return [
            {
                "step": r[0],
                "phase": r[2],
                "duration_ms": r[3],
                "details": json.loads(r[4]) if r[4] else {},
            }
            for r in rows
        ]

    # ── eval reads ───────────────────────────────────────────────────

    def read_eval_results(self, suite_name: str, step: int | None = None) -> list[dict]:
        """Read eval results for a suite, optionally filtered by step."""
        if step is not None:
            rows = self._conn.execute(
                "SELECT suite_name, case_id, step, wall_time, score_name, score_value, artifact_key, details "
                "FROM eval_results WHERE suite_name = ? AND step = ? ORDER BY case_id, score_name",
                (suite_name, step),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT suite_name, case_id, step, wall_time, score_name, score_value, artifact_key, details "
                "FROM eval_results WHERE suite_name = ? ORDER BY step, case_id, score_name",
                (suite_name,),
            ).fetchall()
        return [
            {
                "suite_name": r[0],
                "case_id": r[1],
                "step": r[2],
                "wall_time": r[3],
                "score_name": r[4],
                "score_value": r[5],
                "artifact_key": r[6],
                "details": json.loads(r[7]) if r[7] else {},
            }
            for r in rows
        ]

    def read_eval_results_incremental(self) -> list[dict]:
        """Read new eval results since last call. Uses cursor tracking."""
        self._check_session()
        cursor_key = "__eval_results"
        last = self._last_seen.get(cursor_key, -1)
        rows = self._conn.execute(
            "SELECT suite_name, case_id, step, wall_time, score_name, score_value, artifact_key, details "
            "FROM eval_results WHERE step > ? ORDER BY step, suite_name, case_id",
            (last,),
        ).fetchall()
        if rows:
            self._last_seen[cursor_key] = rows[-1][2]
        return [
            {
                "suite_name": r[0],
                "case_id": r[1],
                "step": r[2],
                "score_name": r[4],
                "score_value": r[5],
            }
            for r in rows
        ]

    def get_eval_suites(self) -> list[str]:
        """Return all distinct eval suite names."""
        rows = self._conn.execute(
            "SELECT DISTINCT suite_name FROM eval_results ORDER BY suite_name"
        ).fetchall()
        return [r[0] for r in rows]

    def get_trace_phases(self) -> list[str]:
        """Return all distinct trace phases."""
        rows = self._conn.execute(
            "SELECT DISTINCT phase FROM trace_events ORDER BY phase"
        ).fetchall()
        return [r[0] for r in rows]

    def get_active_session_id(self) -> str | None:
        """Return the active session ID from metadata."""
        row = self._conn.execute(
            "SELECT value FROM metadata WHERE key = 'active_session_id'"
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def get_resume_step(self, session_id: str) -> int | None:
        """Return the resume_step for a given session, or None."""
        row = self._conn.execute(
            "SELECT resume_step FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return row[0]

    # ── unified metrics ───────────────────────────────────────────────

    def get_all_metric_tags(self) -> dict:
        """Return all tags grouped by type with per-tag metadata (count, last_step, last_wall_time)."""
        result = {}
        for table, tag_col in [("scalars", "tag"), ("tensors", "tag"), ("artifacts", "tag"), ("text_events", "tag")]:
            rows = self._conn.execute(
                f"SELECT {tag_col}, COUNT(*) as cnt, MAX(step) as last_step, MAX(wall_time) as last_wt "  # noqa: S608
                f"FROM {table} GROUP BY {tag_col} ORDER BY {tag_col}",
            ).fetchall()
            result[table] = [
                {"tag": r[0], "count": r[1], "last_step": r[2], "last_wall_time": r[3]}
                for r in rows
            ]
        # audio, pr_curves, graphs, meshes, and embeddings may not exist in older databases
        for table in ("audio", "pr_curves", "graphs", "meshes", "embeddings"):
            try:
                rows = self._conn.execute(
                    f"SELECT tag, COUNT(*) as cnt, MAX(step) as last_step, MAX(wall_time) as last_wt "  # noqa: S608
                    f"FROM {table} GROUP BY tag ORDER BY tag",
                ).fetchall()
                result[table] = [
                    {"tag": r[0], "count": r[1], "last_step": r[2], "last_wall_time": r[3]}
                    for r in rows
                ]
            except Exception:
                result[table] = []
        return result

    def read_metric_timeseries(self, requests: list[dict]) -> list[dict]:
        """Batch fetch multiple tags across types.

        Each request: {"plugin": "scalars"|"tensors"|"artifacts"|"text_events", "tag": "...", "downsample": 100}
        Returns: [{"plugin": "...", "tag": "...", "run": "...", "data": [...]}]
        """
        results = []
        for req in requests:
            plugin = req.get("plugin", "scalars")
            tag = req.get("tag", "")
            downsample = req.get("downsample", 100)

            if plugin == "scalars":
                data = self.read_scalars_downsampled(tag, downsample)
                results.append({"plugin": plugin, "tag": tag, "data": [
                    {"step": r[0], "wall_time": r[1], "value": r[2]} for r in data
                ]})
            elif plugin == "tensors":
                data = self.read_histograms(tag, downsample)
                results.append({"plugin": plugin, "tag": tag, "data": data})
            elif plugin == "text_events":
                data = self.read_text(tag, limit=downsample)
                results.append({"plugin": plugin, "tag": tag, "data": data})
            elif plugin == "artifacts":
                data = self.read_artifacts(tag, downsample)
                results.append({"plugin": plugin, "tag": tag, "data": data})
            elif plugin == "audio":
                data = self.read_audio(tag, downsample)
                results.append({"plugin": plugin, "tag": tag, "data": data})
            elif plugin == "pr_curves":
                data = self.read_pr_curves(tag, downsample)
                results.append({"plugin": plugin, "tag": tag, "data": data})
            elif plugin == "graphs":
                data = self.read_graphs(tag, downsample)
                results.append({"plugin": plugin, "tag": tag, "data": data})
            elif plugin == "meshes":
                data = self.read_meshes(tag=tag, downsample=downsample)
                results.append({"plugin": plugin, "tag": tag, "data": data})
            elif plugin == "embeddings":
                data = self.read_embeddings(tag=tag, downsample=downsample)
                results.append({"plugin": plugin, "tag": tag, "data": data})
        return results

    # ── custom scalars ───────────────────────────────────────────────

    def get_custom_scalar_layout(self) -> dict | None:
        """Return the custom scalars layout config, or None if not set."""
        row = self._conn.execute(
            "SELECT config FROM custom_scalar_layouts WHERE layout_name = 'default'"
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def read_custom_scalars(self, tag_regexes: list[str], downsample: int = 5000) -> dict:
        """Read scalar data for tags matching any of the given regex patterns.

        Returns: {matched_tag: [[step, wall_time, value], ...], ...}
        """
        import re
        # Get all scalar tags
        all_tags = self._conn.execute("SELECT DISTINCT tag FROM scalars").fetchall()
        all_tags = [r[0] for r in all_tags]

        # Match tags against patterns
        matched_tags = set()
        compiled = []
        for pattern in tag_regexes:
            try:
                compiled.append(re.compile(pattern))
            except re.error:
                continue

        for tag in all_tags:
            for regex in compiled:
                if regex.search(tag):
                    matched_tags.add(tag)
                    break

        # Read data for matched tags
        result = {}
        for tag in sorted(matched_tags):
            rows = self.read_scalars_downsampled(tag, downsample)
            result[tag] = [[r[0], r[1], r[2]] for r in rows]
        return result

    # ── run notes ──────────────────────────────────────────────────────

    def _get_write_conn(self) -> sqlite3.Connection:
        """Return a writable connection to the same database.

        The main ``_conn`` is opened read-only (``?mode=ro``). Notes require
        write access, so we lazily create a second connection without the
        read-only flag.
        """
        if not hasattr(self, "_write_conn") or self._write_conn is None:
            self._write_conn = sqlite3.connect(self._db_path)
            self._write_conn.execute(
                "CREATE TABLE IF NOT EXISTS run_notes ("
                "  id INTEGER PRIMARY KEY,"
                "  note TEXT NOT NULL,"
                "  updated_at REAL NOT NULL"
                ")"
            )
            self._write_conn.commit()
        return self._write_conn

    def get_note(self) -> dict:
        """Return the current run note, or an empty placeholder."""
        # Ensure the table exists (it may not in older databases).
        try:
            row = self._conn.execute(
                "SELECT note, updated_at FROM run_notes ORDER BY id LIMIT 1"
            ).fetchone()
        except sqlite3.OperationalError:
            return {"note": "", "updated_at": None}
        if row is None:
            return {"note": "", "updated_at": None}
        return {"note": row[0], "updated_at": row[1]}

    def set_note(self, text: str) -> None:
        """Upsert the run note with the current wall time."""
        import time as _time

        conn = self._get_write_conn()
        now = _time.time()
        conn.execute(
            "INSERT INTO run_notes (id, note, updated_at) VALUES (1, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET note = excluded.note, updated_at = excluded.updated_at",
            (text, now),
        )
        conn.commit()
        # Refresh the read-only connection so get_note sees the new value
        # immediately without waiting for SQLite WAL checkpointing.
        try:
            self._conn.execute("SELECT 1")
        except sqlite3.OperationalError:
            pass

    # ── lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()
        if hasattr(self, "_write_conn") and self._write_conn is not None:
            self._write_conn.close()
