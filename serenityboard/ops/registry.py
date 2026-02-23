"""Artifact Registry service layer for SerenityBoard ops."""
from __future__ import annotations

import json
import time
import uuid

from serenityboard.ops.db import OpsDB

__all__ = ["RegistryService"]

VALID_KINDS = frozenset(["dataset", "model", "checkpoint", "eval_report", "other"])
VALID_RELATIONS = frozenset(["trained_from", "derived_from", "evaluated_with"])


class RegistryService:
    """CRUD for artifact collections, versions, aliases, and lineage."""

    def __init__(self, db: OpsDB) -> None:
        self._db = db

    # ── Collections ─────────────────────────────────────────────────

    def create_collection(
        self,
        name: str,
        kind: str,
        description: str = "",
    ) -> dict:
        if kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {sorted(VALID_KINDS)}")
        now = time.time()
        cid = uuid.uuid4().hex
        self._db.conn.execute(
            "INSERT INTO artifact_collections "
            "(collection_id, name, kind, description, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cid, name, kind, description, now, now),
        )
        self._db.conn.commit()
        return {
            "collection_id": cid,
            "name": name,
            "kind": kind,
            "description": description,
            "created_at": now,
            "updated_at": now,
        }

    def list_collections(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT * FROM artifact_collections ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_collection(self, collection_id: str) -> dict | None:
        row = self._db.execute(
            "SELECT * FROM artifact_collections WHERE collection_id = ?",
            (collection_id,),
        ).fetchone()
        return dict(row) if row else None

    def update_collection(self, collection_id: str, **kwargs) -> dict | None:
        coll = self.get_collection(collection_id)
        if not coll:
            return None
        sets, params = [], []
        if "description" in kwargs:
            sets.append("description = ?")
            params.append(kwargs["description"])
        if "kind" in kwargs:
            if kwargs["kind"] not in VALID_KINDS:
                raise ValueError(f"kind must be one of {sorted(VALID_KINDS)}")
            sets.append("kind = ?")
            params.append(kwargs["kind"])
        if not sets:
            return coll
        now = time.time()
        sets.append("updated_at = ?")
        params.append(now)
        params.append(collection_id)
        self._db.conn.execute(
            f"UPDATE artifact_collections SET {', '.join(sets)} WHERE collection_id = ?",
            tuple(params),
        )
        self._db.conn.commit()
        return self.get_collection(collection_id)

    # ── Versions ────────────────────────────────────────────────────

    def create_version(
        self,
        collection_id: str,
        digest: str,
        size_bytes: int,
        storage_uri: str,
        metadata: dict | None = None,
        created_by_run: str | None = None,
    ) -> dict:
        now = time.time()
        aid = uuid.uuid4().hex

        with self._db.conn:
            # Get next version index
            row = self._db.conn.execute(
                "SELECT COALESCE(MAX(version_index), -1) + 1 AS next_idx "
                "FROM artifact_versions WHERE collection_id = ?",
                (collection_id,),
            ).fetchone()
            next_idx = row["next_idx"]

            self._db.conn.execute(
                "INSERT INTO artifact_versions "
                "(artifact_id, collection_id, version_index, digest, size_bytes, "
                "storage_uri, metadata_json, created_by_run, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    aid,
                    collection_id,
                    next_idx,
                    digest,
                    size_bytes,
                    storage_uri,
                    json.dumps(metadata or {}),
                    created_by_run,
                    now,
                ),
            )

            # Auto-update "latest" alias
            self._db.conn.execute(
                "INSERT OR REPLACE INTO artifact_aliases "
                "(collection_id, alias, artifact_id, updated_at) "
                "VALUES (?, 'latest', ?, ?)",
                (collection_id, aid, now),
            )

        return {
            "artifact_id": aid,
            "collection_id": collection_id,
            "version_index": next_idx,
            "digest": digest,
            "size_bytes": size_bytes,
            "storage_uri": storage_uri,
            "metadata": metadata or {},
            "created_by_run": created_by_run,
            "created_at": now,
        }

    def list_versions(
        self,
        collection_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        rows = self._db.execute(
            "SELECT * FROM artifact_versions WHERE collection_id = ? "
            "ORDER BY version_index DESC LIMIT ? OFFSET ?",
            (collection_id, limit, offset),
        ).fetchall()
        return [
            {
                **{k: r[k] for k in r.keys() if k != "metadata_json"},
                "metadata": json.loads(r["metadata_json"]),
            }
            for r in rows
        ]

    def get_artifact(self, artifact_id: str) -> dict | None:
        row = self._db.execute(
            "SELECT * FROM artifact_versions WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if not row:
            return None
        return {
            **{k: row[k] for k in row.keys() if k != "metadata_json"},
            "metadata": json.loads(row["metadata_json"]),
        }

    # ── Aliases ─────────────────────────────────────────────────────

    def set_alias(
        self, collection_id: str, alias: str, artifact_id: str
    ) -> dict:
        now = time.time()
        self._db.conn.execute(
            "INSERT OR REPLACE INTO artifact_aliases "
            "(collection_id, alias, artifact_id, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (collection_id, alias, artifact_id, now),
        )
        self._db.conn.commit()
        return {
            "collection_id": collection_id,
            "alias": alias,
            "artifact_id": artifact_id,
            "updated_at": now,
        }

    def list_aliases(self, collection_id: str) -> list[dict]:
        rows = self._db.execute(
            "SELECT * FROM artifact_aliases WHERE collection_id = ?",
            (collection_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def resolve_alias(self, collection_id: str, alias: str) -> dict | None:
        row = self._db.execute(
            "SELECT artifact_id FROM artifact_aliases "
            "WHERE collection_id = ? AND alias = ?",
            (collection_id, alias),
        ).fetchone()
        if not row:
            return None
        return self.get_artifact(row["artifact_id"])

    # ── Lineage ─────────────────────────────────────────────────────

    def add_lineage(
        self,
        parent_artifact_id: str,
        child_artifact_id: str,
        relation: str,
    ) -> dict:
        if relation not in VALID_RELATIONS:
            raise ValueError(f"relation must be one of {sorted(VALID_RELATIONS)}")
        now = time.time()
        self._db.conn.execute(
            "INSERT INTO artifact_lineage "
            "(parent_artifact_id, child_artifact_id, relation, created_at) "
            "VALUES (?, ?, ?, ?)",
            (parent_artifact_id, child_artifact_id, relation, now),
        )
        self._db.conn.commit()
        return {
            "parent_artifact_id": parent_artifact_id,
            "child_artifact_id": child_artifact_id,
            "relation": relation,
            "created_at": now,
        }

    def get_lineage(
        self,
        artifact_id: str,
        direction: str = "down",
        depth: int = 3,
    ) -> list[dict]:
        """Get lineage edges. direction='up' = parents, 'down' = children."""
        results = []
        visited = set()
        queue = [artifact_id]

        for _ in range(depth):
            if not queue:
                break
            next_queue = []
            for aid in queue:
                if aid in visited:
                    continue
                visited.add(aid)
                if direction == "up":
                    rows = self._db.execute(
                        "SELECT * FROM artifact_lineage WHERE child_artifact_id = ?",
                        (aid,),
                    ).fetchall()
                    for r in rows:
                        results.append(dict(r))
                        next_queue.append(r["parent_artifact_id"])
                else:
                    rows = self._db.execute(
                        "SELECT * FROM artifact_lineage WHERE parent_artifact_id = ?",
                        (aid,),
                    ).fetchall()
                    for r in rows:
                        results.append(dict(r))
                        next_queue.append(r["child_artifact_id"])
            queue = next_queue

        return results
