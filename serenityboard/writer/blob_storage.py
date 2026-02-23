"""Content-addressed blob storage for SerenityBoard."""
from __future__ import annotations

import hashlib
import os
import sqlite3

__all__ = ["BlobStorage"]


class BlobStorage:
    """Content-addressed blob store backed by flat files.

    blob_key = sha256(data)[:16] + "." + extension
    Writes use tmp+rename for POSIX atomicity. Deduplicates on content hash.
    """

    def __init__(self, blobs_dir: str) -> None:
        """Create blob storage rooted at *blobs_dir* (created if missing)."""
        self._blobs_dir = blobs_dir
        os.makedirs(blobs_dir, exist_ok=True)

    def store(self, data: bytes, extension: str) -> str:
        """Store *data*, return blob_key. Skips write if already present."""
        blob_key = hashlib.sha256(data).hexdigest()[:16] + "." + extension
        path = os.path.join(self._blobs_dir, blob_key)
        if os.path.exists(path):
            return blob_key  # dedup
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.rename(tmp_path, path)
        return blob_key

    def get_path(self, blob_key: str) -> str:
        """Return the full filesystem path for *blob_key*."""
        return os.path.join(self._blobs_dir, blob_key)

    def gc(self, conn: sqlite3.Connection) -> int:
        """Delete orphaned blob files not referenced in the DB.

        Returns the number of files removed.
        """
        live = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT blob_key FROM artifacts"
            ).fetchall()
        }
        # Also include audio blobs so GC doesn't delete them
        try:
            live.update(
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT blob_key FROM audio"
                ).fetchall()
            )
        except sqlite3.OperationalError:
            pass  # audio table may not exist in older databases
        # Also include embedding blobs (matrix + sprite) so GC doesn't delete them
        try:
            live.update(
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT tensor_blob_key FROM embeddings"
                ).fetchall()
            )
            live.update(
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT sprite_blob_key FROM embeddings WHERE sprite_blob_key IS NOT NULL"
                ).fetchall()
            )
        except sqlite3.OperationalError:
            pass  # embeddings table may not exist in older databases
        # Also include mesh blobs (vertices, faces, colors) so GC doesn't delete them
        try:
            live.update(
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT vertices_blob_key FROM meshes"
                ).fetchall()
            )
            live.update(
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT faces_blob_key FROM meshes WHERE faces_blob_key IS NOT NULL"
                ).fetchall()
            )
            live.update(
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT colors_blob_key FROM meshes WHERE colors_blob_key IS NOT NULL"
                ).fetchall()
            )
        except sqlite3.OperationalError:
            pass  # meshes table may not exist in older databases
        # Also include graph blobs so GC doesn't delete them
        try:
            for row in conn.execute(
                "SELECT DISTINCT graph_blob_key FROM graphs"
            ).fetchall():
                # Skip legacy inline JSON values (not blob keys)
                if row[0] and not row[0].startswith("{"):
                    live.add(row[0])
        except sqlite3.OperationalError:
            pass  # graphs table may not exist or column may have old name
        removed = 0
        for entry in os.scandir(self._blobs_dir):
            if entry.is_file() and entry.name not in live:
                os.unlink(entry.path)
                removed += 1
        return removed
