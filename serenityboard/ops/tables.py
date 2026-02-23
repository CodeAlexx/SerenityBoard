"""Tables plugin service layer for SerenityBoard ops."""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from serenityboard.ops.db import OpsDB

__all__ = ["TablesService", "validate_schema", "parse_filter", "VALID_KINDS"]

VALID_KINDS = frozenset(
    ["string", "number", "boolean", "timestamp", "enum", "json", "artifact_ref"]
)

VALID_ROLES = frozenset(["input", "output", "target", "reference"])

_FILTER_OPS = {"==", "!=", ">", ">=", "<", "<="}
_FILTER_MAX_DEPTH = 8
_FILTER_MAX_SIZE = 8192  # 8KB


def validate_schema(schema: dict) -> list[str]:
    """Validate a table schema dict. Returns list of error strings (empty = valid)."""
    errors = []
    if not isinstance(schema, dict):
        return ["schema must be an object"]
    columns = schema.get("columns")
    if not isinstance(columns, list) or len(columns) == 0:
        return ["schema.columns must be a non-empty array"]
    seen_names: set[str] = set()
    for i, col in enumerate(columns):
        if not isinstance(col, dict):
            errors.append(f"columns[{i}] must be an object")
            continue
        name = col.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"columns[{i}].name must be a non-empty string")
            continue
        if name in seen_names:
            errors.append(f"duplicate column name: {name!r}")
        seen_names.add(name)
        kind = col.get("kind")
        if kind not in VALID_KINDS:
            errors.append(
                f"columns[{i}].kind={kind!r} not in {sorted(VALID_KINDS)}"
            )
        if kind == "enum":
            values = col.get("values")
            if not isinstance(values, list) or len(values) == 0:
                errors.append(f"columns[{i}] enum must have non-empty 'values' array")
    return errors


def _get_column_names(schema: dict) -> set[str]:
    return {c["name"] for c in schema.get("columns", [])}


def _validate_filter_depth(node: Any, depth: int = 0) -> int:
    """Return max depth of filter expression tree."""
    if depth > _FILTER_MAX_DEPTH:
        return depth
    if isinstance(node, dict):
        op = node.get("op")
        if op in ("and", "or"):
            children = node.get("children", [])
            return max(
                (_validate_filter_depth(c, depth + 1) for c in children),
                default=depth,
            )
        if op == "not":
            return _validate_filter_depth(node.get("child"), depth + 1)
        # Leaf comparison
        return depth
    return depth


def parse_filter(
    filter_json: str | dict, schema: dict
) -> tuple[str | None, list, list[str]]:
    """Parse a filter DSL expression into a SQL WHERE clause.

    Returns (sql_fragment, params, errors).
    If errors is non-empty, sql_fragment is None.
    """
    if isinstance(filter_json, str):
        if len(filter_json) > _FILTER_MAX_SIZE:
            return None, [], [f"filter exceeds max size ({_FILTER_MAX_SIZE} bytes)"]
        try:
            filter_json = json.loads(filter_json)
        except json.JSONDecodeError as e:
            return None, [], [f"invalid filter JSON: {e}"]

    if not isinstance(filter_json, dict):
        return None, [], ["filter must be a JSON object"]

    col_names = _get_column_names(schema)
    errors: list[str] = []

    depth = _validate_filter_depth(filter_json)
    if depth > _FILTER_MAX_DEPTH:
        return None, [], [f"filter depth {depth} exceeds max {_FILTER_MAX_DEPTH}"]

    sql, params = _compile_filter(filter_json, col_names, errors)
    if errors:
        return None, [], errors
    return sql, params, []


def _compile_filter(
    node: dict, col_names: set[str], errors: list[str]
) -> tuple[str, list]:
    op = node.get("op")

    if op in ("and", "or"):
        children = node.get("children", [])
        if not children:
            errors.append(f"'{op}' requires at least one child")
            return "1=1", []
        parts = []
        params = []
        for child in children:
            sql, p = _compile_filter(child, col_names, errors)
            parts.append(f"({sql})")
            params.extend(p)
        joiner = " AND " if op == "and" else " OR "
        return joiner.join(parts), params

    if op == "not":
        child = node.get("child")
        if not child:
            errors.append("'not' requires a 'child'")
            return "1=1", []
        sql, params = _compile_filter(child, col_names, errors)
        return f"NOT ({sql})", params

    if op == "contains":
        col = node.get("column")
        value = node.get("value")
        if col not in col_names:
            errors.append(f"unknown column: {col!r}")
            return "1=1", []
        return (
            f"json_extract(values_json, '$.{col}') LIKE ?",
            [f"%{value}%"],
        )

    if op == "in":
        col = node.get("column")
        values = node.get("values", [])
        if col not in col_names:
            errors.append(f"unknown column: {col!r}")
            return "1=1", []
        if not values:
            errors.append("'in' filter requires a non-empty 'values' list")
            return "1=1", []
        placeholders = ",".join("?" for _ in values)
        return (
            f"json_extract(values_json, '$.{col}') IN ({placeholders})",
            list(values),
        )

    # Comparison operators
    if op in _FILTER_OPS:
        col = node.get("column")
        value = node.get("value")
        if col not in col_names:
            errors.append(f"unknown column: {col!r}")
            return "1=1", []
        return (
            f"json_extract(values_json, '$.{col}') {op} ?",
            [value],
        )

    errors.append(f"unknown filter op: {op!r}")
    return "1=1", []


class TablesService:
    """CRUD operations for the Tables plugin."""

    def __init__(self, db: OpsDB) -> None:
        self._db = db

    # ── Table Management ────────────────────────────────────────────

    def create_table(
        self, name: str, schema: dict, description: str = ""
    ) -> dict:
        errors = validate_schema(schema)
        if errors:
            raise ValueError("; ".join(errors))
        now = time.time()
        table_id = uuid.uuid4().hex
        self._db.conn.execute(
            "INSERT INTO tables_def (table_id, name, description, created_at, updated_at, schema_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (table_id, name, description, now, now, json.dumps(schema)),
        )
        self._db.conn.commit()
        return {
            "table_id": table_id,
            "name": name,
            "description": description,
            "schema": schema,
            "created_at": now,
            "updated_at": now,
        }

    def list_tables(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT table_id, name, description, created_at, updated_at, schema_json "
            "FROM tables_def ORDER BY created_at"
        ).fetchall()
        return [
            {
                "table_id": r["table_id"],
                "name": r["name"],
                "description": r["description"],
                "schema": json.loads(r["schema_json"]),
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def get_table(self, table_id: str) -> dict | None:
        row = self._db.execute(
            "SELECT table_id, name, description, created_at, updated_at, schema_json "
            "FROM tables_def WHERE table_id = ?",
            (table_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "table_id": row["table_id"],
            "name": row["name"],
            "description": row["description"],
            "schema": json.loads(row["schema_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def update_table(self, table_id: str, **kwargs) -> dict | None:
        tbl = self.get_table(table_id)
        if not tbl:
            return None
        sets = []
        params: list = []
        if "name" in kwargs:
            sets.append("name = ?")
            params.append(kwargs["name"])
        if "description" in kwargs:
            sets.append("description = ?")
            params.append(kwargs["description"])
        if "schema" in kwargs:
            errors = validate_schema(kwargs["schema"])
            if errors:
                raise ValueError("; ".join(errors))
            sets.append("schema_json = ?")
            params.append(json.dumps(kwargs["schema"]))
        if not sets:
            return tbl
        now = time.time()
        sets.append("updated_at = ?")
        params.append(now)
        params.append(table_id)
        self._db.conn.execute(
            f"UPDATE tables_def SET {', '.join(sets)} WHERE table_id = ?",
            tuple(params),
        )
        self._db.conn.commit()
        return self.get_table(table_id)

    def delete_table(self, table_id: str) -> bool:
        with self._db.conn:
            self._db.conn.execute(
                "DELETE FROM tables_row_artifacts WHERE row_id IN "
                "(SELECT row_id FROM tables_rows WHERE table_id = ?)",
                (table_id,),
            )
            self._db.conn.execute(
                "DELETE FROM tables_rows WHERE table_id = ?", (table_id,)
            )
            cur = self._db.conn.execute(
                "DELETE FROM tables_def WHERE table_id = ?", (table_id,)
            )
        return cur.rowcount > 0

    # ── Row Management ──────────────────────────────────────────────

    def insert_row(
        self,
        table_id: str,
        values: dict,
        run_name: str | None = None,
        step: int | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        now = time.time()
        row_id = uuid.uuid4().hex
        self._db.conn.execute(
            "INSERT INTO tables_rows "
            "(row_id, table_id, run_name, step, created_at, updated_at, values_json, tags_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row_id,
                table_id,
                run_name,
                step,
                now,
                now,
                json.dumps(values),
                json.dumps(tags or []),
            ),
        )
        self._db.conn.commit()
        return {
            "row_id": row_id,
            "table_id": table_id,
            "run_name": run_name,
            "step": step,
            "values": values,
            "tags": tags or [],
            "created_at": now,
            "updated_at": now,
        }

    def bulk_insert_rows(
        self, table_id: str, rows: list[dict]
    ) -> list[dict]:
        """Insert multiple rows in a single transaction. All-or-nothing."""
        now = time.time()
        results = []
        with self._db.conn:
            for row_data in rows:
                row_id = uuid.uuid4().hex
                values = row_data.get("values", {})
                run_name = row_data.get("run_name")
                step = row_data.get("step")
                tags = row_data.get("tags", [])
                self._db.conn.execute(
                    "INSERT INTO tables_rows "
                    "(row_id, table_id, run_name, step, created_at, updated_at, values_json, tags_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row_id,
                        table_id,
                        run_name,
                        step,
                        now,
                        now,
                        json.dumps(values),
                        json.dumps(tags),
                    ),
                )
                results.append(
                    {
                        "row_id": row_id,
                        "table_id": table_id,
                        "run_name": run_name,
                        "step": step,
                        "values": values,
                        "tags": tags,
                        "created_at": now,
                        "updated_at": now,
                    }
                )
        return results

    def query_rows(
        self,
        table_id: str,
        filter_expr: dict | str | None = None,
        sort: str | None = None,
        limit: int = 100,
        offset: int = 0,
        schema: dict | None = None,
    ) -> list[dict]:
        """Query rows with optional filter DSL, sort, limit, offset."""
        sql = "SELECT row_id, table_id, run_name, step, created_at, updated_at, values_json, tags_json FROM tables_rows WHERE table_id = ?"
        params: list = [table_id]

        if filter_expr is not None:
            if schema is None:
                tbl = self.get_table(table_id)
                if tbl:
                    schema = tbl["schema"]
                else:
                    schema = {"columns": []}
            where_sql, where_params, errors = parse_filter(filter_expr, schema)
            if errors:
                raise ValueError("; ".join(errors))
            sql += f" AND ({where_sql})"
            params.extend(where_params)

        if sort:
            # Simple sort: "column_name" or "-column_name" for DESC
            if sort.startswith("-"):
                sort_col = sort[1:]
                direction = "DESC"
            else:
                sort_col = sort
                direction = "ASC"
            _BUILTIN_SORT_COLS = {"created_at", "updated_at", "step", "run_name"}
            if sort_col in _BUILTIN_SORT_COLS:
                sql += f" ORDER BY {sort_col} {direction}"
            else:
                # Validate sort_col against schema to prevent SQL injection
                if schema is None:
                    tbl = self.get_table(table_id)
                    if tbl:
                        schema = tbl["schema"]
                    else:
                        schema = {"columns": []}
                valid_cols = _get_column_names(schema)
                if sort_col not in valid_cols:
                    raise ValueError(f"unknown sort column: {sort_col!r}")
                # Safe: sort_col is validated against schema column names;
                # additionally guard with identifier regex
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', sort_col):
                    raise ValueError(f"invalid sort column name: {sort_col!r}")
                sql += f" ORDER BY json_extract(values_json, '$.{sort_col}') {direction}"
        else:
            sql += " ORDER BY created_at DESC"

        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self._db.execute(sql, tuple(params)).fetchall()
        return [
            {
                "row_id": r["row_id"],
                "run_name": r["run_name"],
                "step": r["step"],
                "values": json.loads(r["values_json"]),
                "tags": json.loads(r["tags_json"]),
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def update_row(self, table_id: str, row_id: str, **kwargs) -> dict | None:
        row = self._db.execute(
            "SELECT * FROM tables_rows WHERE row_id = ? AND table_id = ?",
            (row_id, table_id),
        ).fetchone()
        if not row:
            return None
        sets = []
        params: list = []
        if "values" in kwargs:
            sets.append("values_json = ?")
            params.append(json.dumps(kwargs["values"]))
        if "tags" in kwargs:
            sets.append("tags_json = ?")
            params.append(json.dumps(kwargs["tags"]))
        if not sets:
            return self._row_to_dict(row)
        now = time.time()
        sets.append("updated_at = ?")
        params.append(now)
        params.extend([row_id, table_id])
        self._db.conn.execute(
            f"UPDATE tables_rows SET {', '.join(sets)} WHERE row_id = ? AND table_id = ?",
            tuple(params),
        )
        self._db.conn.commit()
        updated = self._db.execute(
            "SELECT * FROM tables_rows WHERE row_id = ?", (row_id,)
        ).fetchone()
        return self._row_to_dict(updated)

    def delete_row(self, table_id: str, row_id: str) -> bool:
        with self._db.conn:
            self._db.conn.execute(
                "DELETE FROM tables_row_artifacts WHERE row_id = ?", (row_id,)
            )
            cur = self._db.conn.execute(
                "DELETE FROM tables_rows WHERE row_id = ? AND table_id = ?",
                (row_id, table_id),
            )
        return cur.rowcount > 0

    @staticmethod
    def _row_to_dict(row) -> dict:
        return {
            "row_id": row["row_id"],
            "run_name": row["run_name"],
            "step": row["step"],
            "values": json.loads(row["values_json"]),
            "tags": json.loads(row["tags_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
