"""Blind-demo cell pool with a deliberately narrow public surface."""

from __future__ import annotations

import copy
import random
import threading
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, SupportsIndex

from pydantic import BaseModel

from batteryguard.evidence.hashing import canonical_json
from batteryguard.schemas.data import CellRecord


class BlindPoolError(RuntimeError):
    """Base exception for blind-pool failures."""


class BlindCellNotFoundError(BlindPoolError, KeyError):
    """Raised when a requested cell is not part of the blind pool."""


class CycleLifeUnavailableError(BlindPoolError):
    """Raised if a pool entry has no observed full lifetime."""


class CellAlreadyRevealedError(BlindPoolError):
    """Raised when a reveal is complete or already in progress for a cell."""


_REVEAL_CAPABILITY = object()
_SENSITIVE_KEYS = frozenset({"cycle_life", "actual_cycle_life", "lifetime"})


def _plain_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_value(item) for item in value]
    return copy.deepcopy(value)


def _scrub_sensitive(value: Any) -> Any:
    """Recursively remove lifetime labels from anything crossing the public boundary."""

    plain = _plain_value(value)
    if isinstance(plain, Mapping):
        return {
            str(key): _scrub_sensitive(item)
            for key, item in plain.items()
            if str(key).casefold() not in _SENSITIVE_KEYS
        }
    if isinstance(plain, list):
        return [_scrub_sensitive(item) for item in plain]
    return plain


class BlindPool:
    """Store hidden lifetime labels and expose only sanitized early-cycle data.

    ``cells`` can be an iterable of :class:`CellRecord`/mapping objects or a
    mapping keyed by cell id.  Early-cycle rows may be supplied separately or in
    each cell mapping under ``early_cycles``.  All public returns are deep copies,
    and lifetime-like fields are recursively removed.
    """

    __slots__ = (
        "_hidden_cycle_life",
        "_lock",
        "_public_cells",
        "_reveal_in_progress",
        "_revealed",
    )

    def __init__(
        self,
        cells: Iterable[CellRecord | Mapping[str, Any]]
        | Mapping[str, CellRecord | Mapping[str, Any]],
        early_cycles: Mapping[str, Iterable[BaseModel | Mapping[str, Any]]] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._public_cells: dict[str, dict[str, Any]] = {}
        self._hidden_cycle_life: dict[str, int] = {}
        self._revealed: set[str] = set()
        self._reveal_in_progress: set[str] = set()
        early_by_cell = dict(early_cycles or {})

        if isinstance(cells, Mapping):
            entries: Iterable[tuple[str | None, CellRecord | Mapping[str, Any]]] = (
                (str(key), value) for key, value in cells.items()
            )
        else:
            entries = ((None, value) for value in cells)

        for keyed_cell_id, source in entries:
            source_data = _plain_value(source)
            if not isinstance(source_data, Mapping):
                raise TypeError("each blind-pool cell must be a CellRecord or mapping")
            cell_data = dict(source_data)
            embedded_early = cell_data.pop("early_cycles", None)
            declared_metadata = cell_data.pop("metadata", {})
            if declared_metadata is None:
                declared_metadata = {}
            if not isinstance(declared_metadata, Mapping):
                raise TypeError("cell metadata must be a mapping")

            if keyed_cell_id is not None:
                declared_cell_id = cell_data.get("cell_id", keyed_cell_id)
                if str(declared_cell_id) != keyed_cell_id:
                    raise ValueError(
                        f"cell mapping key {keyed_cell_id!r} disagrees with its cell_id"
                    )
                cell_data["cell_id"] = keyed_cell_id

            known_names = set(CellRecord.model_fields)
            extras = {key: cell_data.pop(key) for key in list(cell_data) if key not in known_names}
            cell = CellRecord.model_validate(cell_data)
            if cell.cycle_life is None:
                raise CycleLifeUnavailableError(
                    f"blind cell {cell.cell_id!r} has no observed cycle_life"
                )
            if cell.cell_id in self._public_cells:
                raise ValueError(f"duplicate blind cell_id {cell.cell_id!r}")

            if cell.cell_id in early_by_cell and embedded_early is not None:
                raise ValueError(
                    f"early cycles for {cell.cell_id!r} were supplied twice"
                )
            selected_early = early_by_cell.pop(cell.cell_id, embedded_early or [])
            if isinstance(selected_early, (str, bytes, bytearray)) or not isinstance(
                selected_early, Iterable
            ):
                raise TypeError("early_cycles must be an iterable of rows")

            safe_early: list[dict[str, Any]] = []
            for row in selected_early:
                row_data = _scrub_sensitive(row)
                if not isinstance(row_data, Mapping):
                    raise TypeError("each early-cycle row must be a model or mapping")
                row_cell_id = row_data.get("cell_id")
                if row_cell_id is not None and str(row_cell_id) != cell.cell_id:
                    raise ValueError(
                        f"early-cycle row for {row_cell_id!r} attached to {cell.cell_id!r}"
                    )
                safe_early.append(dict(row_data))

            public = cell.model_dump(mode="json", exclude={"cycle_life"})
            public["metadata"] = _scrub_sensitive({**dict(declared_metadata), **extras})
            public["early_cycles"] = safe_early
            self._public_cells[cell.cell_id] = _scrub_sensitive(public)
            self._hidden_cycle_life[cell.cell_id] = cell.cycle_life

        if early_by_cell:
            unknown = sorted(str(cell_id) for cell_id in early_by_cell)
            raise BlindCellNotFoundError(f"early cycles reference unknown cells: {unknown}")
        if not self._public_cells:
            raise ValueError("blind pool must contain at least one cell")

    @classmethod
    def from_records(
        cls,
        cells: Iterable[CellRecord | Mapping[str, Any]],
        early_cycles: Mapping[str, Iterable[BaseModel | Mapping[str, Any]]] | None = None,
    ) -> BlindPool:
        """Named constructor for callers loading normalized records."""

        return cls(cells, early_cycles)

    def list_public_cells(self) -> list[dict[str, Any]]:
        """List sanitized cells in deterministic id order."""

        with self._lock:
            return [copy.deepcopy(self._public_cells[cell_id]) for cell_id in self.cell_ids]

    def get_public_cell(self, cell_id: str) -> dict[str, Any]:
        """Return one sanitized cell; never returns a lifetime label."""

        with self._lock:
            try:
                return copy.deepcopy(self._public_cells[cell_id])
            except KeyError as exc:
                raise BlindCellNotFoundError(cell_id) from exc

    def random_public_cell(self, *, seed: int | None = None) -> dict[str, Any]:
        """Select a public cell reproducibly when a seed is provided."""

        generator = random.Random(seed)
        return self.get_public_cell(generator.choice(self.cell_ids))

    @property
    def cell_ids(self) -> tuple[str, ...]:
        """Return only non-sensitive identifiers."""

        with self._lock:
            return tuple(sorted(self._public_cells))

    def has_cell(self, cell_id: str) -> bool:
        with self._lock:
            return cell_id in self._public_cells

    def is_revealed(self, cell_id: str) -> bool:
        with self._lock:
            if cell_id not in self._public_cells:
                raise BlindCellNotFoundError(cell_id)
            return cell_id in self._revealed

    def model_dump(self) -> dict[str, Any]:
        """Safe Pydantic-style serialization convenience."""

        return {"cells": self.list_public_cells()}

    def keys(self) -> tuple[str, ...]:
        """Support safe ``dict(pool)`` and FastAPI generic serialization."""

        return ("cells",)

    def __getitem__(self, key: str) -> Any:
        if key != "cells":
            raise KeyError(key)
        return self.list_public_cells()

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json(self) -> str:
        return canonical_json(self.model_dump())

    def json(self) -> str:
        """Safe legacy serialization alias."""

        return self.to_json()

    def _begin_reveal(self, cell_id: str, capability: object) -> int:
        if capability is not _REVEAL_CAPABILITY:
            raise PermissionError("hidden lifetime requires the reveal service capability")
        with self._lock:
            try:
                actual = self._hidden_cycle_life[cell_id]
            except KeyError as exc:
                raise BlindCellNotFoundError(cell_id) from exc
            if cell_id in self._revealed or cell_id in self._reveal_in_progress:
                raise CellAlreadyRevealedError(cell_id)
            self._reveal_in_progress.add(cell_id)
            return actual

    def _finish_reveal(self, cell_id: str, capability: object, *, success: bool) -> None:
        if capability is not _REVEAL_CAPABILITY:
            raise PermissionError("reveal state requires the reveal service capability")
        with self._lock:
            if cell_id not in self._public_cells:
                raise BlindCellNotFoundError(cell_id)
            self._reveal_in_progress.discard(cell_id)
            if success:
                self._revealed.add(cell_id)

    def __len__(self) -> int:
        return len(self.cell_ids)

    def __contains__(self, cell_id: object) -> bool:
        return isinstance(cell_id, str) and self.has_cell(cell_id)

    def __repr__(self) -> str:
        return f"BlindPool(cell_ids={self.cell_ids!r})"

    def __reduce_ex__(self, protocol: SupportsIndex) -> Any:
        """Prevent pickle from becoming an accidental hidden-label export path."""

        raise TypeError(
            f"BlindPool cannot be pickled with protocol {protocol}; use to_dict()"
        )
