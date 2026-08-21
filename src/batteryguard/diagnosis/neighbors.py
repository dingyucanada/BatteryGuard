"""Auditable nearest-cell retrieval in the standardized feature space."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from batteryguard.prediction.baselines import numeric_feature_frame


@dataclass(frozen=True, slots=True)
class Neighbor:
    cell_id: str
    distance: float
    similarity: float


class NearestCellRetriever:
    def __init__(self) -> None:
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        self.index_: NearestNeighbors | None = None
        self.cell_ids_: np.ndarray | None = None
        self.feature_names_in_: tuple[str, ...] = ()

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        cell_ids: Iterable[str] | None = None,
    ) -> NearestCellRetriever:
        inferred_ids: list[str] | None = None
        if isinstance(X, pd.DataFrame) and "cell_id" in X.columns:
            inferred_ids = X["cell_id"].astype(str).tolist()
        ids = list(cell_ids) if cell_ids is not None else inferred_ids
        if ids is None:
            raise ValueError("cell_ids are required when X has no cell_id column")
        normalized_ids = np.asarray([str(value) for value in ids], dtype=object)
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("reference cell_ids must be unique")
        frame, columns = numeric_feature_frame(X)
        if len(frame) != normalized_ids.size or len(frame) == 0:
            raise ValueError("X and cell_ids must be non-empty with equal length")
        transformed = self.scaler.fit_transform(self.imputer.fit_transform(frame))
        self.index_ = NearestNeighbors(metric="euclidean", algorithm="auto")
        self.index_.fit(transformed)
        self.cell_ids_ = normalized_ids
        self.feature_names_in_ = columns
        return self

    def query_many(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        k: int = 5,
        exclude_cell_ids: Iterable[str | None] | None = None,
    ) -> list[list[Neighbor]]:
        if self.index_ is None or self.cell_ids_ is None:
            raise RuntimeError("fit NearestCellRetriever before query")
        if k < 1:
            raise ValueError("k must be positive")
        frame, _ = numeric_feature_frame(X, expected_columns=self.feature_names_in_)
        transformed = self.scaler.transform(self.imputer.transform(frame))
        exclusions = (
            list(exclude_cell_ids)
            if exclude_cell_ids is not None
            else [None] * len(frame)
        )
        if len(exclusions) != len(frame):
            raise ValueError("exclude_cell_ids length differs from query rows")
        requested = min(len(self.cell_ids_), k + 1)
        distance, indices = self.index_.kneighbors(transformed, n_neighbors=requested)
        output: list[list[Neighbor]] = []
        for row_distance, row_indices, excluded in zip(distance, indices, exclusions, strict=True):
            neighbors: list[Neighbor] = []
            for value, index in zip(row_distance, row_indices, strict=True):
                cell_id = str(self.cell_ids_[index])
                if excluded is not None and cell_id == str(excluded):
                    continue
                numeric_distance = float(value)
                neighbors.append(
                    Neighbor(
                        cell_id=cell_id,
                        distance=numeric_distance,
                        similarity=float(1.0 / (1.0 + numeric_distance)),
                    )
                )
                if len(neighbors) == k:
                    break
            output.append(neighbors)
        return output

    def query(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        k: int = 5,
        exclude_cell_id: str | None = None,
    ) -> list[Neighbor]:
        if len(X) != 1:
            raise ValueError("query accepts exactly one row; use query_many for batches")
        return self.query_many(X, k=k, exclude_cell_ids=[exclude_cell_id])[0]
