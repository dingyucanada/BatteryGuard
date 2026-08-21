from __future__ import annotations

import numpy as np
import pandas as pd

from batteryguard.prediction import RidgeLifetimeModel, grouped_evaluate


def test_grouped_evaluation_predicts_each_cell_once() -> None:
    protocols = np.repeat(["P0", "P1", "P2", "P3"], 8)
    x = np.tile(np.linspace(-1.0, 1.0, 8), 4)
    X = pd.DataFrame(
        {
            "cell_id": [f"C{i}" for i in range(len(x))],
            "capacity_slope": x,
            "temperature": np.repeat([0.0, 0.1, 0.2, 0.3], 8),
        }
    )
    y = 800 + 100 * x + 10 * X["temperature"].to_numpy()
    result = grouped_evaluate(
        lambda: RidgeLifetimeModel(alpha=0.01),
        X,
        y,
        protocols,
        n_splits=4,
    )
    assert np.isfinite(result.predictions).all()
    assert set(result.fold_index) == {0, 1, 2, 3}
    assert result.report.overall.mae < 2.0
    assert set(result.report.by_group) == {"P0", "P1", "P2", "P3"}
