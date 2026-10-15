"""Gradient-boosted trees on the long (day x hour) feature table."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..config import HOURS, SEED


class GBM:
    """One model for all 24 hours; the hour of day is just another feature."""

    def __init__(self, **params):
        defaults = dict(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=1000,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            categorical_features=["dow"],
            random_state=SEED,
        )
        self.model = HistGradientBoostingRegressor(**{**defaults, **params})

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> GBM:
        self.model.fit(X, np.asarray(y).reshape(-1))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X).reshape(-1, HOURS)

    def staged_predict(self, X: pd.DataFrame):
        """Yield predictions after each boosting iteration (for choosing the number of trees)."""
        for pred in self.model.staged_predict(X):
            yield pred.reshape(-1, HOURS)
