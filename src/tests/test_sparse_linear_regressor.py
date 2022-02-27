from typing import Tuple

import numpy as np
import pytest

from sparsely import SparseLinearRegressor


@pytest.mark.parametrize(
    "model",
    [
        SparseLinearRegressor(
            max_selected_features=1,
            random_state=123
        ),
        SparseLinearRegressor(
            max_selected_features=1,
            max_seconds_per_cut=25,
            max_iter=50,
            random_state=42,
            rescale=False,
            verbose=True
        )
    ]
)
class TestSparseLinearRegressor:

    def test_fit(self, model: SparseLinearRegressor, regression_dataset: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        X, y, weights = regression_dataset
        model.max_selected_features = (~np.isclose(weights, 0)).sum()
        model.fit(X=X, y=y)
        assert len(model.weights) == model.n_features_in_, (
            f"Number of model weights ({len(model.weights)}) does number of features in ({model.n_features_in_})."
        )
        assert (~np.isclose(model.weights, 0)).sum() <= model.max_selected_features, (
            f"The number of non-zero model weights ({(~np.isclose(model.weights, 0)).sum()}) exceeds the maximum"
            f"selected features of the model ({model.max_selected_features})."
        )

    def test_predict(self, model: SparseLinearRegressor, regression_dataset: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        X, y, _ = regression_dataset
        predicted = model.predict(X)
        assert len(y) == len(predicted), (
            f"The number of predictions ({len(predicted)}) does not match the number of samples ({len(y)})."
        )
        assert not np.isnan(predicted).any(), (
            f"The predictions include {np.isnan(predicted).sum()} NaN values."
        )

    def test_score(self, model: SparseLinearRegressor, regression_dataset: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        X, y, _ = regression_dataset
        score = model.score(X, y)
        assert isinstance(score, float), (
            f"The score not a float and is of type '{type(score)} instead."
        )
        assert score <= 1, (
            f"The R^2 score ({score}) exceeds the upper bound of 1."
        )
