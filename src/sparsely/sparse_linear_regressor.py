from typing import Optional

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model import Ridge

from sparsely.base import BaseSparseLinearModel


class SparseLinearRegressor(BaseSparseLinearModel, RegressorMixin):

    def __init__(
            self,
            max_selected_features: int,
            l2_penalty: float = 0.1,
            rescale: bool = True,
            max_iter: int = 100,
            convergence_tol: float = 1e-5,
            max_seconds_per_cut: Optional[int] = None,
            random_state: Optional[int] = None,
            verbose: bool = False
    ):
        super().__init__(
            max_selected_features=max_selected_features,
            l2_penalty=l2_penalty,
            rescale=rescale,
            max_iter=max_iter,
            convergence_tol=convergence_tol,
            max_seconds_per_cut=max_seconds_per_cut,
            random_state=random_state,
            verbose=verbose
        )

    def _initialize_support(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.isin(
            np.arange(self.n_features_in_),
            np.argsort(
                -np.abs(Ridge(alpha=self.l2_penalty * len(X) / self.max_selected_features).fit(X=X, y=y).coef_)
            )[:self.max_selected_features]
        )

    def _compute_weights_for_subset(self, X_subset: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.matmul(
            np.linalg.inv(
                2 * self.l2_penalty * X_subset.shape[0] / self.max_selected_features * np.eye(X_subset.shape[1])
                + np.matmul(X_subset.T, X_subset)
            ),
            np.matmul(X_subset.T, y)
        )

    def _compute_dual_variables(self, X_subset: np.ndarray, y: np.ndarray, weights_subset: np.ndarray) -> np.ndarray:
        return y - np.matmul(X_subset, weights_subset)

    def _compute_objective_value(self,  X_subset: np.ndarray, y: np.ndarray, dual_variables: np.ndarray) -> float:
        return 0.5 * np.dot(y, dual_variables)





