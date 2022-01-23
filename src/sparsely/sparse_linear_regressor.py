from typing import Optional, Tuple, Union

import numpy as np
from sklearn.base import RegressorMixin

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

    def _solver_inner_problem(
            self,
            X: np.ndarray,
            y: np.ndarray,
            support: np.ndarray,
            return_weights: bool = False
    ) -> Union[Tuple[float, np.ndarray], np.ndarray]:

        # Select features
        support = np.round(support).astype(bool)
        X_subset = X[:, support]

        # Compute subset of non-zero weights
        weights_subset = np.matmul(
            np.linalg.inv(2 * self.l2_penalty * X.shape[0] * np.eye(support.sum()) + np.matmul(X_subset.T, X_subset)),
            np.matmul(X_subset.T, y)
        )

        # If `return_weights=True`, return the model weights
        if return_weights:
            weights = np.zeros(self.n_features_in_)
            weights[support] = weights_subset
            return weights

        # Else, return the objective and dual variables
        dual_variables = y - np.matmul(X_subset, weights_subset)
        return 0.5 * np.dot(y, dual_variables), dual_variables


