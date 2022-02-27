from typing import Optional

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression

from sparsely.base import BaseSparseLinearModel


class SparseLinearClassifier(BaseSparseLinearModel, ClassifierMixin):

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
                -np.abs(
                    LogisticRegression(
                        C=2 * self.max_selected_features / len(X) / self.l2_penalty
                    ).fit(X=X, y=y).coef_
                )
            )[:self.max_selected_features]
        )

    def _compute_weights_for_subset(self, X_subset: np.ndarray, y: np.ndarray) -> np.ndarray:
        return LogisticRegression(
            C=2 * self.max_selected_features / X_subset.shape[0] / self.l2_penalty
        ).fit(X=X_subset, y=y).coef_.flatten()

    def _compute_dual_variables(self, X_subset: np.ndarray, y: np.ndarray, weights_subset: np.ndarray) -> np.ndarray:
        return -y / (1 + np.exp(y * np.matmul(X_subset, weights_subset)))

    def _compute_objective_value(self,  X_subset: np.ndarray, y: np.ndarray, dual_variables: np.ndarray) -> float:
        return (
            (
                y * dual_variables * np.log(-y * dual_variables)
                - (1 + y * dual_variables) * np.log(1 + y * dual_variables)
            ).sum()
            - self.max_selected_features / np.sqrt(X_subset.shape[0]) / self.l2_penalty
            * (np.matmul(X_subset.T, dual_variables) ** 2).sum()
        )
