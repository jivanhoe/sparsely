from typing import Optional

import numpy as np
from sklearn.base import ClassifierMixin

from sparsely.base import BaseSparseLinearModel


class SparseLinearClassifier(BaseSparseLinearModel, ClassifierMixin):

    def __init__(
            self,
            max_selected_features: int,
            loss: str = "entropy",
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
        self.loss = loss

    def _solver_inner_problem(
            self,
            X: np.ndarray,
            y: np.ndarray,
            support: np.ndarray,
            return_dual: bool
    ) -> np.ndarray:
        pass

