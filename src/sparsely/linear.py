import numpy as np
import mip

from sparsely.base import BaseRegressor
from sparsely.utils.cutting_planes import cutting_planes_optimizer, Result

from sklearn.preprocessing import StandardScaler

from typing import Any, Callable, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SparseLinearRegressor(BaseRegressor):

    n_selected_features: int
    l2_penalty: float = 0.1
    max_iter: int = 100
    convergence_tol: float = 1e-5
    max_seconds_per_cut: Optional[int] = None

    def __post_init__(self):
        super().__init__()
        self._scaler: StandardScaler = StandardScaler()
        self._weights: Optional[np.ndarray] = None
        self._solve_result: Optional[Result] = None

    def _validate_model(self) -> None:
        assert self.n_selected_features > 0
        assert self.l2_penalty > 0

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:

        # Scale features
        X = self._scaler.fit_transform(X=X)

        # Initialize model
        model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)

        # Define variables
        support = model.add_var_tensor(shape=(self.n_features,), name="support")

        # Set feature selection constraint
        model.add_constr(mip.xsum(support) <= self.n_selected_features)

        self._solve_result = cutting_planes_optimizer(
            func=self._make_callback(X=X, y=y),
            model=model,
            x=support,
            x0=None,
            max_iter=self.max_iter,
            convergence_tol=self.convergence_tol,
            max_seconds_per_cut=self.max_seconds_per_cut
        )

        # Compute weights
        self._weights, _ = self._solve_inner_problem(X=X, y=y, support=np.ndarray([s.x for s in support]))

    def _make_callback(self, X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], Tuple[float, np.ndarray]]:

        def func(support: np.ndarray) -> Tuple[float, np.ndarray]:

            # Solve inner problem
            _, dual_variables = self._solve_inner_problem(X=X, y=y, support=support)

            # Return objective value and gradient
            return 0.5 * np.dot(y, dual_variables), -1 / self.l2_penalty * (np.matmul(X, dual_variables) ** 2)

        return func

    def _warm_start(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def _solve_inner_problem(self, X: np.ndarray, y: np.ndarray, support: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        # Select feature subset
        support = np.round(support).astype(bool)
        X_subset = X[:, support]

        # Compute weights
        weights = np.matmul(
            np.invert(2 * self.l2_penalty * np.eye(support.sum()) - np.matmul(X_subset.T, X_subset)),
            np.matmul(X_subset.T, y)
        )

        # Compute dual variables
        dual_variables = y - np.matmul(X_subset, weights)

        return weights, dual_variables

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return np.matmul(self._scaler.transform(X=X), self._weights)

    @property
    def params(self):
        pass


