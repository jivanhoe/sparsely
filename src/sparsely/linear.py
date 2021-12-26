from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import mip
import numpy as np
from sklearn.preprocessing import StandardScaler

from sparsely.base import BaseRegressor
from sparsely.utils.cutting_planes import cutting_planes_optimizer, Result


@dataclass
class SparseLinearRegressor(BaseRegressor):

    n_selected_features: int
    l2_penalty: float = 0.1
    rescale: bool = True
    max_iter: int = 100
    convergence_tol: float = 1e-5
    max_seconds_per_cut: Optional[int] = None
    random_state: Optional[int] = None
    verbose: bool = False

    def __post_init__(self):
        super().__post_init__()

        assert self.n_selected_features > 0
        assert self.l2_penalty > 0
        assert self.max_iter > 0
        assert self.convergence_tol > 0
        if self.max_seconds_per_cut:
            assert self.max_seconds_per_cut > 0

        self._scaler: Optional[StandardScaler] = StandardScaler() if self.rescale else None
        self._weights: Optional[np.ndarray] = None
        self._solve_result: Optional[Result] = None

    def _validate_model(self) -> None:
        assert self.n_selected_features > 0
        assert self.l2_penalty > 0

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:

        # Set seed
        if self.random_state:
            np.random.seed(self.random_state)

        # Scale features
        if self.rescale:
            X = self._scaler.fit_transform(X=X)

        # Initialize model
        model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)
        model.max_mip_gap = self.convergence_tol
        model.verbose = 0

        # Define variables
        support = model.add_var_tensor(shape=(self.n_features,), var_type=mip.BINARY, name="support")

        # Set feature selection constraint
        model.add_constr(mip.xsum(support) <= self.n_selected_features)

        # Solve problem using cutting planes algorithm
        self._solve_result = cutting_planes_optimizer(
            func=self._make_callback(X=X, y=y),
            model=model,
            x=support,
            x0=self._initialize_support(),
            max_iter=self.max_iter,
            convergence_tol=self.convergence_tol,
            max_seconds_per_cut=self.max_seconds_per_cut,
            verbose=self.verbose
        )

        self._weights = self._solve_inner_problem(X=X, y=y, support=np.array([s.x for s in support]))

    def _make_callback(self, X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], Tuple[float, np.ndarray]]:

        def func(support: np.ndarray) -> Tuple[float, np.ndarray]:

            # Solve inner problem
            dual_variables = self._solve_inner_problem(X=X, y=y, support=support, return_dual=True)

            # Return objective value and gradient
            return (
                0.5 * np.dot(y, dual_variables),
                -1 / X.shape[0] / self.l2_penalty * (np.matmul(X.T, dual_variables) ** 2)
            )

        return func

    def _initialize_support(self) -> np.ndarray:
        selected_features = np.random.choice(self.n_features, self.n_selected_features, replace=False)
        return np.isin(np.arange(self._n_features), selected_features).astype(float)

    def _solve_inner_problem(
            self,
            X: np.ndarray,
            y: np.ndarray,
            support: np.ndarray,
            return_dual: bool = False
    ) -> np.ndarray:

        # Select features
        support = np.round(support).astype(bool)
        X_subset = X[:, support]

        # Compute subset of non-zero weights
        weights_subset = np.matmul(
            np.linalg.inv(2 * self.l2_penalty * X.shape[0] * np.eye(support.sum()) + np.matmul(X_subset.T, X_subset)),
            np.matmul(X_subset.T, y)
        )

        # If `return_dual` is True, compute and return dual variables
        if return_dual:
            return y - np.matmul(X_subset, weights_subset)

        # Else return weights
        weights = np.zeros(self.n_features)
        weights[support] = weights_subset
        return weights

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return np.matmul(self._scaler.transform(X=X) if self.rescale else X, self._weights)

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def solve_result(self) -> Result:
        return self._solve_result



