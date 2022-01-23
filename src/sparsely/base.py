from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Callable, Union

import mip
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from sparsely.cutting_planes_optimizer import CuttingPlanesOptimizer


class BaseSparseLinearModel(BaseEstimator, metaclass=ABCMeta):

    def __init__(
            self,
            max_selected_features: int,
            l2_penalty: float,
            rescale: bool,
            max_iter: int,
            convergence_tol: float,
            max_seconds_per_cut: Optional[int],
            random_state: Optional[int],
            verbose: bool
    ):
        self.max_selected_features = max_selected_features
        self.l2_penalty = l2_penalty
        self.rescale = rescale
        self.max_iter = max_iter
        self.convergence_tol = convergence_tol
        self.max_seconds_per_cut = max_seconds_per_cut
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseSparseLinearModel:

        # Perform validation checks
        self._validate_hyperparameters()
        self._validate_data(X=X, y=y)

        # Rescale data
        if self.rescale:
            self._scaler = StandardScaler()
            X, y = self._scaler.fit_transform(X)

        # Initialize model
        model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)
        model.max_mip_gap = self.convergence_tol
        model.verbose = 0

        # Define variables
        support = model.add_var_tensor(shape=(self.n_features_in_,), var_type=mip.BINARY, name="support")

        # Set feature selection constraint
        model.add_constr(mip.xsum(support) <= self.max_selected_features)

        # Configure cutting planes optimizer
        self._optimizer = CuttingPlanesOptimizer(
            func=self._make_callback(X=X, y=y),
            model=model,
            x=support,
            max_iter=self.max_iter,
            convergence_tol=self.convergence_tol,
            max_seconds_per_cut=self.max_seconds_per_cut,
            verbose=self.verbose
        )

        # Optimize model weights
        self._optimizer.optimize(x0=self._initialize_support())
        self._weights = self._solver_inner_problem(
            X=X,
            y=y,
            support=self._optimizer.solution,
            return_weights=True
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:

        # Perform validation checks
        self._validate_data(X=X, y=None)

        # Rescale data
        if self.rescale:
            X = self._scaler.transform(X)

        return np.matmul(X, self._weights)

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    def _validate_hyperparameters(self):
        self._check_positive_int("max_selected_features")
        self._check_positive_real("l2_penalty")
        self._check_positive_int("max_iter")
        self._check_positive_real("convergence_tol")
        self._check_positive_int("max_seconds_per_cut", none_allowed=True)

    def _check_positive_int(self, name: str, none_allowed: bool = False) -> None:
        value = self.get_params()[name]
        if value is not None or not none_allowed:
            assert isinstance(value, int) and value > 0, (
                f"Invalid value for `{name}`. Positive integer required. Provided value is: {value}."
            )

    def _check_positive_real(self, name: str, none_allowed: bool = False) -> None:
        value = self.get_params()[name]
        if value is not None or not none_allowed:
            assert isinstance(value, (int, float)) and value > 0, (
                f"Invalid value for `{name}`. Positive real number required. Provided value is: {value}."
            )

    def _make_callback(self, X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], Tuple[float, np.ndarray]]:

        def func(support: np.ndarray) -> Tuple[float, np.ndarray]:

            # Solve inner problem
            objective_value, dual_variables = self._solver_inner_problem(X=X, y=y, support=support)

            # Return objective value and gradient
            return (
                objective_value,
                -1 / X.shape[0] / self.l2_penalty * (np.matmul(X.T, dual_variables) ** 2)
            )

        return func

    def _initialize_support(self) -> np.ndarray:
        random_state = np.random.RandomState(self.random_state)
        selected_features = random_state.choice(self.n_features_in_, self.max_selected_features, replace=False)
        return np.isin(np.arange(self.n_features_in_), selected_features).astype(float)

    @abstractmethod
    def _solver_inner_problem(
            self,
            X: np.ndarray,
            y: np.ndarray,
            support: np.ndarray,
            return_weights: bool = False
    ) -> Union[Tuple[float, np.ndarray], np.ndarray] :
        ...
