from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import mip
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sparsely.base import BaseUnsupervisedModel
from sparsely.utils.cutting_planes import cutting_planes_optimizer, Result


@dataclass
class SparseKMeans(BaseUnsupervisedModel):

    n_clusters: int
    n_selected_features: int
    n_restarts: int = 10
    max_correlation: float = 0.9
    max_iter: int = 100
    convergence_tol: float = 1e-5
    max_seconds_per_cut: Optional[int] = None
    random_state: Optional[int] = None,

    def __post_init__(self):
        super().__post_init__()

        assert self.n_clusters > 1
        assert self.n_selected_features > 0
        assert self.n_restarts > 0
        if self.max_correlation is not None:
            assert 1 > self.max_correlation > 0
        assert self.max_iter > 0
        assert self.convergence_tol > 0
        if self.max_seconds_per_cut:
            assert self.max_seconds_per_cut > 0

        self._estimator: KMeans = KMeans(n_clusters=self.n_clusters)
        self._scaler: StandardScaler = StandardScaler()
        self._selected_features: Optional[np.ndarray] = None
        self._iter: int = 0
        self._solve_results: List[Result] = list()

    def _fit(self, X: np.ndarray) -> None:

        # Standardize data
        X = self._scaler.fit_transform(X=X)

        # Set seed
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Make callback to compute objective value and gradient
        func = self._make_callback(X=X)

        # Initialize best objective value
        best_objective_value = -np.inf

        # Compute correlation matrix
        corr = np.abs(np.corrcoef(X))

        for _ in range(self.n_restarts):

            # Initialize MIP model
            model = mip.Model(solver_name=mip.CBC)
            model.max_mip_gap = self.convergence_tol
            model.verbose = 0

            # Define variables
            support = model.add_var_tensor(shape=(self.n_features,), var_type=mip.BINARY, name="support")

            # Add feature selection constraint
            model.add_constr(mip.xsum(support) <= self.n_selected_features)
            model.add_constr(mip.xsum(support) >= 1)

            # Add constraints on highly correlated features
            for j in range(self.n_features):
                for k in range(j, self.n_features):
                    if corr[j, k] > self.max_correlation:
                        model.add_constr(support[j] + support[k] <= 1)

            # Solve problem using cutting planes algorithm
            result = cutting_planes_optimizer(
                func=func,
                model=model,
                x=support,
                x0=self._initialize_support(corr=corr),
                max_iter=self.max_iter,
                convergence_tol=self.convergence_tol,
                max_seconds_per_cut=self.max_seconds_per_cut
            )
            self._solve_results.append(result)

            # Update best result
            if result.objective_value >= best_objective_value:
                best_objective_value = result.objective_value
                self._selected_features = np.argwhere(np.round(result.x).astype(bool)).flatten()

        # Refit estimator with selected features
        self._estimator.fit_transform(X=X[:, self._selected_features])

    def _make_callback(self, X: np.ndarray) -> Callable[[np.ndarray], Tuple[float, np.ndarray]]:

        offset = X.shape[0] * X.var(axis=0)

        def func(support: np.ndarray) -> Tuple[float, np.ndarray]:

            # Solver inner problem
            clusters = self._estimator.fit_predict(X=X[:, np.round(support).astype(bool)])

            # Compute gradient and objective value
            grad = sum((clusters == k).sum() * X[clusters == k].var(axis=0) for k in range(self.n_clusters)) - offset
            obj_val = (grad * support).sum()

            return obj_val, grad

        return func

    def _initialize_support(self, corr: np.ndarray) -> np.ndarray:

        # Uniformly sample selected features
        selected_features = np.random.choice(self.n_features, self.n_selected_features, replace=False)

        # Remove infeasible feature combinations
        for k in range(1, self.n_selected_features):
            if corr[selected_features[0], k] > self.max_correlation:
                selected_features[k] = selected_features[0]

        # Return support vector
        return np.isin(np.arange(self._n_features), selected_features)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self._estimator.predict(X=self._scaler.transform(X=X)[:, self._selected_features])

    @property
    def selected_feature(self) -> np.ndarray:
        return self._selected_features

    @property
    def cluster_centers(self) -> np.ndarray:
        return self._estimator.cluster_centers_

    @property
    def solve_results(self) -> List[Result]:
        return self._solve_results



