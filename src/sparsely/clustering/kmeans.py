from typing import Dict, Optional

import mip
import numpy as np
from sklearn.cluster import KMeans

from sparsely.base import BaseUnsupervisedModel


class ConstraintGenerator(mip.ConstrsGenerator):

    def __init__(
            self,
            X: np.ndarray,
            estimator: KMeans,
            weight: mip.LinExprTensor,
            bcss: mip.Var,
            n_features_to_select: int
    ):
        super().__init__()
        self.X = X
        self.estimator = estimator
        self.weight = weight
        self.bcss = bcss
        self.n_features_to_select = n_features_to_select
        self._iter: int = 0
        self._n_samples: int = X.shape[0]
        self._n_features: int = X.shape[1]
        self._var: np.ndarray = self.X.var(axis=0)

    def generate_constrs(self, model: mip.Model, depth: int = 0, npass: int = 0) -> None:

        # Get incumbent solution
        weight = (
            np.round([w.x for w in self.weight]) if self._iter else
            np.isin(
                np.arange(self._n_features),
                np.random.choice(self._n_features, self.n_features_to_select, replace=False)
            )
        )

        # Solve inner problem
        clusters = self.estimator.fit_predict(X=self.X[:, weight.astype(bool)])

        # Add new cut
        coefficients = (
            self._n_samples * self._var
            - sum((clusters == k).sum() * self.X[clusters == k].var(axis=0) for k in range(self.estimator.n_clusters))
        )
        model.add_constr(self.bcss <= (coefficients * weight).sum() + mip.xsum(coefficients * (self.weight - weight)))


class SparseKMeans(BaseUnsupervisedModel):

    def __init__(
            self,
            n_clusters: int,
            n_features_to_select: int,
            n_restarts: int = 10,
            random_state: Optional[int] = None
    ):
        self.n_clusters = n_clusters
        self.n_features_to_select = n_features_to_select
        self.n_restarts = n_restarts
        self.random_state = random_state
        self._estimator = KMeans(n_clusters=self.n_clusters)
        self._selected_features: Optional[np.ndarray] = None

    def _fit(self, X: np.ndarray) -> None:

        if self.random_state is not None:
            np.random.seed(self.random_state)

        best_objective_value = -np.inf

        for _ in range(self.n_restarts):

            # Initialize MIP model
            model = mip.Model(sense=mip.MAXIMIZE, solver_name=mip.CBC)

            # Define variables
            weight = model.add_var_tensor(shape=(X.shape[1],), var_type=mip.BINARY, name="weight")
            bcss = model.add_var(name="bcss")

            # Define objective
            model.objective = mip.maximize(bcss)

            # Add lazy constraint generator
            model.lazy_constrs_generator = ConstraintGenerator(
                X=X,
                estimator=self._estimator,
                weight=weight,
                bcss=bcss,
                n_features_to_select=self.n_features_to_select
            )

            # Add initial cut
            model.lazy_constrs_generator.generate_constrs(model=model)

            # Add feature selection constraint
            model.add_constr(mip.xsum(weight) == self.n_features_to_select)

            model.optimize()

            if model.objective_value >= best_objective_value:
                best_objective_value = model.objective_value
                self._selected_features = np.argwhere(np.round([w.x for w in weight]).astype(bool)).flatten()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self._estimator.predict(X=X[:, self._selected_features])

    def params(self) -> Dict[str, any]:
        return dict(
            selected_features=self._selected_features,
            cluster_centers=self._estimator.cluster_centers_
        )

