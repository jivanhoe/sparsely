import numpy as np
import mip

from sparsely.base import BaseRegressor

from typing import Any, Dict, Optional


class ConstraintGenerator(mip.ConstrsGenerator):

    def __init__(self, X: np.ndarray, y: np.ndarray, l2_penalty: float):
        self.X = X
        self.y = y
        self.l2_penalty = l2_penalty

    def generate_constrs(self, model: mip.Model, depth: int = 0, npass: int = 0):
        pass


class SparseLinearRegressor(BaseRegressor):

    def __init__(self, n_features_to_select: int, l2_penalty: float = 0.1):
        self.n_features_to_select = n_features_to_select
        self.l2_penalty = l2_penalty
        self._coef: Optional[np.ndarray] = None

    def _validate_model(self) -> None:
        assert self.n_features_to_select > 0
        assert self.l2_penalty > 0

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)
        
        coef = model.add_var_tensor(shape=(self.n_features,), name="coef")
        z = model.add_var_tensor(shape=(self.n_features,), name="z")
        mse = model.add_var(name="mse")

        model.lazy_constrs_generator = ConstraintGenerator(X=X, y=y, l2_penalty=self.l2_penalty)

        model.add_constr(mip.xsum(z) <= self.n_features_to_select)


    def _predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @property
    def params(self) -> Dict[str, Any]:
        return dict(coef=self._coef)