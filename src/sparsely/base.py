from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, r2_score


class BaseUnsupervisedModel(ABC):

    def __init__(self):
        self._n_features: Optional[int] = None

    def fit(self, X: np.ndarray) -> None:
        self._validate_model()
        self._validate_data(X=X)
        self._fit(X=X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._validate_data(X=X)
        return self._predict(X=X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self._fit(X=X)
        return self._predict(X=X)

    def _validate_data(self, X: np.ndarray) -> None:
        assert X.ndim == 2
        if self.n_features is None:
            self._n_features = X.shape[1]
        else:
            assert X.shape[1] == self.n_features

    @property
    def n_features(self) -> int:
        return self._n_features

    @abstractmethod
    def _validate_model(self) -> None:
        ...

    @abstractmethod
    def _fit(self, X: np.ndarray) -> None:
        ...

    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def params(self) -> Dict[str, Any]:
        ...


class BaseSupervisedModel(ABC):

    def __init__(self):
        self._n_features: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._validate_model()
        self._validate_data(X=X, y=y, check_y=True)
        self._fit(X=X, y=y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._validate_data(X=X, y=None, check_y=False)
        return self._predict(X=X)

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit_predict(X=X, y=y)
        return self.predict(X=X)

    def _validate_data(self, X: np.ndarray, y: Optional[np.ndarray], check_y: bool) -> None:
        assert X.ndim == 2
        if self.n_features is None:
            self._n_features = X.shape[1]
        else:
            assert X.shape[1] == self.n_features
        if check_y:
            assert y is not None
            assert y.ndim == 1
            assert y.shape[0] == X.shape[0]

    @property
    def n_features(self) -> int:
        return self._n_features

    @abstractmethod
    def _validate_model(self) -> None:
        ...

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        ...

    @property
    @abstractmethod
    def params(self) -> Dict[str, Any]:
        ...


class BaseRegressor(ABC, BaseSupervisedModel):

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return r2_score(y_true=y, y_pred=self.predict(X=X))


class BaseClassifier(ABC, BaseSupervisedModel):

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return roc_auc_score(y_true=y, y_score=self.predict(X=X))
