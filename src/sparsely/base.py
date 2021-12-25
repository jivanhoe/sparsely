from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score, r2_score
from dataclasses import dataclass


@dataclass
class ModelParameters(ABC):

    pass


@dataclass
class BaseUnsupervisedModel(ABC):

    def __post_init__(self):
        self._n_features: Optional[int] = None

    def fit(self, X: np.ndarray) -> None:
        self._validate_data(X=X)
        self._fit(X=X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._validate_data(X=X)
        return self._predict(X=X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X=X)
        return self._predict(X=X)

    def _validate_data(self, X: np.ndarray) -> None:
        assert X.ndim == 2, (
            f"The feature matrix X must be 2-dimensional. Provided data has {X.ndim} dimensions."
        )
        if self.n_features is None:
            self._n_features = X.shape[1]
        else:
            assert X.shape[1] == self.n_features, (
                f"The expected number of features is {self.n_features}. Provided data has {X.shape[1]} features."
            )

    @property
    def n_features(self) -> int:
        return self._n_features

    @abstractmethod
    def _fit(self, X: np.ndarray) -> None:
        ...

    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def params(self) -> ModelParameters:
        ...


@dataclass
class BaseSupervisedModel(ABC):

    def __post_init__(self):
        self._n_features: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._validate_data(X=X, y=y, check_y=True)
        self._fit(X=X, y=y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._validate_data(X=X, y=None, check_y=False)
        return self._predict(X=X)

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit_predict(X=X, y=y)
        return self.predict(X=X)

    def _validate_data(self, X: np.ndarray, y: Optional[np.ndarray], check_y: bool) -> None:
        assert X.ndim == 2, (
            f"The feature matrix 'X' must be 2-dimensional. Provided data has {X.ndim} dimensions."
        )
        if self.n_features is None:
            self._n_features = X.shape[1]
        else:
            assert X.shape[1] == self.n_features, (
                f"The expected number of features is {self.n_features}. Provided data has {X.shape[1]} features."
            )
        if check_y:
            assert y is not None
            assert y.ndim == 1, (
                f"The target vector 'y' must be 2-dimensional. Provided data has {y.ndim} dimensions."
            )
            assert y.shape[0] == X.shape[0], (
                f"The number of samples in the feature matrix 'X' ({X.shape[0]}) does not match the number of samples "
                f"in the target vector 'y' ({y.shape[0]})."
            )

    @property
    def n_features(self) -> int:
        return self._n_features

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
    def params(self) -> ModelParameters:
        ...


@dataclass
class BaseRegressor(BaseSupervisedModel):

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return r2_score(y_true=y, y_pred=self.predict(X=X))


@dataclass
class BaseClassifier(BaseSupervisedModel):

    def __init__(self):
        super().__init__()
        self._classes: Optional[np.ndarray] = None

    @property
    def classes(self) -> np.ndarray:
        return self._classes

    @property
    def n_classes(self) -> int:
        return len(self._classes)

    def _validate_data(self, X: np.ndarray, y: Optional[np.ndarray], check_y: bool) -> None:
        super()._validate_data(X=X, y=y, check_y=check_y)
        if check_y:
            assert y.dtype in (int, bool), (
                f"The target variable for a classifier must be of type int or bool. Provided data is of type {y.dtype}."
            )
            if self._classes is None:
                self._classes = np.unique(y)
                assert self.n_classes > 1, (
                    f"The number of classes must be greater than 1."
                )
            else:
                assert np.isin(y, self.classes), (
                    f"Unexpected target classes."
                )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return roc_auc_score(y_true=y, y_score=self.predict(X=X), multi_class="ovr" if self.n_classes > 2 else "raise")
