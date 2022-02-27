from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np


class BaseDataGenerator(ABC):

    def __init__(
            self,
            n_samples: int,
            n_features: int,
            n_informative: int,
            n_redundant: int,
            noise: float,
            min_weight_magnitude: float
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.noise = noise
        self.min_weight_magnitude = min_weight_magnitude

    def sample(
            self,
            shuffle_features: bool = False,
            return_weights: bool = False,
            random_state: Optional[int] = None,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        rng = np.random.RandomState(random_state)
        X, mask = self._generate_features(shuffle_features=shuffle_features, rng=rng)
        weights = self._generate_weights(mask=mask, rng=rng)
        y = self._generate_targets(X=X, weights=weights, rng=rng)
        if return_weights:
            return X, y, weights
        return X, y

    def _generate_features(
            self,
            shuffle_features: bool,
            rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = rng.randn(self.n_samples, self.n_features - self.n_redundant)
        if self.n_redundant > 0:
            X = np.column_stack([
                X,
                np.matmul(X[:, :self.n_informative], rng.randn(self.n_informative, self.n_redundant))
                + rng.randn(self.n_samples, self.n_redundant)
            ])
        feature_order = np.arange(self.n_features)
        if shuffle_features:
            rng.shuffle(feature_order)
        return X, feature_order < self.n_informative

    def _generate_weights(
            self,
            mask: np.ndarray,
            rng: np.random.RandomState
    ) -> np.ndarray:
        weights = rng.randn(self.n_features)
        weights = np.where(
            np.abs(weights) >= self.min_weight_magnitude,
            weights,
            np.sign(weights) * self.min_weight_magnitude
        )
        return weights * mask

    @abstractmethod
    def _generate_targets(
            self,
            X: np.ndarray,
            weights: np.ndarray,
            rng: np.random.RandomState
    ) -> np.ndarray:
        ...


class RegressionDataGenerator(BaseDataGenerator):

    def __init__(
            self,
            n_samples: int = 500,
            n_features: int = 10,
            n_informative: int = 3,
            n_redundant: int = 2,
            noise: float = 0.1,
            min_weight_magnitude=0.1
    ):
        super().__init__(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            noise=noise,
            min_weight_magnitude=min_weight_magnitude
        )

    def _generate_targets(
            self,
            X: np.ndarray,
            weights: np.ndarray,
            rng: np.random.RandomState
    ) -> np.ndarray:
        return np.matmul(X, weights) + self.noise * rng.randn(self.n_samples)


class ClassificationDataGenerator(BaseDataGenerator):

    def __init__(
            self,
            n_samples: int = 500,
            n_features: int = 10,
            n_informative: int = 3,
            n_redundant: int = 2,
            noise: float = 0.,
            min_weight_magnitude: float = 0.1,
            flip_proba: float = 0.1
    ):
        super().__init__(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            noise=noise,
            min_weight_magnitude=min_weight_magnitude
        )
        self.flip_proba = flip_proba

    def _generate_targets(
            self,
            X: np.ndarray,
            weights: np.ndarray,
            rng: np.random.RandomState
    ) -> np.ndarray:
        y = 1 / (1 + np.exp(-np.matmul(X, weights) + self.noise * rng.randn(self.n_samples))) > .5
        flip_mask = rng.rand(self.n_samples) > self.flip_proba
        y[flip_mask] = ~y[flip_mask]
        return 2 * y.astype(int) - 1
