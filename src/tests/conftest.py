from typing import Tuple

import numpy as np
import pytest

from sparsely import ClassificationDataGenerator, RegressionDataGenerator


@pytest.fixture(
    params=[
        dict(),
        dict(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=2,
            noise=0.01
        )
    ]
)
def regression_data_generator(request) -> RegressionDataGenerator:
    return RegressionDataGenerator(**request.param)


@pytest.fixture(
    params=[
        dict(),
        dict(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=2,
            noise=0.01
        )
    ]
)
def classification_data_generator(request) -> ClassificationDataGenerator:
    return ClassificationDataGenerator(**request.param)


@pytest.fixture(params=range(3))
def regression_dataset(
        request,
        regression_data_generator: RegressionDataGenerator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return regression_data_generator.sample(
        random_state=request.param,
        shuffle_features=True,
        return_weights=True
    )


@pytest.fixture(params=range(3))
def classification_dataset(
        request,
        classification_data_generator: ClassificationDataGenerator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return regression_data_generator.sample(
        random_state=request.param,
        shuffle_features=True,
        return_weights=True
    )
