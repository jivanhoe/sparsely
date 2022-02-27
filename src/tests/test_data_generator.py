import numpy as np
import pytest

from sparsely import ClassificationDataGenerator, RegressionDataGenerator
from sparsely.data_generator import BaseDataGenerator


def validate_data_generator(
        data_generator: BaseDataGenerator,
        return_weights: bool,
        shuffle_features: bool,
        random_state: bool
):
    if return_weights:
        X, y, weights = data_generator.sample(
            shuffle_features=shuffle_features,
            return_weights=return_weights,
            random_state=random_state
        )

        # Check weights
        assert weights.shape[0] == (data_generator.n_features,)
        assert weights.dtype == float
        assert not np.isnan(weights).any()
        if shuffle_features:
            assert (~np.isclose(weights, 0)).sum() == data_generator.n_informative
        else:
            assert not np.isclose(weights[:data_generator.n_informative], 0).any()
            assert np.isclose(weights[data_generator.n_informative:], 0).all()

    else:
        X, y = data_generator.sample(
            shuffle_features=shuffle_features,
            return_weights=return_weights,
            random_state=random_state
        )

    # Check dataset
    assert X.shape == (data_generator.n_samples, data_generator.n_features)
    assert y.shape == (data_generator.n_samples,)
    assert X.dtype == float
    assert y.dtype == float
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()


@pytest.mark.parametrize("return_weights", [True, False])
@pytest.mark.parametrize("shuffle_features", [True, False])
@pytest.mark.parametrize("random_state", range(5))
def test_regression_data_generator(
        regression_data_generator: RegressionDataGenerator,
        return_weights: bool,
        shuffle_features: bool,
        random_state: bool
):
    validate_data_generator(
        data_generator=regression_data_generator,
        return_weights=return_weights,
        shuffle_features=shuffle_features,
        random_state=random_state
    )


@pytest.mark.parametrize("return_weights", [True, False])
@pytest.mark.parametrize("shuffle_features", [True, False])
@pytest.mark.parametrize("random_state", range(5))
def test_classification_data_generator(
        classification_data_generator: ClassificationDataGenerator,
        return_weights: bool,
        shuffle_features: bool,
        random_state: bool
):
    validate_data_generator(
        data_generator=classification_data_generator,
        return_weights=return_weights,
        shuffle_features=shuffle_features,
        random_state=random_state
    )



