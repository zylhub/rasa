import numpy as np
import pytest
import scipy.sparse

from rasa.nlu.featurizers.featurizer import (
    SparseFeaturizer,
    DenseFeaturizer,
    sequence_to_sentence_features,
)
from rasa.nlu.constants import DENSE_FEATURE_NAMES, SPARSE_FEATURE_NAMES, TEXT
from rasa.nlu.training_data import Message


def test_combine_with_existing_dense_features():

    featurizer = DenseFeaturizer()
    attribute = DENSE_FEATURE_NAMES[TEXT]

    existing_features = [[1, 0, 2, 3], [2, 0, 0, 1]]
    new_features = [[1, 0], [0, 1]]
    expected_features = [[1, 0, 2, 3, 1, 0], [2, 0, 0, 1, 0, 1]]

    message = Message("This is a text.")
    message.set(attribute, existing_features)

    actual_features = featurizer._combine_with_existing_dense_features(
        message, new_features, attribute
    )

    assert np.all(expected_features == actual_features)


def test_combine_with_existing_dense_features_shape_mismatch():
    featurizer = DenseFeaturizer()
    attribute = DENSE_FEATURE_NAMES[TEXT]

    existing_features = [[1, 0, 2, 3], [2, 0, 0, 1]]
    new_features = [[0, 1]]

    message = Message("This is a text.")
    message.set(attribute, existing_features)

    with pytest.raises(ValueError):
        featurizer._combine_with_existing_dense_features(
            message, new_features, attribute
        )


def test_combine_with_existing_sparse_features():
    featurizer = SparseFeaturizer()
    attribute = SPARSE_FEATURE_NAMES[TEXT]

    existing_features = scipy.sparse.csr_matrix([[1, 0, 2, 3], [2, 0, 0, 1]])
    new_features = scipy.sparse.csr_matrix([[1, 0], [0, 1]])
    expected_features = [[1, 0, 2, 3, 1, 0], [2, 0, 0, 1, 0, 1]]

    message = Message("This is a text.")
    message.set(attribute, existing_features)

    actual_features = featurizer._combine_with_existing_sparse_features(
        message, new_features, attribute
    )
    actual_features = actual_features.toarray()

    assert np.all(expected_features == actual_features)


def test_combine_with_existing_sparse_features_shape_mismatch():
    featurizer = SparseFeaturizer()
    attribute = SPARSE_FEATURE_NAMES[TEXT]

    existing_features = scipy.sparse.csr_matrix([[1, 0, 2, 3], [2, 0, 0, 1]])
    new_features = scipy.sparse.csr_matrix([[0, 1]])

    message = Message("This is a text.")
    message.set(attribute, existing_features)

    with pytest.raises(ValueError):
        featurizer._combine_with_existing_sparse_features(
            message, new_features, attribute
        )


@pytest.mark.parametrize(
    "features, expected",
    [
        (None, None),
        ([[1, 0, 2, 3], [2, 0, 0, 1]], [[2, 0, 0, 1]]),
        (
            scipy.sparse.coo_matrix([[1, 0, 2, 3], [2, 0, 0, 1]]),
            scipy.sparse.coo_matrix([2, 0, 0, 1]),
        ),
        (
            scipy.sparse.csr_matrix([[1, 0, 2, 3], [2, 0, 0, 1]]),
            scipy.sparse.csr_matrix([2, 0, 0, 1]),
        ),
    ],
)
def test_sequence_to_sentence_features(features, expected):
    actual = sequence_to_sentence_features(features)

    if isinstance(expected, scipy.sparse.spmatrix):
        assert np.all(expected.toarray() == actual.toarray())
    else:
        assert np.all(expected == actual)


@pytest.mark.parametrize(
    "pooling, features, expected",
    [
        (
            "mean",
            np.array([[0.5, 3, 0.4, 0.1], [0, 0, 0, 0], [0.5, 3, 0.4, 0.1]]),
            np.array([[0.5, 3, 0.4, 0.1]]),
        ),
        (
            "max",
            np.array([[1.0, 3.0, 0.0, 2.0], [4.0, 3.0, 1.0, 0.0]]),
            np.array([[4.0, 3.0, 1.0, 2.0]]),
        ),
        (
            "max",
            np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0, 0.0]]),
        ),
    ],
)
def test_calculate_cls_vector(pooling, features, expected):
    actual = DenseFeaturizer._calculate_cls_vector(features, pooling)

    assert np.all(actual == expected)
