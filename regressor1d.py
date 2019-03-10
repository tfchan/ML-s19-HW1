"""Implementation of 1-D regressor."""
import numpy as np


class SimpleRegressor:
    """The base of a 1-D polynomial regressor."""

    def __init__(self, max_order):
        """Initialize the regressor with max_order polynomial."""
        self._order = max_order
        self._coefficient = [0] * self._order

    def predict(self, input_data):
        """Predict output using learnt parameters."""
        a = [[d ** order for order in range(self._order)] for d in input_data]
        output = np.dot(a, self._coefficient)
        return list(output)


class LSERegressor(SimpleRegressor):
    """1-D polynomial regressor using LSE."""

    def __init__(self, max_order, lambda_):
        """Initialize LSE with lambda for regularization."""
        super().__init__(max_order)
        self._lambda = lambda_

    def fit(self, input_data, output_data):
        """Fit input data with output data."""
        a = [[d ** order for order in range(self._order)] for d in input_data]
        ata = np.dot(np.transpose(a), a)
        ata_lambda = ata + self._lambda * np.identity(ata.shape[0])
        ata_lambda_inv = np.linalg.inv(ata_lambda)
        atb = np.dot(np.transpose(a), output_data)
        w = np.dot(ata_lambda_inv, atb)
        self._coefficient = list(w)


class NewtonsRegressor(SimpleRegressor):
    """1-D polynomial regressor using Newton's method."""
