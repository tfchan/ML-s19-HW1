"""Implementation of 1-D regressor."""
import numpy as np


class SimpleRegressor:
    """The base of a 1-D polynomial regressor."""

    def __init__(self, max_order):
        """Initialize the regressor with max_order polynomial."""
        self._order = max_order
        self._coefficient = [0] * self._order

    def predict(self, input_data, real_output=None):
        """Predict output using learnt parameters."""
        a = [[d ** order for order in range(self._order)] for d in input_data]
        output = np.dot(a, self._coefficient)
        if real_output is None:
            return list(output)
        else:
            se = sum((output - real_output) ** 2)
            return list(output), se

    def get_equation(self):
        """Return string of the polynomail equation."""
        equation = ''
        for i, coe in reversed(list(enumerate(self._coefficient))):
            if i >= len(self._coefficient) - 1:
                equation += f'{coe}'
            elif coe >= 0:
                equation += f' + {coe}'
            else:
                equation += f' - {-coe}'
            equation += f'X^{i}' if i != 0 else ''
        return equation


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
