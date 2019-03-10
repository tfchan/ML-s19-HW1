"""Implementation of 1-D regressor."""
import numpy as np


class SimpleRegressor:
    """The base of a 1-D polynomial regressor."""

    def __init__(self, max_order):
        """Initialize the regressor with max_order polynomial."""
        self._order = max_order
        self._coefficient = np.zeros(self._order)

    def _calculate_a(self, input_data):
        """Calculate a by substituting input into each order of x."""
        a = np.asarray(
            [[d ** order for order in range(self._order)] for d in input_data])
        return a

    def predict(self, input_data, real_output=None):
        """Predict output using learnt parameters."""
        a = self._calculate_a(input_data)
        output = a @ self._coefficient
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
        a = self._calculate_a(input_data)
        ata = np.transpose(a) @ a
        ata_lambda = ata + self._lambda * np.identity(ata.shape[0])
        ata_lambda_inv = np.linalg.inv(ata_lambda)
        atb = np.transpose(a) @ output_data
        w = ata_lambda_inv @ atb
        self._coefficient = w


class NewtonsRegressor(SimpleRegressor):
    """1-D polynomial regressor using Newton's method."""

    def fit(self, input_data, output_data):
        """Fit input data with output data."""
        w0 = np.random.rand(self._order)
        while True:
            a = self._calculate_a(input_data)
            b = output_data
            at = np.transpose(a)
            f_1 = 2 * at @ a @ w0 - 2 * at @ b
            f_2 = 2 * at @ a
            w1 = w0 - np.linalg.inv(f_2) @ f_1
            if np.allclose(w1, w0):
                break
            w0 = w1
        self._coefficient = w1
