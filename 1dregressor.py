"""Implementation of 1-D regressor."""


class SimpleRegressor:
    """The base of a 1-D polynomial regressor."""

    def __init__(self, max_order):
        """Initialize the regressor with max_order polynomial."""
        self._order = max_order


class LSERegressor(SimpleRegressor):
    """1-D polynomial regressor using LSE."""

    def __init__(self, max_order, lambda_):
        """Initialize LSE with lambda for regularization."""
        super().__init__(max_order)
        self._lambda = lambda_


class NewtonsRegressor(SimpleRegressor):
    """1-D polynomial regressor using Newton's method."""
