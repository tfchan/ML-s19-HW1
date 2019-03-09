"""Module that contains operations related to matrix."""
import numpy as np


def _pivoting(matrix_a):
    """Pivoting matrix_a."""
    matrix_size = matrix_a.shape[0]
    id_matrix = np.identity(matrix_size)
    matrix_pivoted = np.copy(matrix_a)
    for j in range(matrix_size):
        # Find row with absolutely largest element at column j
        row = max(range(j, matrix_size), key=lambda i: abs(matrix_a[i][j]))
        if j != row:
            id_matrix[[j, row]] = id_matrix[[row, j]]
            matrix_pivoted[[j, row]] = matrix_pivoted[[row, j]]
    return id_matrix, matrix_pivoted


def lu_decomposition(matrix_a):
    """Do LU decomposition on the square matrix matrix_a."""
    assert matrix_a.ndim == 2, 'Input matrix must be a 2d-array'
    assert matrix_a.shape[0] == matrix_a.shape[1], 'Must input square matrix'

    matrix_p, matrix_pa = _pivoting(matrix_a)
