"""Module that contains operations related to matrix."""
import numpy as np


def _pivoting(m):
    """Pivoting matrix m."""
    m_size = m.shape[0]
    identity = np.identity(m_size)
    m_copy = np.copy(m)
    for j in range(m_size):
        # Find row with absolutely largest element at column j
        row = max(range(j, m_size), key=lambda i: abs(m_copy[i][j]))
        if j != row:
            identity[[j, row]] = identity[[row, j]]
            m_copy[[j, row]] = m_copy[[row, j]]
    return identity, m_copy


def lu_decomposition(a):
    """Do LU decomposition on the square matrix matrix a."""
    assert a.ndim == 2, 'Input matrix must be a 2d-array'
    assert a.shape[0] == a.shape[1], 'Must input square matrix'

    p, pa = _pivoting(a)

    a_size = a.shape[0]
    l_ = np.zeros((a_size, a_size))
    u = np.zeros((a_size, a_size))
    for j in range(a_size):
        l_[j][j] = 1
        for i in range(j + 1):
            u[i][j] = pa[i][j] - sum(l_[i][k] * u[k][j] for k in range(i))
        for i in range(j + 1, a_size):
            l_[i][j] = pa[i][j] - sum(l_[i][k] * u[k][j] for k in range(j))
            l_[i][j] /= u[j][j]
    return p, l_, u


def inverse_of_l(l_):
    """Find inverse of lower triangular matrix l_ with diagonal of 1."""
    l_size = l_.shape[0]
    l_inv = np.zeros((l_size, l_size))
    for j in range(l_size):
        l_inv[j][j] = 1
        for i in range(j + 1, l_size):
            l_inv[i][j] = -sum(l_[i][k] * l_inv[k][j] for k in range(i))
    return l_inv


def inverse_of_u(u):
    """Find inverse of upper triangular matrix u with diagonal of 0."""
    u_size = u.shape[0]
    u_inv = np.zeros((u_size, u_size))
    for i_ in range(u_size):
        i = u_size - i_ - 1
        for j in range(i, u_size):
            u_inv[i][j] = -sum(
                u[i][k] * u_inv[k][j] for k in range(i + 1, j + 1))
            u_inv[i][j] = 1 / u[i][i] if i == j else u_inv[i][j] / u[i][i]
    return u_inv


def inverse(a):
    """Find inverse of square matrix a."""
    assert a.ndim == 2, 'Input matrix must be a 2d-array'
    assert a.shape[0] == a.shape[1], 'Must input square matrix'

    p, l_, u = lu_decomposition(a)
    l_inv = inverse_of_l(l_)
    u_inv = inverse_of_u(u)
    a_inv = np.dot(u_inv, l_inv)
    return a_inv
