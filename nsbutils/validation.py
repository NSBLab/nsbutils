"""Validation utilities for matrices and vectors."""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

def check_orthogonal_matrix(
    matrix: ArrayLike,
    atol: float = 1e-6
) -> bool:
    """
    Check if a matrix is orthonormal, i.e., its rows and columns are both orthogonal and normalized.

    Parameters
    ----------
    matrix : array_like
        The matrix to be checked for orthonormality.
    tol : float, optional
        The tolerance value for checking orthogonality and normalization. Default is 1e-6.

    Returns
    -------
    bool
        True if the matrix is orthonormal, False otherwise.

    Raises
    ------
    TypeError
        If the input cannot be converted to a numpy array.
    AssertionError
        If the input does not meet the dimensionality or value requirements.
    """

    matrix = np.asarray(matrix)

    return (
        matrix.shape[0] == matrix.shape[1] and                      # short-circuit if not square
        check_orthonormal_vectors(matrix, axis=0, atol=atol) and
        check_orthonormal_vectors(matrix, axis=1, atol=atol)
        )

def check_orthonormal_vectors(
    matrix: ArrayLike,
    axis: int = 0,
    atol: float = 1e-6
) -> bool:
    """
    Check if a set of real-valued vectors (emodes) in a matrix (rows or columns) are orthonormal.
    
    Parameters
    ----------
    matrix : array_like
        The set of vectors (emodes) to be checked for orthonormality.
    axis : int, optional
        If 0, vectors (emodes) are the matrix's columns. If 1, they are the matrix's rows. Default
        is 0.
    atol : float, optional
        The tolerance value for checking orthonormality. Default is 1e-6.
    
    Returns
    -------
    bool
        True if the vectors are orthonormal, False otherwise.

    Raises
    ------
    ValueError
        If the matrix is not 2 dimensional, or if the axis parameter is not 0 or 1.
    """

    matrix = np.asarray(matrix)
    if matrix.ndim != 2: 
        raise ValueError("Input array must be 2-dimensional.")

    if axis==0: 
        gram = matrix.T @ matrix
    elif axis==1:
        gram = matrix @ matrix.T
    else:
        raise ValueError("Axis must be 0 (columns) or 1 (rows).")

    return np.allclose(gram, np.eye(gram.shape[0]), atol=atol)

def check_orthogonal_vectors(
    matrix: ArrayLike,
    axis: int = 0,
    atol: float = 1e-6
) -> bool:
    """
    Check if a set of real-valued vectors (emodes) in a matrix (rows or columns) are orthogonal.

    Parameters
    ----------
    matrix : array_like
        The set of vectors (emodes) to be checked for orthogonality.
    colvec : bool, optional
        If True, vectors (emodes) are the matrix's columns. If False, they are the matrix's rows.
        Default is True.
    tol : float, optional
        The tolerance value for checking orthogonality. Default is 1e-6.

    Returns
    -------
    bool
        True if the vectors are orthogonal, False otherwise.

    Raises
    ------
    TypeError
        If the input cannot be converted to a numpy array.
    AssertionError
        If the input does not meet the dimensionality or value requirements.
    """

    matrix = np.asarray(matrix)
    assert matrix.ndim == 2, "Input array must be 2-dimensional."

    if axis==0: 
        gram = matrix.T @ matrix
    elif axis==1:
        gram = matrix @ matrix.T
    else:
        raise ValueError("Axis must be 0 (columns) or 1 (rows).")

    np.fill_diagonal(gram, 0.0)
    return np.allclose(gram, 0.0, atol=atol)

def check_normalized_vectors(
    matrix: ArrayLike,
    axis: int = 0,
    atol: float = 1e-6
) -> bool:
    """
    Check if a set of real-valued vectors (emodes) in a matrix (rows or columns) have unit
    magnitude.

    Parameters
    ----------
    matrix : array_like
        The input matrix.
    colvec : bool, optional
        If True, vectors (emodes) are the matrix's columns. If False, they are the matrix's rows.
        Default is True.
    tol : float, optional
        The tolerance for comparing the magnitudes to 1. By default, tol=1e-6.

    Returns
    -------
    bool
        True if all vector magnitudes are close to 1 within the given tolerance, False otherwise.

    Raises
    ------
    TypeError
        If the input cannot be converted to a numpy array.
    AssertionError
        If the input does not meet the dimensionality or value requirements.
    """

    return np.allclose(np.linalg.norm(np.asarray(matrix), axis=axis), 1.0, atol=atol)
