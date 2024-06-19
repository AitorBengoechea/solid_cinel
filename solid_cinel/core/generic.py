# -*- coding: utf-8 -*-
"""
Python file for generic function.

@author: AB272525
"""
import scipy as sp
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from typing import Iterable
from scipy.stats import qmc


def integrate(series: pd.Series, kind="trapezoidal") -> float:
    """
    Get normalization coefficient of a function.

    Parameters
    ----------
    series : "pd.Series"
        Function to test if it is normlize.
    kind : "str", optional
        integral type. The default is "trapezoidal".

    Returns
    -------
    "float"
        The integration value.

    Raises
    ------
    ValueError
        Kind is not available in scipy.

    Example
    -------
    >>> f = pd.Series([1, 2, 4], index=[1, 2, 4])
    >>> integrate(f)
    7.5
    """
    # Get the values and the index of the series
    y = series.values
    x = series.index.values

    # Integrate the function
    if kind == "trapezoidal":
        y_norm = sp.integrate.trapezoid(y, x=x)
    elif kind == "simpson":
        y_norm = sp.integrate.simpson(y, x=x)
    else:
        raise ValueError("kind is not properly introduced")

    return y_norm

@nb.jit(nopython=True, nogil=True, parallel=True, cache=True)
def trapz_parallel(data: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Trapezoidal integration of a 2D array in parallel for the same x grid.

    Parameters
    ----------
    data: np.ndarray, (N, M)
        2D array to integrate
    x: np.ndarray, (M,)
        x grid

    Returns
    -------
    np.ndarray, (N,)
    """
    result = np.zeros(data.shape[0])
    for i in prange(data.shape[0]):
        result[i] += np.trapz(data[i], x)
    return result


def reshape_differential(data: pd.Series, xnew: Iterable,
                         kind: str = "slinear",
                         bounds_error: bool = False) -> np.ndarray:
    """
    Linearly interpolate array over new energy grid structure.
    Extrapolated values are replaced by zeros.

    Parameters
    ----------
    data: pd.Series
        Original data
    xnew : 1d array-like object with at least two entries
        new energy grid
    kind: "str"
        Specifies the kind of interpolation as a string or as an integer
        specifying the order of the spline interpolator to use. The string has
        to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’,
        ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’,
        ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth,
        first, second or third order; ‘previous’ and ‘next’ simply return the
        previous or next value of the point; ‘nearest-up’ and ‘nearest’ differ
        when interpolating half-integers (e.g. 0.5, 1.5) in that ‘nearest-up’
        rounds up and ‘nearest’ rounds down. Default is ‘linear’
    bounds_error: "bool"
        if True, a ValueError is raised any time interpolation is attempted on a
        value outside of the range of x (where extrapolation is necessary). If
        False, out of bounds values are assigned fill_value. By default, an
        error is raised unless fill_value="extrapolate".

    Returns
    -------
    `numpy.ndarray` with length `len(xnew)`
        interpolated array

    Examples
    --------
    Vector interpolation:
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> data = pd.Series(y, index=x)
    >>> xnew = np.array([1.5, 2.5, 3.5, 4.5])
    >>> reshape_differential(data, xnew)
    array([1.5, 2.5, 3.5, 4.5])

    Matrix interpolation:
    >>> xnew = np.array([[1.5, 2.5, 3.5, 4.5], [1.75, 2.75, 3.75, 4.75]])
    >>> pd.Series(reshape_differential(data, xnew)[0], index=xnew[0])
    1.5    1.5
    2.5    2.5
    3.5    3.5
    4.5    4.5
    dtype: float64
    >>> pd.Series(reshape_differential(data, xnew)[1], index=xnew[1])
    1.75    1.75
    2.75    2.75
    3.75    3.75
    4.75    4.75
    dtype: float64
    """
    foo = sp.interpolate.interp1d(
                                  data.index.values,
                                  data.values,
                                  axis=0,
                                  copy=False,
                                  kind=kind,
                                  bounds_error=bounds_error,
                                  fill_value=0.,
                                  assume_sorted=True,
                                  )
    return foo(xnew)

@nb.jit(nopython=True, cache=True, parallel=True, nogil=True)
def interp_xnewParallel(xnew: np.ndarray, xnewShape: tuple,
                        x: np.ndarray, y: np.ndarray):
    """
    Interpolate multiple xnew using the (x, y) function.

    Parameters
    ----------
    xnew: 'np.ndarray', (M, Z)
        New grid values.
    xnewShape: 'tuple'
        Shape of the new grid.
    x: 'np.ndarray', (T,)
        Original grid values.
    y: 'np.ndarray', (T,)
        Original function values.

    Returns
    -------
    'np.ndarray', (M, Z)
        Interpolated values.
    """
    yinterp = np.zeros(xnewShape)
    for n in prange(xnewShape[0]):
        yinterp[n] += np.interp(xnew[n], x, y)
    return yinterp

@nb.jit(nopython=True, cache=True, parallel=True, nogil=True)
def interp_multyParallel(xnew: np.ndarray, x: np.ndarray, y: np.ndarray):
    """
    Interpolate to xnew using multiple function with the same x grid.

    Parameters
    ----------
    xnew: 'np.ndarray', (M,)
        New grid values.
    xnewShape: 'tuple'
        Shape of the new grid.
    x: 'np.ndarray', (T, )
        Original grid values.
    y: 'np.ndarray', (Z, T)
        Original function values.

    Returns
    -------
    'np.ndarray', (Z, M)
        Interpolated values.
    """
    Nrow, Ncolumn = y.shape[0], len(xnew)
    yinterp = np.zeros((Nrow, Ncolumn))
    for n in prange(Nrow):
        yinterp[n] += np.interp(xnew, x, y[n])
    return yinterp
def interpolation(data: pd.Series, xnew: Iterable,
                  values: bool = False, parallel=False,
                  **kwargs) -> [np.ndarray, pd.Series]:
    """
    Interpolate the data over new energy grid structure.

    Parameters
    ----------
    data: pd.Series
        Original data
    xnew: 1d array-like object with at least two entries
        new energy grid
    values: bool
        If True, the function returns a numpy array instead a pd.Series
    kind: "str"
        Specifies the kind of interpolation as a string or as an integer
        specifying the order of the spline interpolator to use. The string has
        to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’,
        ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’,
        ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth,
        first, second or third order; ‘previous’ and ‘next’ simply return the
        previous or next value of the point; ‘nearest-up’ and ‘nearest’ differ
        when interpolating half-integers (e.g. 0.5, 1.5) in that ‘nearest-up’
        rounds up and ‘nearest’ rounds down. Default is ‘linear’
    bounds_error: "bool"
        if True, a ValueError is raised any time interpolation is attempted on a
        value outside of the range of x (where extrapolation is necessary). If
        False, out of bounds values are assigned fill_value. By default, an
        error is raised unless fill_value="extrapolate".

    Returns
    -------
    pd.Series
        interpolated array

    Examples
    --------
    Vector interpolation:
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> data = pd.Series(y, index=x)
    >>> xnew = np.array([1.5, 2.5, 3.5, 4.5])
    >>> interpolation(data, xnew)
    1.5    1.5
    2.5    2.5
    3.5    3.5
    4.5    4.5
    dtype: float64
    """
    # Interpolate the data
    xnewShape = xnew.shape
    if parallel or xnewShape[0] > 100:
        yinterp = interp_xnewParallel(xnew, xnewShape, data.index.values, data.values)
    else:
        yinterp = reshape_differential(data, xnew, **kwargs)

    # Return the interpolated values as a numpy array or a pd.Series
    if values or len(xnewShape) == 2:
        return yinterp
    else:
        return pd.Series(yinterp, index=xnew)

def reshift(data: pd.Series, dx: [float, np.ndarray]) -> pd.Series:
    """
    Reshift the data to the original grid. For example, if the data is shifted
    to the left, the data interpolated to the original grid.

    Parameters
    ----------
    data : pd.Series
        Data to reshift
    dx : np.ndarray, float
        Shifted grid

    Returns
    -------
    pd.Series
        Reshifted data

    Examples
    --------
    Vector interpolation:
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> data = pd.Series(y, index=x)
    >>> dx = - 0.5
    >>> reshift(data, dx)
    1    1.5
    2    2.5
    3    3.5
    4    4.5
    5    0.0
     dtype: float64

    >>> dx = + 0.5
    >>> reshift(data, dx)
    1    0.0
    2    1.5
    3    2.5
    4    3.5
    5    4.5
    dtype: float64

    array interpolation:
    >>> dx = np.array([-0.25, 0.0, 0.25, 0.5, 0.75])
    >>> reshift(data, dx)
    1    1.2
    2    2.0
    3    2.8
    4    3.6
    5    4.4
    dtype: float64
    """
    x = data.index.values
    reshifted_data = reshape_differential(data.set_axis(x + dx), x)
    return pd.Series(reshifted_data, index=data.index.values)


def sampling(d: int, n: int) -> np.array:
    """
    Generate a latin hypercube sampling between 0 and 1.

    Parameters
    ----------
    d : "int"
        Dimension of the sampling
    n : "int"
        Number of samples

    Returns
    -------
    "np.array"
        Array of random numbers between 0 and 1 based on LHS
    """
    samples = qmc.LatinHypercube(d=d).random(n=n)
    return samples if d > 1 else samples[:, 0]
