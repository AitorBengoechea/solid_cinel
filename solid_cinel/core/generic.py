# -*- coding: utf-8 -*-
"""
Python file for generic function.

@author: AB272525
"""
import scipy as sp
import numpy as np
import pandas as pd
import re
import os
import tempfile
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
    y = series.values
    x = series.index.values
    if kind == "trapezoidal":
        y_norm = sp.integrate.trapezoid(y, x=x)
    elif kind == "simpson":
        y_norm = sp.integrate.simpson(y, x=x)
    else:
        raise ValueError("kind is not properly introduced")
    return y_norm


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
    >>> reshape_differential(data, xnew)[0]
    array([1.5, 2.5, 3.5, 4.5])
    >>> reshape_differential(data, xnew)[1]
    array([1.75, 2.75, 3.75, 4.75])
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


def interpolation(data: pd.Series, xnew: Iterable, values=False) -> [np.ndarray, pd.Series]:
    """
    Interpolate the data over new energy grid structure.

    Parameters
    ----------
    data: pd.Series
        Original data
    xnew: 1d array-like object with at least two entries
        new energy grid

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
    data_interp = reshape_differential(data, xnew)
    if values:
        return data_interp
    else:
        return pd.Series(data_interp, index=xnew)

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


def read_file(file_path, header=None, index_col=None, usecols=None,
              engine="python"):
    """
    Read a file, extract numbers from each line, and return a pandas DataFrame.

    Parameters:
    file_path: str
        The path to the file.
    header:  int, list of int
        Row(s) to use as the column names, and the start of the data.
    index_col: int, str, sequence[int/str], or False:
        Column(s) to set as index(MultiIndex). The default is None.
    usecols: list-like or callable
        Return a subset of the columns. The default is None.
    engine: str
        Parser engine to use. The default is "python".

    Returns:
    df : pd.DataFrame
        A pandas DataFrame containing the processed data.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        processed_lines = [' '.join(re.findall(r'\d+\.?\d*', line)) + '\n' for line in lines]

        with tempfile.NamedTemporaryFile('w', delete=False) as temp_file:
            temp_file.writelines(processed_lines)
            temp_file.seek(0)  # move the cursor to the beginning of the file
            df = pd.read_csv(temp_file.name, sep=" ", index_col=index_col,
                             usecols=usecols, engine=engine, header=header)

        os.remove(temp_file.name)  # delete the temporary file
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
