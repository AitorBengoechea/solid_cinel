# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:11:55 2022

@author: AB272525
"""
import scipy as sp
import pandas as pd
import numpy as np


def integrate(series, kind="trapezoidal") -> float:
    """
    Get normalization coefficient of a function.

    Parameters
    ----------
    series : TYPE
        Function to test if it is normlize.
    kind : "str", optional
        integral type. The default is "trapezoidal".

    Raises
    ------
    ValueError
        Kind is not available in github.

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


def reshape_differential(x, y, xnew):
    """
    Linearly interpolate array over new energy grid structure.
    Extrapolated values are replaced by zeros.

    Parameters
    ----------
    x : 1d array-like object with at least two entries
        energy grid
    xnew : 1d array-like object with at least two entries
        new energy grid
    y : `numpy.ndarray` with at least two entries and same length as `x`
        array to interpolate
    Returns
    -------
    `numpy.ndarray` with length `len(xnew)`
        interpolated array
    """
    foo = sp.interpolate.interp1d(
                                  x,
                                  y,
                                  axis=0,
                                  copy=False,
                                  kind="slinear",
                                  bounds_error=False,
                                  fill_value=0.,
                                  assume_sorted=True,
                                  )
    return foo(xnew)
