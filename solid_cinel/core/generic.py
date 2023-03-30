# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:11:55 2022

@author: AB272525
"""
import scipy as sp
import pandas as pd
from collections.abc import Iterable


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


def reshape_differential(x: Iterable[:], y: Iterable[:], xnew: Iterable[:],
                         kind: str="slinear", bounds_error: bool=False):
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
    """
    foo = sp.interpolate.interp1d(
                                  x,
                                  y,
                                  axis=0,
                                  copy=False,
                                  kind=kind,
                                  bounds_error=bounds_error,
                                  fill_value=0.,
                                  assume_sorted=True,
                                  )
    return foo(xnew)
