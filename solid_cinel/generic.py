# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:11:55 2022

@author: AB272525
"""
import scipy as sp
import pandas as pd
import numpy as np


def normalization_coeff(series, kind="trapezoidal") -> float:
    """
    Get normalization coefficient of a function.

    Parameters
    ----------
    series : TYPE
        DESCRIPTION.
    kind : "str", optional
        integral type. The default is "trapezoidal".

    Raises
    ------
    ValueError
        Kind is not available in github.
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
