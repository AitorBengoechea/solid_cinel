# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:46:42 2022
@author: Aitor Bengoechea
"""

import pandas as pd
import numpy as np
import scipy as sp
from scipy.constants import physical_constants as const
import matplotlib

# Examples variables:
rho_in_energy_str = '''
    0 .0066 .0264 .0594 .1055 .1649 .2374 .3232 .4221
    .5342 .6595 .7980 .9497 1.1146 1.2927 1.4839 1.6884
    2.0169 2.4373 2.9366 3.6133 4.6775 7.1346 7.3650
    7.5156 7.6733 7.8309 8.0740 8.4419 9.0595 9.6773
    7.3645 6.2674 5.1965 4.7958 4.8024 4.6841 4.4673
    4.1914 3.8169 3.3439 2.7855 3.2782 5.3082 8.5930
    12.3377 8.4616 5.6695 4.1585 2.6081 0.0
'''
rho_in_energy = np.fromstring(rho_in_energy_str, dtype=np.float64, sep=' ')
interv_in_energy = 0.0008

class Pdos():
    """
    Object containing the method and properties of the phonon density of states
    """

    def __init__(self, *args, **kwargs):
        """
        Initialization of the pdos object

        Parameters
        ----------
        *args : variables
            variables for the creation of the pandas Series.
        **kwargs : 'dict'
            Dictionary to create the pandas series of rho.

        Returns
        -------
        'Pdos'
            Object containing the method and properties of rho in energy.

        """
        self.data = pd.Series(*args, **kwargs)

    @property
    def data(self) -> pd.Series:
        """Pandas Series containing the rho values in energy (index)."""
        return self._data

    @data.setter
    def data(self, rho) -> pd.Series:
        """
        Data setter for rho to ensure the following properties of the data:
            - Shape of the data: 1 dimension
            - Energy index monotoally increasing
            - Rho values normalization

        Parameters
        ----------
        rho : pd.Series
            rho values in energy.

        Raises
        ------
        TypeError
            Rho is not 1 dimension pd.Series
        SyntaxError
            Energy grid is not monotonically increasing.

        Examples
        --------
        Object initialization:
        >>> p = pdos.from_data(rho_in_energy, interv_in_energy)

        Test the results:
        >>> assert sp.integrate.trapezoid(p.data.values, p.data.index) == 1.0
        """
        rho_ = pd.Series(rho, dtype=float, name="rho in energy")

        if not len(rho_.shape) == 1:
            raise TypeError("Rho must have one dimension")

        if not rho_.index.is_monotonic_increasing:
            raise SyntaxError("energy grid is not monotonically increasing")

        self._data = self.normalization(rho_)

    @classmethod
    def from_data(cls, rho, interval_energy):
        """
        Extract rho in energy from the introduced data.

        Parameters
        ----------
        rho : 1D iterable
            rho values.
        interval_energy : 'float'
            Energy interval in eV.

        Examples
        --------
        Object initialization:
        >>> p = Pdos.from_data(rho_in_energy, interv_in_energy)

        Test the results:
        >>> p.data.iloc[0:10]
        0.0000    0.000000
        0.0008    0.041157
        0.0016    0.164629
        0.0024    0.370415
        0.0032    0.657892
        0.0040    1.028308
        0.0048    1.480414
        0.0056    2.015458
        0.0064    2.632193
        0.0072    3.331243
        Name: rho in energy, dtype: float64
        """
        rho_ = np.array(rho)
        index = np.arange(len(rho_)) * interval_energy
        return cls(rho, index=index)

    def change_grid(self, eg=None, T=None):
        """
        Change the energy grid of rho. Two options available:
            - Tranform energy grid in beta grid by introducing T
            - Linearly interpolate a new energy grid that is a union between
              the old one and the new one.

        Parameters
        ----------
        eg : 1D iterable, optional
            New energy grid. The default is None.
        T : 'float', optional
            Temperature to change energy grid to beta grid. The default is
            None.

        Returns
        -------
        pdos
            pdos object with the new grid.

        Examples
        --------
        Object initialization:
        >>> p = Pdos.from_data(rho_in_energy, interv_in_energy)
        >>> p.data.iloc[0:5]
        0.0000    0.000000
        0.0008    0.041157
        0.0016    0.164629
        0.0024    0.370415
        0.0032    0.657892
        Name: rho in energy, dtype: float64

        Test the results:
        >>> p.change_grid(T=300).data.iloc[0:5]
        0.000000    0.000000
        0.030945    0.001064
        0.061891    0.004256
        0.092836    0.009576
        0.123782    0.017008
        Name: rho in energy, dtype: float64

        >>> p = Pdos([1, 2, 4], index=[1, 2, 4])
        >>> p.change_grid([1, 2, 3, 4, 5]).data
        1.0    0.105263
        2.0    0.210526
        3.0    0.315789
        4.0    0.421053
        5.0    0.000000
        Name: rho in energy, dtype: float64
        """
        grid = self.data.index
        if T:
            enew = grid / (const["Boltzmann constant in eV/K"][0] * T)
            rho_new = self.data.values
        if eg:
            enew = grid.union(eg).astype("float").values
            rho_new = self.reshape_differential(
                grid.values,
                self.data.values,
                enew,
                )
        return self.__class__(rho_new, index=enew)

    def plot(self) -> matplotlib:
        """Plot rho (y) vs grid (x)."""
        return self.data.plot(title='PDOS')

    @staticmethod
    def normalization(rho, kind="trapezoidal") -> pd.Series:
        """
        Normalize a pd.Series

        Parameters
        ----------
        rho : pd.Series
            Pandas Series to normalize.
        kind : 'str', optional
            Integration technique. The default is "trapezoidal". Options:
                - trapezoidal: Integrate along the given axis using the
                  composite trapezoidal rule.
                - simpson: Integrate y(x) using samples along the given axis
                  and the composite Simpson’s rule.

        Examples
        --------
        >>> p = pd.Series([1, 2, 4], index=[1, 2, 4])
        >>> Pdos.normalization(p).round(6)
        1    0.133333
        2    0.266667
        4    0.533333
        dtype: float64
        >>> Pdos.normalization(p, kind="simpson").round(6)
        1    0.133333
        2    0.266667
        4    0.533333
        dtype: float64
        """
        y = rho.values
        x = rho.index.values
        if kind == "trapezoidal":
            y_norm = sp.integrate.trapezoid(y, x=x)
        elif kind == "simpson":
            y_norm = sp.integrate.simpson(y, x=x)
        else:
            raise ValueError("kind is not properly introduced")
        return rho / y_norm

    @staticmethod
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
                x, y,
                axis=0,
                copy=False,
                kind="slinear",
                bounds_error=False,
                fill_value=0.,
                assume_sorted=True,
                )
        return foo(xnew)
