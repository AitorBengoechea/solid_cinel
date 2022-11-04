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
import warnings

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
        rho_ = pd.Series(rho, dtype=float, name="rho")

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
        E
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
        Name: rho, dtype: float64
        """
        rho_ = np.array(rho)
        index = pd.Index(np.arange(len(rho_)) * interval_energy)
        index.name = "E"
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
        E
        0.0000    0.000000
        0.0008    0.041157
        0.0016    0.164629
        0.0024    0.370415
        0.0032    0.657892
        Name: rho, dtype: float64

        Test the results:
        >>> p.change_grid(T=300).data.iloc[0:5]
        beta
        0.000000    0.000000
        0.030945    0.001064
        0.061891    0.004256
        0.092836    0.009576
        0.123782    0.017008
        Name: rho, dtype: float64

        >>> p = Pdos([1, 2, 4], index=[1, 2, 4])
        >>> p.change_grid([1, 2, 3, 4, 5]).data
        E
        1.0    0.105263
        2.0    0.210526
        3.0    0.315789
        4.0    0.421053
        5.0    0.000000
        Name: rho, dtype: float64
        """
        grid = self.data.index
        if T:
            enew = grid / (const["Boltzmann constant in eV/K"][0] * T)
            enew.name = "beta"
            rho_new = self.data.values
        if eg:
            enew = grid.union(eg).astype("float")
            enew.name = "E"
            rho_new = self.reshape_differential(
                grid.values,
                self.data.values,
                enew.values,
                )
        return self.__class__(rho_new, index=enew)

    def plot(self) -> matplotlib:
        """Plot rho (y) vs grid (x)."""
        return self.data.plot(title='PDOS')

    def P(self, T, threshold=1.e-6) -> pd.Series:
        """
        Calculate P function for LEAPR formalism with PDOS.
        .. math::
            P(\beta^\prime)=\dfrac{\rho(\beta^\prime)}{2\beta^\prime\sinh(\beta^\prime/2)}

        Parameters
        ----------
        T : 'int'
            Temperature in K.
        threshold : 'float', optional
            Value to chech the initial DOS. The default is 1.e-6.

        Raises
        ------
        ValueError
            Initial point of input DOS is not zero.

        Example
        -------
        Object initialization:
        >>> pdos = Pdos.from_data(rho_in_energy, interv_in_energy)

        Test the results:
        >>> T = 300
        >>> pdos.P(T).iloc[0:6].round(6)
        beta
        0.000000    1.111089
        0.030945    1.111045
        0.061891    1.110912
        0.092836    1.110690
        0.123782    1.109328
        0.154727    1.109309
        Name: P, dtype: float64
        """
        data = self.change_grid(T=T).data
        rho_in_beta = data.values
        beta_values = data.index.values
        if abs(beta_values[0]) > threshold:
            raise ValueError("Initial point of input DOS is not zero")
        P_values = np.zeros(len(rho_in_beta))

        # rho_in_beta is assumed to vary as beta^2 in the nearby of 0
        P_values[0] = rho_in_beta[1] / beta_values[1] ** 2

        # Rest of P values calculation:
        P_values[1:] = 0.5 * rho_in_beta[1:] / beta_values[1:] / np.sinh(0.5 * beta_values[1:])
        return pd.Series(P_values, index=data.index, name="P")

    def Teff(self, T, twt=None) -> float:
        """
        Calculate the effective temperature for a certain pdos information.
        .. math::
            overline{T} = \left(w_t+\int_{0}^{\infty}\beta^2P(\beta)\cosh(\beta/2)d\beta\right)T

        Parameters
        ----------
        T : 'int'
            Temperature in K.
        twt : 'float', optional
            Translational weight, for solid is zero. The default is None.

        Example
        -------
        Object initialization:
        >>> p = Pdos.from_data(rho_in_energy, interv_in_energy)

        Test the results:
        >>> p.Teff(T=20).round(4)
        149.1699
        >>> p.Teff(T=80).round(4) 
        159.1632
        """
        data = self.P(T)
        P = data.values
        beta = data.index.values
        Teff_weight = sp.integrate.trapezoid(beta ** 2 * P * np.cosh(0.5 * beta),
                                              x=beta)
        if twt is not None:
            Teff_weight += twt
        return Teff_weight * T

    def DebyeWallerCoeff(self, T) -> float:
        """
        Calculate Debye Waller Coefficient in LEAPR formalism for a certain
        pdos information.
        .. math::
            c=2\int_{0}^{\beta_{\textrm{max}}}P_s(\beta)\cosh(\dfrac{\beta}{2})d\beta

        Parameters
        ----------
        T : 'int'
            Temperature in K.

        Examples
        --------
        Object initialization:
        >>> p = Pdos.from_data(rho_in_energy, interv_in_energy)

        Test the results:
        >>> p.DebyeWallerCoeff(T=20).round(6)
        0.077454
        >>> p.DebyeWallerCoeff(T=80).round(6)
        0.379937
        """
        data = self.P(T)
        P = data.values
        beta = data.index.values
        return 2 * sp.integrate.trapezoid(P * np.cosh(0.5 * beta), x=beta)

    def B(self, T, atomic_mass, anstrom=True) -> float:
        """
        Calculate mean square displacement for a certain pdos information.

        Parameters
        ----------
        T : 'int'
            Temperature in K
        atomic_mass : 'float'
            Atomic mass of the nucleus in amu.
        anstrom : 'bool', optional
            Option to obtain the B unit in A^2. The default is True.

        Examples
        --------
        Object initialization:
        >>> p = Pdos.from_data(rho_in_energy, interv_in_energy)
        >>> atomic_mass = 26.98153433356103

        Test the results:
        >>> T = 20
        >>> p.B(T, atomic_mass).round(6)
        0.274871

        >>> T = 80
        >>> p.B(T, atomic_mass).round(6)
        0.337081
        """
        constant = (4 * sp.constants.c ** 2 * np.pi**2) * const["reduced Planck constant in eV s"][0] ** 2
        constant /= const["atomic mass unit-electron volt relationship"][0] * const["Boltzmann constant in eV/K"][0]
        B = constant * self.DebyeWallerCoeff(T) / (T * atomic_mass)
        if anstrom:
            B *= 1.0e20
        return B    

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
