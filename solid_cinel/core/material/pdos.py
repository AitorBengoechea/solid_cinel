# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:46:42 2022
@author: Aitor Bengoechea
"""

from solid_cinel.core.generic import normalization_coeff, reshape_differential
from solid_cinel.core._numba import tau_n_CPU
import pandas as pd
import numpy as np
import scipy as sp
from scipy.constants import physical_constants as const
from scipy.interpolate import interp1d
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
        self.rho = pd.Series(*args, **kwargs)

    @property
    def rho(self) -> pd.Series:
        """Pandas Series containing the rho values in energy (index)."""
        return self.data

    @rho.setter
    def rho(self, rho_data) -> pd.Series:
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
        rho_ = pd.Series(rho_data, dtype=float, name="rho")

        if not len(rho_.shape) == 1:
            raise TypeError("Rho must have one dimension")

        if not rho_.index.is_monotonic_increasing:
            raise SyntaxError("energy grid is not monotonically increasing")

        self.data = rho_ / normalization_coeff(rho_)

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
        >>> p.rho.iloc[0:10]
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
        return cls(rho_, index=index)

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
        >>> p.rho.iloc[0:5]
        E
        0.0000    0.000000
        0.0008    0.041157
        0.0016    0.164629
        0.0024    0.370415
        0.0032    0.657892
        Name: rho, dtype: float64

        Test the results:
        >>> p.change_grid(T=300).rho.iloc[0:5]
        beta
        0.000000    0.000000
        0.030945    0.001064
        0.061891    0.004256
        0.092836    0.009576
        0.123782    0.017008
        Name: rho, dtype: float64

        >>> p = Pdos([1, 2, 4], index=[1, 2, 4])
        >>> p.change_grid([1, 2, 3, 4, 5]).rho
        E
        1.0    0.105263
        2.0    0.210526
        3.0    0.315789
        4.0    0.421053
        5.0    0.000000
        Name: rho, dtype: float64
        """
        grid = self.rho.index
        if T:
            enew = grid / (const["Boltzmann constant in eV/K"][0] * T)
            enew.name = "beta"
            rho_new = self.rho.values
        if eg:
            enew = grid.union(eg).astype("float")
            enew.name = "E"
            rho_new = reshape_differential(
                grid.values,
                self.rho.values,
                enew.values,
                )
        return Pdos(rho_new, index=enew)

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
        data = self.change_grid(T=T).rho
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

    def _get_tau_1(self, T) -> pd.Series:
        """
        Get the Tau(-beta) function for 1 phonon expansion in LEAPR formalism.

        Parameters
        ----------
        T : 'int'
            Temperature in K.

        Raises
        ------
        ValueError
            Tau function doesnt satisfy normalization condition.

        Examples
        --------
        Object initialization:
        >>> p = Pdos.from_data(rho_in_energy, interv_in_energy)

        Test the results:
        >>> p._get_tau_1(20).iloc[:10]
        beta
        0.000000    0.004250
        0.464181    0.005313
        0.928361    0.006524
        1.392542    0.007875
        1.856723    0.009344
        2.320904    0.010932
        2.785084    0.012606
        3.249265    0.014359
        3.713446    0.016167
        4.177627    0.018020
        Name: 1, dtype: float64
        """
        P = self.P(T)
        beta = P.index.values
        tau1 = P * np.exp(0.5 * beta) / self.DebyeWallerCoeff(T)
        if normalization_coeff(tau1 * (1 + np.exp(-beta))) < 1.e-5:
            raise ValueError("Tau function for 1 phonon expansion doesnt satisfy the normalization condition")
        tau1.name = 1
        return tau1

    @staticmethod
    def _check_tau_norm(tau):
        """
        Check if the tau functions are normalize

        Parameters
        ----------
        tau : 'pd.DataFrame'
            tau functions.

        Raises
        ------
        ValueError
            The tau functions doenst satisfy the normalization.
        """
        norm = tau.apply(lambda x: normalization_coeff(x * (1 + np.exp(-x.index.values))), axis=0)
        if (norm < 1.e-5).any():
            raise ValueError("Tau function doesnt satisfy the normalization condition")
        return 

    def get_tau(self, T, nphonon=1000, beta=None, threshold=1.0e-14,
                norm_check=True) -> pd.DataFrame:
        """
        Get tau function for the selected phonon expansion.
        .. math::
            For n=1:
                \mathcal{T}_1(-\beta)=\dfrac{P(-\beta)\exp(\beta/2)}{\lambda}
            For n>1:
                \mathcal{T}_n(-\beta)=\int_{0}^{\infty}\mathcal{T}_1(-\beta^\prime)\left(\mathcal{T}_{n-1}(-\beta+\beta^\prime)+\exp(-\beta^\prime)\mathcal{T}_{n-1}(-\beta-\beta^\prime)\right)d\beta^\prime

        Parameters
        ----------
        T : 'int'
            Temperature in K.
        nphonon : 'int', optional
            Phonon expansion order. The default is 1.
        beta : '1D iterable', optional
            If this options is activate, the tau function will be linearly
            interpolate to the 1D iterable introduce. The default is None.
        threshold : 'float', optional
            Minimun value to take into account. The default is 1.0e-14.

        Examples
        --------
        Object initialization:
        >>> p = Pdos.from_data(rho_in_energy, interv_in_energy)

        Test the results:
        >>> p.get_tau(20, nphonon=1).iloc[:10]
        tau_n              1 
        beta
        0.000000    0.004250
        0.464181    0.005313
        0.928361    0.006524
        1.392542    0.007875
        1.856723    0.009344
        2.320904    0.010932
        2.785084    0.012606
        3.249265    0.014359
        3.713446    0.016167
        4.177627    0.018020

        >>> p.get_tau(20, nphonon=5).iloc[:10]
        tau_n            1         2         3             4             5
        beta
        0.000000  0.004250  0.000120  0.000004  1.530025e-07  5.908313e-09
        0.464181  0.005313  0.000151  0.000005  1.927121e-07  7.444051e-09
        0.928361  0.006524  0.000188  0.000007  2.420754e-07  9.359578e-09
        1.392542  0.007875  0.000233  0.000008  3.032675e-07  1.174370e-08
        1.856723  0.009344  0.000287  0.000010  3.789125e-07  1.470474e-08
        2.320904  0.010932  0.000351  0.000013  4.721669e-07  1.837448e-08
        2.785084  0.012606  0.000427  0.000015  5.868175e-07  2.291294e-08
        3.249265  0.014359  0.000517  0.000019  7.273949e-07  2.851401e-08
        3.713446  0.016167  0.000621  0.000023  8.993050e-07  3.541201e-08
        4.177627  0.018020  0.000743  0.000029  1.108981e-06  4.388966e-08

        >>> beta = [.000, .025, .050, .075, .100, .125, .150, .175, .200, .225]
        >>> p.get_tau(20, nphonon=5, beta=beta)
        tau_n         1         2         3             4             5
        beta
        0.000  0.004250  0.000120  0.000004  1.530025e-07  5.908313e-09
        0.025  0.004308  0.000122  0.000004  1.551412e-07  5.991025e-09
        0.050  0.004365  0.000123  0.000004  1.572799e-07  6.073737e-09
        0.075  0.004422  0.000125  0.000004  1.594186e-07  6.156450e-09
        0.100  0.004479  0.000127  0.000004  1.615573e-07  6.239162e-09
        0.125  0.004537  0.000128  0.000004  1.636960e-07  6.321874e-09
        0.150  0.004594  0.000130  0.000004  1.658347e-07  6.404587e-09
        0.175  0.004651  0.000131  0.000005  1.679733e-07  6.487299e-09
        0.200  0.004708  0.000133  0.000005  1.701120e-07  6.570011e-09
        0.225  0.004765  0.000135  0.000005  1.722507e-07  6.652724e-09
        """
        tau1 = self._get_tau_1(T)
        tau = [tau1.values]
        delta_beta = tau1.index[1]
        if nphonon > 1:
            tau_n_minus_1 = tau1.values
            for n in range(1, nphonon):
                tau_n = tau_n_CPU(delta_beta, tau1.values,
                                  tau_n_minus_1, threshold)
                tau.append(tau_n)
                tau_n_minus_1 = tau_n
        tau = pd.DataFrame(tau).fillna(0).T
        tau.index = pd.Index(np.arange(tau.shape[0]) * delta_beta,
                             name="beta")
        tau.columns = pd.Index(np.arange(1, nphonon + 1), name="tau_n")

        # Check if the tau functions satisfy the normalization condition
        if norm_check:
            Pdos._check_tau_norm(tau)

        # Change the beta grid for another one introduce by the user. If beta
        # values are not in the original grid, apply linear interpolation.
        if beta is not None:
            reshape_tau_values = reshape_differential(tau.index.values,
                                                      tau.values,
                                                      beta)
            beta_ = pd.Index(beta, name="beta")
            tau = pd.DataFrame(reshape_tau_values,
                               index=beta_,
                               columns=tau.columns)
        return tau
