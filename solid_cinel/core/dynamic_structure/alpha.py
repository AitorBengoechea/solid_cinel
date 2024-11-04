# -*- coding: utf-8 -*-
"""
Python file for working with alpha function.

@author: AB272525
"""
from scipy.constants import physical_constants as const
from solid_cinel.core.dynamic_structure.beta import Beta
from solid_cinel.core.material.pdos import Pdos
from solid_cinel.core.generic import to_arrays
from typing import Iterable, Union
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from math import exp, log

# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]

class AlphaBase:
    def __init__(self, alpha: Iterable):
        """
        Initialize the Alpha class

        Parameters
        ----------
        array : Iterable
            Array of alpha values
        """
        self.data = alpha

    @property
    def data(self) -> pd.DataFrame:
        """Dataframe with the S(alpha, -beta) matrix values."""
        return self._data

    @data.setter
    def data(self, alphaData: Iterable):
        """
        Construct the S(alpha, -beta) matrix and check if the data achieve the
        normalization and sum rule constrain.

        Parameters
        ----------
        df : 2D iterable, (N, M)
            Iterable containing the S(alpha, -beta) matrix.
        """
        # Sort and define the style of the dataframe:
        alphaData_ = np.unique(alphaData)

        # Alpha constrains:
        if (alphaData_ < 0).any():
            raise ValueError("Alpha values must be positive")

        # save the data:
        self._data = alphaData_

    def recoil(self, T: float) -> np.ndarray:
        return self.data * kb * T

    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def checkDiff(alphaCumsum: np.ndarray, decimal: float,
                  orderMax: int) -> int:
        """
        Check the difference between the cumulative sum of the alpha values because
        the cumulative sum can not reach the unity, so the difference between the
        cumulative sum value will identify the order of the expansion.

        Parameters
        ----------
        alphaCumsum: 'np.ndarray', (N,)
            alpha cumulative sum.
        decimal: 'float'
            Decimal precision
        orderMax: 'int'
            Maximun order for the expansion.

        Returns
        -------
        n: 'int'
            Expansion order.
        """
        # Check the difference between the cumulative sum of the alpha values
        alphaCumsumDiff = np.diff(alphaCumsum)
        n = alphaCumsumDiff == 0.0

        # If the difference is zero, the expansion order is the firts value
        if n.any():
            return np.argmax(n)
        else:
            # If the difference is not zero, the expansion order is compare with
            # the decimal precision
            n = alphaCumsumDiff <= decimal
            return np.argmax(n) if n.any() else orderMax

    def mulCumSum(self, DebyeWallerCoeff: float, orderMax: int) -> np.ndarray:
        """
        Get the alpha multiplication for the phonon expansion cumulative sum for
        the given alpha value and Debye Waller coefficient and the maximun order

        Parameters
        ----------
        alpha: 'np.ndarray', (N,) or (N, M)
            alpha grid values.
        DebyeWallerCoeff: 'float'
            Debye Waller coefficient.
        orderMax: 'int'
            Maximun order for the expansion.

        Returns
        -------
        'np.ndarray'
            Cumulative sum of the alpha values.
        """
        # Define the constant:
        alphaMul = np.zeros(orderMax)
        alphaDebye = self.data.max() * DebyeWallerCoeff
        log_alphaDebye = log(alphaDebye)


        # 1 phonon expansion:
        iterSum = log(alphaDebye)
        alphaMul[0] += exp(iterSum)

        # Higher phonon expansion (nphonon >= 1):
        for n in range(1, orderMax):
            iterSum += log_alphaDebye - log(n + 1)
            alphaMul[n] += exp(iterSum)
        return exp(- alphaDebye) * alphaMul.cumsum()

    def _expansionOrderMax(self, alphaCumsum: np.ndarray, decimal: float,
                       orderMax: int) -> int:
        """
        Get the expansion order for the phonon expansion method using the maximun
        alpha value and the decimal precision.

        Parameters
        ----------
        alphaCumsum : np.ndarray
            Cumulative sum of the alpha values.
        decimal : float
            Decimal precision.
        orderMax : int
            Maximun order for the expansion.

        Returns
        -------
        int
            Expansion order.
        """
        # Check the decimal precision
        nMin = np.argmax((1 - alphaCumsum) <= decimal)

        # If the decimal precision is not reached, the difference between the
        # cumulative sum of the alpha values will identify the order of the
        # expansion.
        return nMin if nMin > 0 else self.checkDiff(alphaCumsum, decimal, orderMax)

    def _expansionOrderMin(self, alphaCumsum: np.ndarray,
                           decimal: float) -> int:
        """
        Get the expansion order for the phonon expansion method using the maximun

        Parameters
        ----------
        alphaCumsum : np.ndarray
            Cumulative sum of the alpha values.
        decimal : float
            Decimal precision.

        Returns
        -------
        int
            Expansion order.
        """
        # Check the decimal precision
        return np.searchsorted(alphaCumsum, decimal, side='right')

    def expansionOrder(self, DebyeWallerCoeff: float, decimal: float,
                       orderMax: int) -> int:
        """
        Get the expansion order for the phonon expansion method using the maximun
        alpha value and the decimal precision.
        .. math::
            \exp(-\alpha\lambda)\sum_{n=0}^{N}\dfrac{(\alpha\lambda)^n}{n!} = 1.0

        Parameters
        ----------
        alpha: 'np.ndarray', (N,) or (N, M)
            alpha grid values.
        DebyeWallerCoeff: 'float'
            Debye Waller coefficient.
        decimal: 'float'
            Decimal precision
        orderMax: 'int'
            Maximun order for the expansion.

        Returns
        -------
        n: 'int'
            Expansion order.

        Example
        -------
        >>> from solid_cinel.core.material import Pdos
        >>> from solid_cinel.data.examples.Al27 import beta0_, alpha0_, rho_in_energy, interv_in_energy
        >>> T = 800
        >>> alpha_grid = AlphaVect(alpha0_).scale(T)
        >>> pdos = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> debye_waller = pdos.DebyeWallerCoeff(T)
        >>> expan = alpha_grid.expansionOrder(debye_waller, 1.0e-6, 5000)
        >>> assert expan == 798
        """
        # Get the cumulative sum of the alpha values
        alphaCumsum = self.mulCumSum(DebyeWallerCoeff, orderMax)

        # Get the expansion order
        return self._expansionOrderMax(alphaCumsum, decimal, orderMax)

    def expansionRange(self,  DebyeWallerCoeff: float, decimal: float,
                       orderMax: int) -> (int, int):
        # Get the cumulative sum of the alpha values
        alphaCumsum = self.mulCumSum(DebyeWallerCoeff, orderMax)

        # Get the expansion order
        nMax = self._expansionOrderMax(alphaCumsum, decimal, orderMax)

        # Get the expansion range
        nMin = self._expansionOrderMin(alphaCumsum, decimal)

        # Return the expansion range
        return nMin, nMax

    def update(self, newValues):
        np.copyto(self.data, newValues)


class AlphaDynamic(AlphaBase):
    """
    Abstract class for the alpha grid calculation for the function:
    .. math::
        \alpha = \frac{E^\prime + E - 2 \mu\sqrt{E^\prime E}}{A * kb * T}
    """
    __slots__ = ['Ein', 'M', 'T', 'Eout', 'mu']
    def __init__(self, Eout: np.ndarray, Ein: [float, np.ndarray], T: float,
                 M: float, mu: np.ndarray, alpha: np.ndarray):
        """
        Initialize the AlphaDynaMic class

        Parameters
        ----------
        Eout : np.ndarray
            Output energy of the neutron in eV.
        Ein : float
            Incidente energy of the neutron in eV.
        T : float
            Temperature in K.
        M : float
            Mass in amu of the scatterer.
        mu : np.ndarray
            Cosine of the scattering angle.
        alpha : np.ndarray
            Array of alpha values
        """
        self.Eout = to_arrays(Eout)
        self.Ein = Ein
        self.T = T
        self.M = M
        self.mu = to_arrays(mu)
        super().__init__(alpha)

    @classmethod
    def from_param(cls, Ein: float, M: float, T: float, Eout: np.ndarray,
                   mu: np.ndarray) -> 'AlphaDynamic':
        """
        Initialize the AlphaDynaMic class from the parameters of the function:
        .. math::
            \alpha = \frac{E^\prime + E - 2 \mu\sqrt{E^\prime E}}{A * kb * T}

        Parameters
        ----------
        Eout : np.ndarray
            Output energy of the neutron in eV.
        Ein : float
            Incidente energy of the neutron in eV.
        M : float
            Mass in amu of the scatterer.
        T : float
            Temperature in K.
        mu : np.ndarray
            Cosine of the scattering angle.

        Returns
        -------
        'AlphaDynamic'
            AlphaDynamic class initialized
        """
        Eout_ = to_arrays(Eout)
        mu_ = to_arrays(mu)
        return cls(Eout_, Ein, T, M, mu_,
                   calc_alpha(Ein, M, T, Eout_, mu_[::, np.newaxis]))

    @classmethod
    def from_capt(cls, Ein: np.ndarray, M: float, T: float) -> "AlphaDynamic":
        """
        Initialize the AlphaDynaMic class from the parameters of the function:
        .. math::
            \alpha = \frac{E}{A * kb * T}
        Parameters
        ----------
        Ein : np.ndarray
            Incidente energy of the neutron in eV.
        M : float
            Mass in amu of the scatterer.
        T : float
            Temperature in K.

        Returns
        -------
        'AlphaDynamic'
            AlphaDynamic class initialized
        """
        return cls.from_param(Ein, M, T, 0.0, 0.0)

    @property
    def recoil(self) -> np.ndarray:
        """
        Get the alpha recoil energy.

        Returns
        -------
        np.ndarray
            Array with the alpha recoil energy.
        """
        return super().recoil(self.T)


class AlphaVect(AlphaBase):
    """
    Abstract class for the alpha grid.
    """
    def __init__(self, alpha: np.ndarray):
        """
        Initialize the Alpha class

        Parameters
        ----------
        array : Iterable
            Array of alpha values
        """
        super().__init__(alpha)

    @classmethod
    def generate_grid(cls, T: float, M: float, num_grid: int = 300,
                      min_E: float = 2.8e-3, thermal_threshold: float = 5.,
                      scale: bool = False, **kwargs):
        """
        Generate a alpha grid for a given temperature and atomic mass.

        Parameters
        ----------
        T : 'float'
            Temperature in K.
        M : 'float'
            atomic mass of scatterer in amu.
        num_grid : 'int', optional
            Number of grid. The default is 400.
        mid_E : 'float', optional
            minimum of energy transfer in eV. The default is 0.08.
        thermal_threshold : 'float', optional
            thermal energy threshold in eV. The default is 5.
        scale : 'bool', optional
            Option to scale beta and alpha grid with the method scale_grid. The
            default is False.

        Parameters for scale_grid
        -------------------------
        therm : 'float', optional
            factor for regrid alpha and beta. The default is 0.0253.

        Returns
        -------
        "Alpha"
            Generate grid from minimun alpha to maximun alpha for a certain
            range of energies.

        Example
        -------
        >>> AlphaVect.generate_grid(300, 26, num_grid=10).data.round(6)
        array([1.0500000e-03, 3.2850000e-03, 1.0270000e-02, 3.2114000e-02,
               1.0041300e-01, 3.1397500e-01, 9.8174500e-01, 3.0697450e+00,
               9.5985550e+00, 3.0013001e+01])
        """
        # Calculate the constant AkT
        AkT = M * kb * T / m

        # Calculate the minimum alpha value
        min_alpha = min_E / 4 / AkT

        # Calculate the maximum alpha value
        max_alpha = 4 * thermal_threshold / AkT

        # Generate the alpha grid
        alpha_grid = np.logspace(np.log10(min_alpha), np.log10(max_alpha),
                                 num=num_grid)

        # Scale the alpha grid if the scale option is True
        return cls(alpha_grid).scale(T, **kwargs) if scale else cls(alpha_grid)

    @classmethod
    def from_file(cls, file_path: str, delimiter: str = None, skiprows: int = 0,
                  usecols: list = None):
        """
        Read a 1D array from a file.

        Parameters
        ----------
        file_path : str
            The path to the file.
        delimiter : str, optional
            The string used to separate values in the file.
        skiprows : int, optional
            The number of lines to skip at the beginning of the file.
        usecols : int or sequence, optional
            Which columns to read, with 0 being the first.

        Returns
        -------
        "Alpha"
            Alpha grid generated for the given combination of the input
            parameters.
        """
        return cls(np.loadtxt(file_path, delimiter=delimiter, skiprows=skiprows,
                              usecols=usecols))

    def scale(self, T: float, therm: float = 0.0253):
        """
        Scale alpha or beta spectrum.
        .. math::
            \alpha_{esc}= \alpha * \dfrac{therm}{k_BT}

        Parameters
        ----------
        grid : 'np.ndarray' of 1D or 2D
            Alpha o Beta grid.
        T : 'float'
            Temperature in K.
        therm : 'float', optional
            factor for regrid alpha and beta. The default is 0.0253.

        Returns
        -------
        "Alpha"
            Scaled alpha grid

        Example
        -------
        >>> T = 300
        >>> alpha0 = AlphaVect.generate_grid(T, 26, num_grid=10)
        >>> alpha0.scale(T).data.round(6)
        array([1.0280000e-03, 3.2140000e-03, 1.0051000e-02, 3.1428000e-02,
               9.8269000e-02, 3.0727100e-01, 9.6078300e-01, 3.0041990e+00,
               9.3936040e+00, 2.9372154e+01])
        """
        self.update(self.data * therm / (kb * T))
        return self


@nb.jit(nopython=True, cache=True)
def calc_alphaRecoil(Ein: [float, np.ndarray], M: float, Eout: np.ndarray,
                     mu: np.ndarray) -> np.ndarray:
    """
    Get the alpha recoil value from the parameters of the function:
    .. math::
        \alpha = \frac{E^\prime + E - 2 \mu\sqrt{E^\prime E}}{A}

    Parameters
    ----------
    Eout: 'np.ndarray', (N,)
        Output energy of the neutron in eV.
    Ein: 'float'
        Incidente energy of the neutron in eV.
    M: "float"
        Mass in amu of the scatterer.
    mu: 'float'
        Cosine of the scattering angle.

    Returns
    -------
    'np.ndarray', (N,)
        Array containing all posible alpha values for the input parameters.
    """
    return (Eout + Ein - 2 * mu * np.sqrt(Eout * Ein)) / (M / m)

@nb.jit(nopython=True, cache=True)
def calc_alpha(Ein: [float, np.ndarray], M: float, T: float, Eout: np.ndarray,
               mu: np.ndarray) -> np.ndarray:
    """
    Get the alpha recoil value from the parameters of the function:
    .. math::
        \alpha = \frac{E^\prime + E - 2 \mu\sqrt{E^\prime E}}{A * kb * T}

    Parameters
    ----------
    Eout: 'np.ndarray', (N,)
        Output energy of the neutron in eV.
    Ein: 'float'
        Incidente energy of the neutron in eV.
    M: "float"
        Mass in amu of the scatterer.
    mu: 'float'
        Cosine of the scattering angle.

    Returns
    -------
    'np.ndarray', (N,)
        Array containing all posible alpha values for the input parameters.
    """
    return calc_alphaRecoil(Ein, M, Eout, mu) / (kb * T)

@nb.jit(nopython=True, cache=True)
def get_alphaMatMod(Eout: np.ndarray, Ein: float, T: float, M: float,
                    mu: np.ndarray, DebyeWallerCoeff: float, alpha0: float) -> np.ndarray:
    """
    Get all the posible alpha modified values from the parameters of the function
    Parameters
    ----------
    Eout: 'np.ndarray', (N,)
        Output energy of the neutron in eV.
    Ein: 'float'
        Incidente energy of the neutron in eV.
    T: 'float'
        Temperature in K.
    M: "float"
        Mass in amu of the scatterer.
    mu: 'np.ndarray', (K,)
        Cosine of the scattering angle.
    DebyeWallerCoeff : float
        Debye Waller coefficient.
    alpha0 : float
        Alpha zero value.
    Returns
    -------
    'np.ndarray', (K, N)
        Array containing all posible modified alpha values for the input
        parameters.
    Example
    -------
    >>> T = 800
    >>> Ein = 0.33118
    >>> Eout = np.array([0.331180, 0.331812, 0.332445, 0.333077, 0.333710])
    >>> M = 26.98153433356103
    >>> theta = np.array([45, 90, 135, 180])
    >>> mu = np.cos(np.deg2rad(theta))
    >>> DebyeWallerCoeff = 7.5
    >>> alpha0 = Ein / (M / m * kb * T)
    >>> values = get_alphaMatMod(Eout, Ein, T, M, mu, DebyeWallerCoeff, alpha0)
    >>> pd.DataFrame(values.round(6), index=theta, columns=Eout)
         0.331180  0.331812  0.332445  0.333077  0.333710
    45   0.160246  0.160272  0.160298  0.160324  0.160351
    90   0.226290  0.226379  0.226468  0.226558  0.226647
    135  0.292334  0.292486  0.292639  0.292791  0.292943
    180  0.319691  0.319869  0.320047  0.320225  0.320404

    # Test for the case of a single angle:
    >>> theta = np.array([90])
    >>> mu = np.cos(np.deg2rad(theta))
    >>> values = get_alphaMatMod(Eout, Ein, T, M, mu, DebyeWallerCoeff, alpha0)
    >>> pd.DataFrame(values.round(6), index=theta, columns=Eout)
        0.331180  0.331812  0.332445  0.333077  0.333710
    90   0.22629  0.226379  0.226468  0.226558  0.226647
    """
    if alpha0 == 0 or DebyeWallerCoeff == 0:
        return calc_alpha(Eout, Ein, T, M, mu[::, np.newaxis])
    # Define the constants for the calculation
    AkbT = M / m * kb * T
    expTerm = exp(- alpha0 * DebyeWallerCoeff)

    # Crete the alpha matrix filled with alpha capture:
    alphaMat = np.full((len(mu), len(Eout)), Ein / AkbT)

    # Modify the alpha matrix with outgoing energy and cosine of the scattering angle modification
    return alphaMat + (Eout - 2 * mu[:, np.newaxis] * np.sqrt(Eout * Ein)) / AkbT * expTerm