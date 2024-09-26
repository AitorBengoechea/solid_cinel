# -*- coding: utf-8 -*-
"""
Python file for working with alpha function.

@author: AB272525
"""
from scipy.constants import physical_constants as const
from solid_cinel.core.dynamic_structure.beta import Beta
from solid_cinel.core.material.pdos import Pdos
from typing import Iterable, Union
import numpy as np
import pandas as pd
import numba as nb
from numba import prange, float64
from math import exp

# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]


class Alpha:
    """
    Class with all the method for the creation and manipulation of alpha
    grids

    Attributes
    ----------
    data : 'np.ndarray'
        Array of alpha values.
    to_index : 'pd.Index'
        pandas Index of alpha values.

    Methods
    -------
    generate_grid -> 'Alpha
        Generate a alpha grid for a given temperature and atomic mass
    from_parameters -> 'Alpha'
        Generate the alpha values for the given combination of the input
        parameters
    scale -> 'Alpha'
        Scale alpha or beta spectrum
    get_theta -> pd.Series
        Based on the S(alpha, -beta) matrix, get the posible scattering angles
        for a scattering atom, temperature and incident neutron energy.
     """

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
    def data(self, vector: Iterable):
        """
        Construct the S(alpha, -beta) matrix and check if the data achieve the
        normalization and sum rule constrain.

        Parameters
        ----------
        df : 2D iterable, (N, M)
            Iterable containing the S(alpha, -beta) matrix.
        """
        # Sort and define the style of the dataframe:
        vector_ = np.unique(vector)

        # Alpha constrains:
        if (vector_ < 0).any():
            raise ValueError("Alpha values must be positive")

        # save the data:
        self._data = vector_

    @property
    def to_index(self) -> pd.Index:
        """Tranform the Beta class data into a pandas Index."""
        return pd.Index(self.data, name="alpha")

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
        >>> Alpha.generate_grid(300, 26, num_grid=10).data.round(6)
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
    def from_parameters(cls, Eout: Union[Iterable, float],
                        Ein: Union[Iterable, float],
                        T: Union[Iterable, float], M: float,
                        theta: Union[Iterable, float]):
        """
        Generate the alpha values for the given combination of the input
        parameters:
        .. math::
            \alpha = \frac{E^\prime + E - 2 \mu\sqrt{E^\prime E}}{Ak_BT}

        Parameters
        ----------
        Eout : 1D iterable or 'float'
            Neutron output energies in eV.
        Ein : 1D iterable or 'float'
            Neutron incident energy in eV.
        T : 1D iterable or 'float'
            Temperature in Kelvin.
        M : 'float'
            Atom mass, amu
        theta : 1D iterable or 'float'
            scattering angle in Degrees.

        Returns
        -------
        "Alpha"
            Alpha grid generated for the given combination of the input
            parameters.

        Example
        -------
        >>> T = 800
        >>> Ein = 0.33118
        >>> Eout = [0.331180, 0.331812, 0.332445, 0.333077, 0.333710]
        >>> M = 26.98153433356103
        >>> theta = 0.101125 * 180 / np.pi
        >>> Alpha.from_parameters(Eout, Ein, T, M, theta).data.round(6)
        array([0.001835, 0.001837, 0.001839, 0.001842, 0.001845])
        """
        # Calculate the cosine of the scattering angle
        mu = np.cos(theta * np.pi / 180)

        return cls(get_alpha(Eout, Ein, T, M, mu))

    @classmethod
    def from_recoil(cls, Ein: [int, float, np.ndarray] , T: float,
                        M: float):
        """
        Generate the alpha values using the recoil energy.

        Parameters
        ----------
        Ein: 'int', 'float' or 'np.ndarray'
            Incident energy in eV.
        T: 'float'
            Temperature in K.
        M: 'float'
            Mass in amu.

        Returns
        -------
        "Alpha"
            Alpha grid generated for the given combination of the input
            parameters.

        Example
        -------
        >>> T = 800
        >>> Ein = np.array([0.33, 0.4, 0.8, 1.5, 2.33118])
        >>> M = 26.98153433356103
        >>> Alpha.from_recoil(Ein, T, M).data.round(6)
        array([0.118447, 0.155038, 0.36413 , 0.730042, 1.164525])
        """
        return cls(get_gressierRecoil(Ein, T, M) / (kb * T))

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

    def get_recoil(self, T: float) -> pd.Series:
        """
        Get the recoil energy for a given temperature.

        Parameters
        ----------
        T: 'float'
            Temperature in K.

        Returns
        -------
        "pd.Series"
            Recoil energy for a given temperature.

        Example
        -------
        >>> T = 800
        >>> Ein = np.array([0.33, 0.4, 0.8, 1.5, 2.33118])
        >>> M = 26.98153433356103
        >>> alpha = Alpha.from_recoil(Ein, T, M)
        >>> pd.Series(alpha.get_recoil(T), index=alpha.data).round(6)
        0.118447    0.008166
        0.155038    0.010688
        0.364130    0.025103
        0.730042    0.050328
        1.164525    0.080281
        dtype: float64
        """
        return self.data * kb * T

    def get_expansPorcen(self, pdos: Pdos, T: float) -> np.ndarray:
        """
        Using phonon expansion method, determine the percentage lost due to
        zero phonon term

        Parameters
        ----------
        pdos: Pdos
            Pdos object
        T: float
            Temperature in Kelvin

        Returns
        -------
        np.ndarray
            Percentage of the Xs calculate using the phono expansion model

        Examples
        --------
        >>> from solid_cinel.core.material import Pdos
        >>> from solid_cinel.data.examples.UO2 import rho_in_energy_U238, interv_in_energy_U238
        >>> T = 800
        >>> Ein = np.array([0.33, 0.4, 0.8, 1.5, 2.33118])
        >>> M = 26.98153433356103
        >>> alpha = Alpha.from_recoil(Ein, T, M)
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> pd.Series(alpha.get_expansPorcen(pdos, T), index=Ein).round(6)
        0.33000    0.99965
        0.40000    0.99997
        0.80000    1.00000
        1.50000    1.00000
        2.33118    1.00000
        dtype: float64

        >>> T = 300
        >>> Ein = np.array([0.33, 0.4, 0.8, 1.5, 2.33118])
        >>> M = 238.05077040419212
        >>> alpha = Alpha.from_recoil(Ein, T, M)
        >>> pd.Series(alpha.get_expansPorcen(pdos, T), index=Ein).round(6)
        0.33000    0.366062
        0.40000    0.431847
        0.80000    0.696211
        1.50000    0.898431
        2.33118    0.972345
        dtype: float64
        """
        tempPdos = pdos if pdos.type == "Tpdos" else pdos.get_Tpdos(T)
        return 1 - np.exp(- self.data * tempPdos.DebyeWallerCoeff)

    def get_theta(self, T: float, Ein: float, M: float,
                  beta_grid: Union[Beta, Iterable]) -> pd.Series:
        """
        Based on the S(alpha, -beta) matrix, get the posible scattering angles
        for a scattering atom, temperature and incident neutron energy.
        .. math::
            \mu = \frac{E^\prime + E - \alpha Ak_BT}{2\sqrt{E^\prime E}}
            \theta = \arccos(\mu)

        Parameters
        ----------
        beta_grid: 'Beta' or 1D iterable
            Beta grid.
        T : 'float'
            Temperature in K.
        Ein : 'float'
            Incident neutron energy in eV.
        m : 'float'
            Atom mass, amu.

        Returns
        -------
        "pd.Series"
            Series with the theta values for a range of alpha and a fix Ein, M
            and Beta grid.

        Example
        -------
        >>> import numpy as np
        >>> from solid_cinel.data.examples.Al27 import beta0_, alpha0_
        >>> T = 800
        >>> M = 26.98153433356103
        >>> Ein = 0.33118
        >>> beta_grid = Beta(beta0_).scale(T)
        >>> alpha_grid = Alpha(alpha0_).scale(T)
        >>> alpha_grid.get_theta(T, Ein, M, beta_grid).iloc[0:5].round(6)
        alpha
        0.001835    0.101125
        0.003670    0.143002
        0.005505    0.175125
        0.007340    0.202199
        0.009175    0.226045
        Name: mu, dtype: float64

        >>> T = 800
        >>> Ein = 0.33118
        >>> Eout = np.array([0.331180, 0.331812, 0.332445, 0.333077, 0.333710])
        >>> beta_grid = Beta.from_Eout(Eout, Ein, T)
        >>> M = 26.98153433356103
        >>> theta = 45
        >>> alpha = Alpha.from_parameters(Eout, Ein, T, M, theta)
        >>> theta = alpha.get_theta(T, Ein, M, beta_grid)
        >>> theta * 180 / np.pi
        alpha
        0.105201    45.0
        0.105302    45.0
        0.105403    45.0
        0.105504    45.0
        0.105605    45.0
        Name: mu, dtype: float64
        """
        # Get the alpha values
        alpha = self.data

        # Calculate the mass ratio
        A = M / m

        # Calculate the beta values
        beta = beta_grid if isinstance(beta_grid, Beta) else Beta(beta_grid)

        # Calculate the outgoing energy grid
        Eout = beta.get_Eout(T, Ein).values

        # Cut the alpha and Eout values if they have different lengths
        if len(Eout) > len(alpha):
            E_prima = Eout[:len(alpha)]
        elif len(Eout) < len(alpha):
            alpha = alpha[:len(Eout)]

        # Calculate the cosine of the scattering angle
        mu = Eout + Ein - alpha * A * kb * T
        mu /= 2 * np.sqrt(Eout * Ein)
        mu = np.arccos(mu[abs(mu) <= 1])

        return pd.Series(mu, index=Alpha(alpha[:len(mu)]).to_index, name="mu")

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
        >>> alpha0 = Alpha.generate_grid(T, 26, num_grid=10)
        >>> alpha0.scale(T).data.round(6)
        array([1.0280000e-03, 3.2140000e-03, 1.0051000e-02, 3.1428000e-02,
               9.8269000e-02, 3.0727100e-01, 9.6078300e-01, 3.0041990e+00,
               9.3936040e+00, 2.9372154e+01])
        """
        return Alpha(Beta(self.data).scale(T, therm=therm).data)

    def expansionOrder(self, DebyeWallerCoeff: float, decimal: float, orderMax: int) -> int:
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
        >>> alpha_grid = Alpha(alpha0_).scale(T)
        >>> pdos = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> debye_waller = pdos.DebyeWallerCoeff(T)
        >>> alpha_grid.expansionOrder(debye_waller, 1.0e-6, 5000)
        798
        """
        return get_expansionOrder(self.data.max(), DebyeWallerCoeff, decimal, orderMax)


@nb.jit(nopython=True, cache=True)
def get_alphaRecoil(Eout: np.ndarray, Ein: float, M: float, mu: float):
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
def get_alphaFromEout(Eout: np.ndarray, Ein: float, T: float, M: float,
                      mu: float) -> np.ndarray:
    """
    Get the alpha value from the parameters of the function:
    .. math::
        \alpha = \frac{E^\prime + E - 2 \mu\sqrt{E^\prime E}}{Ak_BT}
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
    mu: 'float'
        Cosine of the scattering angle.

    Returns
    -------
    'np.ndarray', (N,)
        Array containing all posible alpha values for the input parameters.
    """
    return get_alphaRecoil(Eout, Ein, M, mu) / (kb * T)


@nb.vectorize(['float64(float64, float64, float64, float64, float64)'],
              cache=True, target='parallel')
def get_alpha(Eout: float, Ein: float, T: float, M: float, mu: float) -> float:
    """
    Get all the posible alpha values from the parameters of the function:
    .. math::
        \alpha = \frac{E^\prime + E - 2 \mu\sqrt{E^\prime E}}{Ak_BT}

    Parameters
    ----------
    Eout : 'np.ndarray', (N,)
        Output energy of the neutron.
    Ein : 'np.ndarray', (M,)
        Incidente energy of the neutron.
    T : 'np.ndarray', (Z,)
        Temperature in K.
    M : "float"
        Mass in amu of the scatterer.
    mu : 'np.ndarray', (K,)
        Cosine of the scattering angle.

    Returns
    -------
    'np.ndarray', (N + M + Z + K,)
        Array containing all posible alpha values for the input parameters.
    """
    return get_alphaFromEout(Eout, Ein, T, M, mu)


@nb.jit(float64[:, :](float64[:], float64, float64, float64, float64[:]),
        nopython=True, nogil=True)
def get_alphaMat(Eout: np.ndarray, Ein: float, T: float, M: float,
                 mu: np.ndarray) -> np.ndarray:
    """
    Get all the posible alpha values from the parameters of the function:
    .. math::
        \alpha = \frac{E^\prime + E - 2 \mu\sqrt{E^\prime E}}{Ak_BT}

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

    Returns
    -------
    'np.ndarray', (K, N)
        Array containing all posible alpha values for the input parameters.

    Example
    -------
    >>> T = 800
    >>> Ein = 0.33118
    >>> Eout = np.array([0.331180, 0.331812, 0.332445, 0.333077, 0.333710])
    >>> M = 26.98153433356103
    >>> theta = np.array([45, 90, 135, 180])
    >>> mu = np.cos(np.deg2rad(theta))
    >>> pd.DataFrame(get_alphaMat(Eout, Ein, T, M, mu).round(6), index=theta, columns=Eout)
         0.331180  0.331812  0.332445  0.333077  0.333710
    45   0.105201  0.105302  0.105403  0.105504  0.105605
    90   0.359179  0.359522  0.359865  0.360208  0.360551
    135  0.613158  0.613743  0.614328  0.614913  0.615498
    180  0.718359  0.719044  0.719730  0.720415  0.721100

    >>> theta = np.array([90])
    >>> mu = np.cos(np.deg2rad(theta))
    >>> pd.DataFrame(get_alphaMat(Eout, Ein, T, M, mu).round(6), index=theta, columns=Eout)
        0.331180  0.331812  0.332445  0.333077  0.333710
    90  0.359179  0.359522  0.359865  0.360208  0.360551
    """
    return get_alphaFromEout(Eout, Ein, T, M, mu[::, np.newaxis])

@nb.jit(nopython=True, cache=True)
def get_alphaMulCumsum(alpha: float, DebyeWallerCoeff: float, orderMax: int) -> np.ndarray:
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
    'np.ndarray', (N,)
        Array containing all posible alpha values for the input parameters.

    Example
    -------
    >>> orderMax = 10
    >>> M = 238.05077040419212
    >>> mu = np.cos(np.deg2rad(np.arange(1, 180, 1)))
    >>> from solid_cinel.core.material import Pdos
    >>> from solid_cinel.data.examples.Al27 import rho_in_energy, interv_in_energy
    >>> pdos = Pdos.from_dE(rho_in_energy, interv_in_energy)
    >>> T = 300
    >>> debye_waller = pdos.DebyeWallerCoeff(T)
    >>> Ein = 6.68
    >>> alphaMat = get_alphaMat(np.linspace(Ein * 0.9 , Ein * 1.1, 5000), Ein, T, M, mu)
    >>> alphaCumsum = get_alphaMulCumsum(alphaMat[-1, -1], debye_waller, orderMax)
    >>> pd.Series(alphaCumsum, index=np.arange(1, orderMax + 1)).round(6)
    1     0.000001
    2     0.000011
    3     0.000065
    4     0.000287
    5     0.001018
    6     0.003017
    7     0.007711
    8     0.017351
    9     0.034950
    10    0.063864
    dtype: float64
    """
    alphaMul = np.zeros(orderMax)

    # Zero phonon expansion:
    iterSum = np.log(alpha * DebyeWallerCoeff)
    alphaMul[0] += np.exp(- alpha * DebyeWallerCoeff + iterSum)

    # Higher phonon expansion (nphonon >= 1):
    for n in range(1, orderMax):
        iterSum += np.log(alpha * DebyeWallerCoeff / (n + 1))
        alphaMul[n] += np.exp(- alpha * DebyeWallerCoeff + iterSum)
    return alphaMul.cumsum()


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
        return get_alphaMat(Eout, Ein, T, M, mu)
    # Define the constants for the calculation
    AkbT = M / m * kb * T
    expTerm = exp(- alpha0 * DebyeWallerCoeff)

    # Crete the alpha matrix filled with alpha capture:
    alphaMat = np.full((len(mu), len(Eout)), Ein / AkbT)

    # Modify the alpha matrix with outgoing energy and cosine of the scattering angle modification
    alphaMat += (Eout - 2 * mu[:, np.newaxis] * np.sqrt(Eout * Ein)) / AkbT * expTerm
    return alphaMat


@nb.jit(nopython=True, cache=True)
def get_expansionOrder(alpha: [float, np.ndarray], DebyeWallerCoeff: float,
                       decimal: int, orderMax: int) -> int:
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
    >>> decimal = 1.0e-6
    >>> orderMax = 5000
    >>> M = 238.05077040419212
    >>> mu = np.cos(np.deg2rad(np.arange(1, 180, 1)))
    >>> from solid_cinel.core.material import Pdos
    >>> from solid_cinel.data.examples.Al27 import rho_in_energy, interv_in_energy
    >>> pdos = Pdos.from_dE(rho_in_energy, interv_in_energy)
    >>> T = 300
    >>> debye_waller = pdos.DebyeWallerCoeff(T)
    >>> Ein = 6.68
    >>> alphaMat = get_alphaMat(np.linspace(Ein * 0.9 , Ein * 1.1, 5000), Ein, T, M, mu)
    >>> get_expansionOrder(alphaMat, debye_waller, decimal, orderMax)
    38

    >>> Ein =  36.68
    >>> alphaMat = get_alphaMat(np.linspace(Ein * 0.9 , Ein * 1.1, 5000), Ein, T, M, mu)
    >>> get_expansionOrder(alphaMat, debye_waller, decimal, orderMax)
    138

    >>> T = 1474
    >>> debye_waller = pdos.DebyeWallerCoeff(T)
    >>> Ein = 6.68
    >>> alphaMat = get_alphaMat(np.linspace(Ein * 0.9 , Ein * 1.1, 5000), Ein, T, M, mu)
    >>> get_expansionOrder(alphaMat, debye_waller, decimal, orderMax)
    121

    >>> Ein = 36.68
    >>> alphaMat = get_alphaMat(np.linspace(Ein * 0.9 , Ein * 1.1, 5000), Ein, T, M, mu)
    >>> get_expansionOrder(alphaMat, debye_waller, decimal, orderMax)
    524

    >>> Ein = 100
    >>> alphaMat = get_alphaMat(np.linspace(Ein * 0.9 , Ein * 1.1, 5000), Ein, T, M, mu)
    >>> get_expansionOrder(alphaMat, debye_waller, decimal, orderMax)
    1320
    """
    # Get the maximun alpha value
    alphaMax = alpha if isinstance(alpha, (int, float)) else alpha.max()

    # Get the cumulative sum of the alpha values
    alphaCumsum = get_alphaMulCumsum(alphaMax, DebyeWallerCoeff, orderMax)

    # Check the decimal precision
    nMin = np.argmax((1 - alphaCumsum) <= decimal)

    # If the decimal precision is not reached, the difference between the
    # cumulative sum of the alpha values will identify the order of the
    # expansion.
    return nMin if nMin > 0 else checkDiff(alphaCumsum, decimal, orderMax)


@nb.jit(nopython=True, cache=True)
def checkDiff(alphaCumsum: np.ndarray, decimal: float, orderMax: int) -> int:
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


@nb.jit(nopython=True, cache=True)
def get_gressierRecoil(Ein: [int, float, np.ndarray] , T: float,
                        M: float) -> np.ndarray:
    """
    Get the recoil energy for a given incident energy, temperature and mass.

    Parameters
    ----------
    Ein: 'int', 'float' or 'np.ndarray'
        Incident energy in eV.
    T: 'float'
        Temperature in K.
    M: 'float'
        Mass in amu.

    Returns
    -------
    'np.ndarray'
        Recoil energy in eV.

    Example
    -------
    >>> T = 800
    >>> Ein = np.array([0.33, 0.4, 0.8, 1.5, 2.33118])
    >>> M = 26.98153433356103
    >>> get_gressierRecoil(Ein, T, M).round(6)
    array([0.008166, 0.010688, 0.025103, 0.050328, 0.080281])
    """
    return m / (m + M) * (Ein - 3 / 2 * kb * T)

@nb.jit(nopython=True, cache=True)
def get_recoilMat(Ein: np.ndarray, T: [float, np.ndarray], M: float) -> np.ndarray:
    """
    Get the recoil energy for a given incident energy, temperature and mass.

    Parameters
    ----------
    Ein: 'np.ndarray', (N,)
        Incident energy in eV.
    T: 'float'
        Temperature in K.
    M: 'float'
        Mass in amu.

    Returns
    -------
    'np.ndarray', (N,)
        Recoil energy in eV.

    Example
    -------
    >>> T = 800
    >>> Ein = np.array([[0.33, 0.4, 0.8, 1.5, 2.33118], [0.4, 0.5, 0.9, 1.6, 2.43118]])
    >>> M = 26.98153433356103
    >>> pd.DataFrame(get_recoilMat(Ein, T, M)).round(6)
              0         1         2         3         4
    0  0.008166  0.010688  0.025103  0.050328  0.080281
    1  0.010688  0.014292  0.028706  0.053932  0.083884

    >>> T = np.array([300, 800])
    >>> pd.DataFrame(get_recoilMat(Ein, T, M)).round(6)
              0         1         2         3         4
    0  0.010495  0.013017  0.027432  0.052657  0.082610
    1  0.010688  0.014292  0.028706  0.053932  0.083884
    """
    recoil_mat = np.zeros(Ein.shape)
    if isinstance(T, (int, float)):
        for i in prange(Ein.shape[0]):
            recoil_mat[i] += get_gressierRecoil(Ein[i], T, M)
    else:
        for i in prange(Ein.shape[0]):
            recoil_mat[i] += get_gressierRecoil(Ein[i], T[i], M)
    return recoil_mat
