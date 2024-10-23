"""
Python file for working with beta function.

@author: AB272525
"""
from typing import Iterable, Union
import numpy as np
import pandas as pd
import numba as nb
from scipy.constants import physical_constants as const

# Constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]


class Beta:
    """
    Class with all the method for the creation and manipulation of beta grids

    Parameters
    ----------
    array : Iterable
        Iterable with the beta grid data

    Attributes
    ----------
    data : "np.ndarray"
        Array with the beta grid data
    to_index : "pd.Index"
        Transform the Beta class data into a pandas Index
    kind : "str"
        Analise the beta grid to know if the beta grid contains only absolute
        values or mix (positive and negative) values

    Methods
    -------
    generate_grid -> "Beta"
        Generate beta grid for a given temperature
    from_dE -> "Beta"
        Generate beta grid for a given temperature using the dE grid
    from_parameters -> "Beta"
        Generate beta grid for user parameters
    get_dE -> "pd.Series"
        Return the dE grid for a given beta grid
    get_Eout -> "pd.Series"
        Return the Eout grid for a given beta grid
    scale -> "Beta"
        Scale the beta grid to a given temperature
    """

    def __init__(self, array: Iterable):
        """
        Initialize the Beta class

        Parameters
        ----------
        array : Iterable
            Iterable with the beta grid data
        """
        self.data = np.unique(array)

    @property
    def to_index(self) -> pd.Index:
        """Transform the Beta class data into a pandas Index."""
        return pd.Index(self.data, name="beta")

    @property
    def kind(self) -> str:
        """
        Analise the beta grid to know if the beta grid contains only absolute
        values or mix (positive and negative) values.

        Returns
        -------
        kind : "str"
            "abs" if the beta grid contains only absolute values or "mix" if
            the beta grid contains mix (positive and negative) values
        """
        if (self.data >= 0).all():
            kind = "abs"
        else:
            kind = "mix"
        return kind

    @property
    def grid(self):
        """
        Return the beta grid. The beta grid is the difference between the
        elements of the beta grid.

        Returns
        -------
        "np.ndarray"
            Array with the beta grid data
        """
        diff = np.ediff1d(self.data)
        return np.append(diff, diff[-1])

    @classmethod
    def from_default(cls, T: float, kind: str = "abs"):
        """
        Generate beta grid for a given temperature using the default beta grid.

        Parameters
        ----------
        T: 'float'
            Temperature in Kelvin.

        Returns
        -------
        "Beta"
            Generate beta grid for a given temperature.
        """
        if kind == "abs":
            return cls(default_absBeta(T))
        elif kind == "mix":
            return cls(default_beta(T))
        else:
            raise ValueError("kind must be 'abs' or 'mix'")

    @classmethod
    def generate_grid(cls, T: float, num_grid: int = 400, mid_E: int = 0.08,
                      thermal_threshold: float = 5., scale: bool = False,
                      **kwargs):
        """
        Generate beta grid for a given temperature. The grid is created using
        linear separated grid from 0eV to mid_E and in logaritmic scale
        separation in between mid_E to thermal_threshold.

        Parameters
        ----------
        T : 'float'
            Temperature in K.
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
        "Beta"
            Generate beta grid for a given temperature.

        Example
        -------
        >>> Beta.generate_grid(300, num_grid=10).data.round(6)
        array([  0.      ,   0.515756,   1.031513,   1.547269,   2.063025,
                 2.578782,   3.094538,  12.280683,  48.735922, 193.408635])
        """
        # Get the first half of the grid:
        mid_beta = mid_E / (kb * T)
        first_half = np.linspace(0, mid_beta,
                                 num=int(num_grid * 0.6),
                                 endpoint=False)

        # Get the second half of the grid:
        max_beta = thermal_threshold / (kb * T)
        second_half = np.logspace(np.log10(mid_beta), np.log10(max_beta),
                                  num=int(num_grid * 0.4),
                                  endpoint=True)

        # Concatenate the two halfs to get the full grid:
        beta_grid = np.concatenate((first_half, second_half))

        # Scale the grid if needed:
        if scale:
            return cls(beta_grid).scale(T, **kwargs)
        else:
            return cls(beta_grid)

    @classmethod
    def from_dE(cls, energy_grid: Iterable, T: float):
        """
        Tranform a energy grid into a beta grid. (use in pdos.py)
        .. math::
            \beta=\dfrac{dE}{k_BT}

        Parameters
        ----------
        energy_grid : 1D iterable, (N,)
            Energy grid.
        T : 'float'
            Temperature in Kelvin.

        Returns
        -------
        "Beta"
            Generate beta grid for a dE.

        Example
        -------
        >>> from solid_cinel.data.examples.Al27 import rho_in_energy, interv_in_energy
        >>> energy_grid = np.arange(len(rho_in_energy)) * interv_in_energy
        >>> T = 300
        >>> Beta.from_dE(energy_grid, T).data[0:5].round(6)
        array([0.      , 0.030945, 0.061891, 0.092836, 0.123782])
        """
        return cls(np.array(energy_grid) / (kb * T))

    @classmethod
    def from_Eout(cls, Eout: np.ndarray, Ein: float, T: float):
        """
        Generate a beta grid based on the output energies and the incident
        neutron energy.
        .. math::
            \beta=\dfrac{E_{out} - E_{in}}{k_BT}

        Parameters
        ----------
        Eout : np.ndarray
            Neutron output energies in eV.
        Ein : 'float'
            Neutron incident energy in eV.
        T : 1D iterable or 'float'
            Temperature in Kelvin.

        Returns
        -------
        "Beta"
            Generate beta grid from the parameters of the equation.

        Example
        -------
        >>> T = 800
        >>> Ein = 0.33118
        >>> Eout = np.array([0.331180, 0.331812, 0.332445, 0.333077, 0.333710])
        >>> Beta.from_Eout(Eout, Ein, T).data.round(6)
        array([0.      , 0.009168, 0.01835 , 0.027517, 0.036699])
        """
        return cls(get_AbsBeta(Eout, Ein, T))

    @classmethod
    def from_Ein(cls, Eout: float, Ein: np.ndarray, T: float):
        """
        Generate a beta grid based on the incident energies and the output
        neutron energy.
        .. math::
            \beta=\dfrac{E_{out} - E_{in}}{k_BT}
        Parameters
        ----------
        Eout: 'float'
            Neutron output energy in eV.
        Ein: np.ndarray
            Neutron incident energies in eV.
        T: 'float'
            Temperature in Kelvin.

        Returns
        -------
        Returns
        -------
        "Beta"
            Generate beta grid from the parameters of the equation.

        Example
        -------
        >>> T = 800
        >>> Eout = 0.33118
        >>> Ein = np.array([0.331180, 0.331812, 0.332445, 0.333077, 0.333710])
        >>> Beta.from_Ein(Eout, Ein, T).data.round(6)
        array([0.      , 0.009168, 0.01835 , 0.027517, 0.036699])
        """
        return cls(get_AbsBeta(Ein, Eout, T))

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

    def get_dE(self, T: float) -> np.ndarray:
        """
        Get the dE from a beta grid:
        .. math::
            dE = \beta k_B T

        Parameters
        ----------
        T : 'float'
            Temperature in K.

        Returns
        -------
        "pd.Series"
            Series containing the dE in the grid

        Example
        -------
        >>> from solid_cinel.data.examples.Al27 import beta0_
        >>> T = 800
        >>> beta_grid = Beta(beta0_).scale(T)
        >>> pd.Series(beta_grid.get_dE(T), index=beta_grid.data).iloc[0:5]
        0.000000    0.000000
        0.009175    0.000633
        0.018350    0.001265
        0.027524    0.001898
        0.036699    0.002530
        dtype: float64
        """
        return self.data * kb * T

    def get_Eout(self, T: float, Ein: Union[Iterable, float],
                 side: str = "upscattering") -> pd.Series:
        """
        Based on the S(alpha, -beta) matrix, get the posible
        output energies for a incident neutron energy and that beta grid.
        .. math::
            E^\prime = \beta k_B T + E

        Parameters
        ----------
        T : 'float'
            Temperature in K.
        Ein : 1D iterable or 'float'
            Incident neutron energy in eV.
        side : 'str', optional
            Argument to chose the outgoing energy grid side. The default is
            the "upscatterign" side. Available options are:
                - "upscattering" : Eout > Ein
                - "downscattering" : Eout < Ein
                - "full": "upscattering" side + "downscattering" side.

        Returns
        -------
        "pd.Series"
            Series with the output energy

        Example
        -------
        >>> from solid_cinel.data.examples.Al27 import beta0_
        >>> T = 800
        >>> Ein = 0.33118
        >>> beta_grid = Beta(beta0_).scale(T)
        >>> beta_grid.get_Eout(T, Ein).iloc[0:5]
        beta
        0.000000    0.331180
        0.009175    0.331812
        0.018350    0.332445
        0.027524    0.333077
        0.036699    0.333710
        Name: Eout, dtype: float64

        >>> beta_grid.get_Eout(T, Ein, side="downscattering").iloc[0:5]
        beta
        -0.000000    0.331180
        -0.009175    0.330547
        -0.018350    0.329915
        -0.027524    0.329282
        -0.036699    0.328650
        Name: Eout, dtype: float64

        >>> beta_grid.get_Eout(T, Ein, side="full").iloc[104:113]
        beta
        -0.036699    0.328650
        -0.027524    0.329282
        -0.018350    0.329915
        -0.009175    0.330547
         0.000000    0.331180
         0.009175    0.331812
         0.018350    0.332445
         0.027524    0.333077
         0.036699    0.333710
        Name: Eout, dtype: float64
        """
        # Get the dE grid:
        dE = pd.Series(self.get_dE(T), index=pd.Index(self.data, name="beta"))
        dE.name = "Eout"

        # Get the upscattering energy grid:
        Eout_positive = Ein + dE
        if side == "upscattering":
            return Eout_positive

        # Get the downscattering energy grid:
        Eout_negative = Ein - dE
        Eout_negative.index *= -1

        # Remove the negative downscattering grid values:
        Eout_negative = Eout_negative[Eout_negative >= 0]
        if side == "downscattering":
            return Eout_negative

        # Concatenate the two sides:
        if side == "full":
            Eout = pd.concat([Eout_negative.iloc[1::], Eout_positive]).sort_index()
            return Eout
        else:
            raise SyntaxError("Side option not available")

    def scale(self, T: float, therm: float = 0.0253):
        """
        Scale alpha or beta spectrum.
        .. math::
            \beta_{esc}= \beta * \dfrac{therm}{k_BT}

        Parameters
        ----------
        T : 'float'
            Temperature in K.
        therm : 'float', optional
            factor for regrid alpha and beta. The default is 0.0253.

        Returns
        -------
        "Beta"
            Beta grid scaled.

        Example
        -------
        >>> T = 300
        >>> beta0 = Beta.generate_grid(T, num_grid=10)
        >>> beta0.scale(T).data.round(6)
        array([  0.      ,   0.504744,   1.009488,   1.514231,   2.018975,
               2.523719,   3.028463,  12.018462,  47.695298, 189.278915])
        """
        scale_grid = self.data * therm / (kb * T)
        return Beta(scale_grid)

@nb.jit(nopython=True, cache=True)
def calc_dE(Eout: [np.ndarray, float], Ein: [np.ndarray, float]) -> np.ndarray:
    """
    Calculate the dE values from the parameters of the function:
    .. math::
        dE = \beta k_B T

    Parameters
    ----------
    beta : 'np.ndarray', (N,) or 'float'
        Beta values.
    T : float
        Temperature in K.

    Returns
    -------
    'np.ndarray', (N,)
        Array containing all posible dE values for the input parameters.
    """
    # Get the dE values:
    return Eout - Ein


@nb.jit(nopython=True, cache=True)
def calc_Beta(Eout: [np.ndarray, float], Ein: [np.ndarray, float],
              T: float) -> np.ndarray:
    """
    Calculate the beta values from the parameters of the function:
    .. math::
        \beta=\dfrac{E_{out} - E_{in}}{k_BT}

    Parameters
    ----------
    Eout : 'np.ndarray', (N,) or 'float'
        Output energy of the neutron.
    Ein : 'np.ndarray', (N,) or 'float'
        Incidente energy of the neutron.
    T : float
        Temperature in K.

    Returns
    -------
    'np.ndarray', (N,)
        Array containing all posible beta values for the input parameters.
    """
    # Get the beta values:
    return calc_dE(Eout, Ein) / (kb * T)


@nb.jit(nopython=True, cache=True)
def get_AbsBeta(Eout: Union[np.ndarray, float], Ein: Union[np.ndarray, float],
                T: float, unique: bool = True, sort: bool = True) -> np.ndarray:
    """
    Get the positive beta values from the parameters of the function:
    .. math::
        \beta = \left| \dfrac{E_{out} - E_{in}}{k_B T} \right|

    Parameters
    ----------
    Eout : np.ndarray or float
        Output energy of the neutron.
    Ein : np.ndarray or float
        Incident energy of the neutron.
    T : float
        Temperature in K.
    unique : bool, optional
        If True, return unique beta values. Default is True.
    sort : bool, optional
        If True, return sorted beta values. Default is True.

    Returns
    -------
    np.ndarray
        Array containing all possible positive beta values for the input parameters.
    """
    kb = 8.617333262145e-5  # Boltzmann constant in eV/K
    betaAbs = np.abs((Eout - Ein) / (kb * T))

    if unique:
        betaAbs = np.unique(betaAbs)
    elif sort:
        betaAbs = np.sort(betaAbs)

    return betaAbs

@nb.jit(nopython=True, nogil=False, cache=True)
def default_absBeta(T: float) -> np.ndarray:
    """
    Generate the default beta grid for a certain temperature

    Parameters
    ----------
    T: float
        Temperature in K

    Returns
    -------
    beta: np.ndarray
        Beta grid
    """
    # Get the first half of the grid:
    betaMid = 0.08 / (kb * T)
    betaSmall = np.linspace(0, betaMid,2000)

    # Get the second half of the grid:
    betaMax = 5.0 / (kb * T)
    beta_great = np.logspace(np.log10(betaMid), np.log10(betaMax), 1000)

    return np.concatenate((betaSmall, beta_great[1::]))


@nb.jit(nopython=True, nogil=False, cache=True)
def default_beta(T: float) -> np.ndarray:
    """
    Generate the default beta grid for a certain temperature

    Parameters
    ----------
    T: float
        Temperature in K

    Returns
    -------
    beta: np.ndarray
        Beta grid
    """
    absBeta = default_absBeta(T)
    return np.concatenate((-absBeta[::-1], absBeta[1::]))
