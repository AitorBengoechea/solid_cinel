"""
Python for working with Diferential XS.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from scipy.constants import physical_constants as const
from solid_cinel.core.generic import interpolation, reshape_differential, to_arrays
import os
import dask.bag as db
from math import pi, log10

# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]

# Avoid numba fast math:
nb.config.FASTMATH_DEFAULT = False


class Xs0K:
    def __init__(self, M:float, *args, **kwargs):
        """
        Initialize the Xs0K class

        Parameters
        ----------
        M: float
            The mass of the target in amu
        args: list
            The values of the cross section
        kwargs:
            The keyword arguments of the cross section
        """
        self.M = M
        self.data = pd.Series(*args, **kwargs)

    @property
    def data(self) -> pd.Series:
        """
        Get the data of the cross section

        Returns
        -------
        pd.Series
            The data of the cross section
        """
        return self._data

    @data.setter
    def data(self, xs: pd.Series):
        """
        Set the data of the cross section

        Parameters
        ----------
        xs: pd.Series
            The data of the cross section
        """
        # Sort the Series by index
        xs_ = pd.Series(xs, name="0").sort_index()
        xs_.index.name = "Ein"

        # Drop duplicates, keeping the first occurrence
        self._data = xs_.drop_duplicates(keep='first')

    @property
    def values(self) -> np.ndarray:
        """
        Get the values of the cross section

        Returns
        -------
        np.ndarray
            The values of the cross section
        """
        return self.data.values

    @property
    def EinGrid(self) -> np.ndarray:
        """
        Get the incident energy grid

        Returns
        -------
        np.ndarray
            The incident energy grid
        """
        return self.data.index.values

    @staticmethod
    def read_xs(filename: str, header: [int, list] = None,
                usecols: [int, list] = [0, 1], index_col: int = 0,
                engine: str = "python") -> pd.Series:
        """
        Read the xs data from a file

        Parameters
        ----------
        filename: str
            The filename of the xs data
        header: int, list, optional
            The header of the file. The default is None, so no header is used.
        usecols: int, list, optional
            The columns to use. The default is [0, 1], so the first two columns
            are used.
        index_col: int, optional
            The index column. The default is 0, so the first column is used as
            index.
        engine: str, optional
            The engine to use. The default is "python".

        Returns
        -------
        pd.Series
            The xs data
        """
        # Read the data from the file into a pandas DataFrame
        xsData = pd.read_csv(filename, sep='\s+', header=header, index_col=index_col,
                             usecols=usecols, engine=engine).squeeze("columns")

        xsData.index.name = "E"
        # Ensure not duplicated index and if they are duplicated, take the first
        xsData = xsData.reset_index().drop_duplicates(subset='E', keep='first')

        return xsData.set_index('E').squeeze("columns")

    @classmethod
    def from_file(cls, filename: str, M: float, **kwargs) -> "Xs0K":
        """
        Create an instance of the Xs0K class from a file

        Parameters
        ----------
        filename: str
            The filename of the xs data
        M: float
            The mass of the target in amu
        kwargs: dict
            The keyword arguments of the read_xs method

        Returns
        -------
        Xs0K
            The instance of the Xs0K class
        """
        return cls(M, cls.read_xs(filename, **kwargs))

    def interpolate(self, EinSmall: [float, np.array], inplace: bool = False,
                    values: bool = False) -> ["Xs0K", pd.Series, np.ndarray]:
        """
        Interpolate the cross section data

        Parameters
        ----------
        EinSmall : float, np.array
            The incident energy grid to interpolate
        inplace : bool, optional
            If True, update the cross section data inplace. The default is False.
        values : bool, optional
            If True, return the values of the cross section. The default is False.

        Returns
        -------
        Xs0K, pd.Series, np.ndarray
            The instance of the Xs0K class, the interpolated cross section data or
            the values of the cross section

        Examples
        --------
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs0K.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs0K = Xs0K.from_file("u238.0.2", M)
        >>> os.chdir(wd)

        >>> xs0K.interpolate(1., values=False)
        1.0    9.269424
        dtype: float64

        >>> xs0K.interpolate([1.0, 2.0], values=False)
        1.0    9.269424
        2.0    9.085342
        dtype: float64
        """
        # Check if the values and inplace are True at the same time
        if values and inplace:
            raise ValueError("Values and inplace cannot be True at the same time")

        # Convert the input to arrays
        EinSmall = to_arrays(EinSmall)

        # Interpolate the data
        if values:
            return reshape_differential(self.data, EinSmall)
        else:
            xs0Kinterp = interpolation(self.data, EinSmall)
            return self.update(xs0Kinterp) if inplace else xs0Kinterp

    def update(self, dataNew: pd.Series) -> "Xs0K":
        """
        Update the cross section data

        Parameters
        ----------
        dataNew: pd.Series
            The new cross section data

        Returns
        -------
        Xs0K
            The instance of the Xs0K class
        """
        self.data = dataNew
        return self

    def sigma1(self, T: [float, np.ndarray], Ein: [float, np.ndarray] = None,
               values: bool = False) -> [pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Calculate the angle-integrated cross section matrix based on SIGMA1 model

        Parameters
        ----------
        T: float, np.ndarray (N,)
            The temperature grid in K
        Ein: float, np.ndarray
            The incoming energy grid in eV
        values: bool
            If True, return the values of the cross section. The default is False.

        Returns
        -------
        pd.DataFrame, np.ndarray
            The angle-integrated cross section matrix

        Examples
        --------
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs0K.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs0K = Xs0K.from_file("u238.0.2", M)
        >>> os.chdir(wd)

        >>> T = 300
        >>> EinGrid = [2.0, 6.67]
        >>> xs0K.sigma1(T, EinGrid)
        2.00      9.086237
        6.67    455.670534
        Name: 300, dtype: float64

        >>> T = [300, 1000]
        >>> xs0K.sigma1(T, EinGrid)
                  2.00        6.67
        300   9.086237  455.670534
        1000  9.086042  282.297098

        >>> T = np.array([100, 300])
        >>> EinMat = np.array([2.0, 3.0, 6.67, 7.0])
        >>> xs0K.sigma1(T, EinMat).round(6)
                 2.00      3.00        6.67       7.00
        100  9.086957  8.844076  664.556512  19.893739
        300  9.086237  8.843855  455.670534  20.039076
        """
        # Check if T and Ein:
        T_ = to_arrays(T)
        Ein_ = self.EinGrid if Ein is None else to_arrays(Ein)

        # Calculate the angle-integrated cross section matrix:
        xsDb = sigma1_XsMat(T_, Ein_, self.M, self.EinGrid, self.values)

        if values:
            return xsDb
        elif len(T_) == 1:
            return pd.Series(xsDb.squeeze(), index=Ein_, name=T)
        else:
            return pd.DataFrame(xsDb, index=T_, columns=Ein_)

    def nuclearInteract(self, Tinteract: np.ndarray,
                               EinMat: np.ndarray) -> np.ndarray:
        """
        Calculate the angle-integrated cross section matrix based on SIGMA1 model
        for the nuclear interaction in 4PCF model

        Parameters
        ----------
        Tinteract: np.ndarray
            The temperature grid in K
        EinMat: np.ndarray
            The incoming energy grid in eV

        Returns
        -------
        np.ndarray
            The angle-integrated cross section matrix

        Examples
        --------
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs0K.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs0K = Xs0K.from_file("u238.0.2", M)
        >>> os.chdir(wd)

        # Example with a single temperature and multiple incident energies:
        >>> T = np.array([100, 300])
        >>> EinMat = np.array([[2.0, 7.0], [3.0, 6.67]])
        >>> pd.DataFrame(xs0K.nuclearInteract(T, EinMat), index=T).round(6)
                    0           1
        100  9.086957   19.893739
        300  8.843855  455.670534

        # Example with multiple temperatures and incident energies:
        >>> T = np.array([[500, 1000], [100, 300]])
        >>> pd.DataFrame(xs0K.nuclearInteract(T, EinMat)).round(6)
                  0           1
        0  9.086010   20.673028
        1  8.844076  455.670534
        """
        # Check the inputs:
        Tinteract = to_arrays(Tinteract)
        EinMat = to_arrays(EinMat)

        # Check the shape of Tinteract:
        if len(Tinteract.shape) == 2:
            return sigma1_NucInteract_Strict(Tinteract, EinMat, self.M, self.EinGrid, self.values)
        else:
            return sigma1_NucInteract_Aprox(Tinteract, EinMat, self.M, self.EinGrid, self.values)


@nb.jit(nopython=True, cache=True)
def default_Eout(Ein: float) -> np.ndarray:
    """
    Generate the default Eout grid for the convolution. The grid is tested with
    NJOY values to ensure a relative difference smaller than 0.4%

    Parameters
    ----------
    Ein : float
        Incident energy in eV

    Returns
    -------
    Eout : ndarray
        Outgoing energy grid in eV
    """
    # Define the constant values:
    EinSmall = 0.99 * Ein
    EinMid = 1.01 * Ein
    EinGreat = log10(5.0) if Ein < 2.5 else np.log10(2 * Ein)

    # Define the small Eout grid:
    EoutSmall = np.linspace(0, EinSmall, 2000)

    # Define the mid Eout grid:
    EoutMid = np.linspace(EinSmall, EinMid, 3000)

    # Define the upscattering Eout grid:
    EoutGreat = np.logspace(log10(EinMid), EinGreat, 2000)

    # Return the unique Eout grid:
    return np.unique(np.concatenate((EoutGreat[1:], EoutSmall, EoutMid)))


@nb.jit(nopython=True, cache=True)
def transferFunc_sigma1(T: float, Ein: float, M: float, Eout: float):
    """
    Sigma1 function for Energy differential Transfer function
    ..math::
           S(E, E^\prime, M, T) = \frac{1}{2}\sqrt{\frac{M}{m\pi k_BT}}\frac{\sqrt{E^\prime}}{E}\left(exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} - \sqrt{E^\prime}\right)^2 \right) - exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} + \sqrt{E^\prime}\right)^2 \right)\right)

    Parameters
    ----------
    T : float
        Temperature in K
    Ein : float
        Incoming energy in eV
    M :
        Mass of the target in amu
    Eout : np.array
        Outgoing energy grid in eV

    Returns
    -------
    scatfunc : np.array
        Transfer function based on sigma1 model

    Examples
    --------
    >>> Ein = 7.2
    >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> transferFunc = transferFunc_sigma1(T, Ein, M, Eout)
    >>> pd.Series(transferFunc, index=Eout).round(6)
    6.7554    0.000000
    6.9050    0.001153
    7.0439    0.522804
    7.2000    5.501786
    7.3157    1.568599
    7.4480    0.017808
    dtype: float64
    """
    # Define the constants:
    AkbT = M / (m * kb * T)
    EinSqrt = np.sqrt(Ein)
    EoutSqrt = np.sqrt(Eout)

    # Get the negative exponential part:
    exponetials = np.exp(- AkbT * (EinSqrt - EoutSqrt) ** 2)

    # Get the positive exponential part:
    exponetials -= np.exp(- AkbT * (EinSqrt + EoutSqrt) ** 2)

    # Calculate the Transfer function:
    return 0.5 * exponetials * np.sqrt(AkbT / pi) * EoutSqrt / Ein


@nb.jit(nopython=True, cache=True)
def calc_sigma1(T: float, Ein: float, M: float,  xs0KEin: np.ndarray,
                xs0Kvalues: np.ndarray) -> float:
    """
    Calculate the angle-integrated cross section based on sigma1 model
    Parameters
    ----------
    T : float
        Temperature in K
    Ein : float
        Incoming energy in eV
    M : float
        Mass of the target in amu
    xs0KEin : np.ndarray
        Incident energy grid in eV
    xs0Kvalues : np.ndarray
        Values of the cross section

    Returns
    -------
    float
        Angle-integrated cross section in barns

    Examples
    --------
    # 0K xs data for U238:
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("xs0K.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> M = 238.05077040419212
    >>> xs0K = Xs0K.from_file("u238.0.2", M)
    >>> os.chdir(wd)

    >>> Ein = 6.67
    >>> T = 300
    >>> xsdb = calc_sigma1(T, Ein, M, xs0K.EinGrid, xs0K.values)
    >>> assert round(xsdb, 2) == 455.67
    """
    Eout = default_Eout(Ein)
    transferFunc = transferFunc_sigma1(T, Ein, M, Eout)
    transferFunc *= np.interp(Eout, xs0KEin, xs0Kvalues)
    return np.trapz(transferFunc, x=Eout)

@nb.jit(nopython=True, parallel=True, cache=True, nogil=True)
def sigma1_XsMat(Tcalc: np.ndarray, Eincalc: np.ndarray,
                      M: float,  xs0KEin: np.ndarray,
                      xs0Kvalues: np.ndarray) -> np.ndarray:
    """
    Calculate angle-integrated cross section matrix based on sigma1 model

    Parameters
    ----------
    Tcalc : np.ndarray, (N,)
        Temperature grid in K
    Eincalc : np.ndarray, (M,)
        Incoming energy grid in eV
    M : float
        Mass of the target in amu
    xs0KEin : np.ndarray, (Z,)
        Incident energy grid in eV
    xs0Kvalues : np.ndarray, (Z,)
        Values of the cross section

    Returns
    -------
    XsMat : np.ndarray, (N, M)
        Angle-integrated cross section matrix, where N is the number of temperatures
        and M is the number of incident energies.

    Examples
    -------
    # 0K xs data for U238:
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("xs0K.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> M = 238.05077040419212
    >>> xs0K = Xs0K.from_file("u238.0.2", M)
    >>> os.chdir(wd)

    >>> Ein = np.array([2.0, 6.67])
    >>> T = np.array([0, 300, 1000])

    >>> XsMat = sigma1_XsMat(T, Ein, M, xs0K.EinGrid, xs0K.values)
    >>> pd.DataFrame(XsMat, index=T, columns=Ein).round(6)
              2.00         6.67
    0     9.085342  1269.792131
    300   9.086237   455.670534
    1000  9.086042   282.297098
    """
    # Create the output matrix:
    rows, cols = len(Tcalc), len(Eincalc)
    XsMat = np.zeros((rows, cols))

    # Calculate the angle-integrated cross section for each combination of
    # T and Einc
    for i in prange(rows):
        if Tcalc[i] == 0.0:
            # If the temperature is 0K, use the 0K cross section values
            XsMat[i, :] = np.interp(Eincalc, xs0KEin, xs0Kvalues)
        else:
            for j in prange(cols):
                XsMat[i, j] = calc_sigma1(Tcalc[i], Eincalc[j],
                                          M, xs0KEin, xs0Kvalues)

    # Return the angle-integrated cross section matrix:
    return XsMat


@nb.jit(nopython=True, parallel=True, cache=True, nogil=True)
def sigma1_NucInteract_Aprox(Tcalc: np.ndarray, Eincalc: np.ndarray,
                             M: float, xs0KEin: np.ndarray,
                             xs0Kvalues: np.ndarray) -> np.ndarray:
    """
    Update the angle-integrated cross section matrix based on SIGMA1 model
    for the nuclear interaction in 4PCF model

    Parameters
    ----------
    Tcalc: np.ndarray, (N,)
        The temperature grid in K
    Eincalc: np.ndarray, (N, M)
        The incoming energy grid in eV
    M: float
        The mass of the target in amu
    xs0Kvalues: np.ndarray, (Z,)
        The values of the cross section
    xs0KEin: np.ndarray, (Z,)
        The incident energy grid

    Returns
    -------
    XsMat : np.ndarray, (N, M)
        Angle-integrated cross section matrix.

    Examples
    --------
    # 0K xs data for U238:
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("xs0K.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> M = 238.05077040419212
    >>> xs0K = Xs0K.from_file("u238.0.2", M)
    >>> os.chdir(wd)

    # Example with a single temperature and multiple incident energies:
    >>> T = np.array([0.0, 100, 300])
    >>> EinMat = np.array([[5.0, 10.0], [2.0, 3.0], [6.67, 7.0]])

    # Calculate the angle-integrated cross section matrix:
    >>> XsMat = sigma1_NucInteract_Aprox(T, EinMat, M, xs0K.EinGrid, xs0K.values)
    >>> pd.DataFrame(XsMat, index=T).round(6)
                    0          1
    0.0      7.805580   9.681257
    100.0    9.086957   8.844076
    300.0  455.670534  20.039076
    """
    # Allocate the output matrix:
    rows, cols = len(Tcalc), Eincalc.shape[1]
    XsMat = np.zeros((rows, cols))

    # Calculate the angle-integrated cross section for each combination of
    # T and Einc
    for i in prange(rows):
        if Tcalc[i] == 0.0:
            # If the temperature is 0K, use the 0K cross section values
            XsMat[i, :] = np.interp(Eincalc[i, :], xs0KEin, xs0Kvalues)
        else:
            for j in prange(cols):
                XsMat[i, j] = calc_sigma1(Tcalc[i], Eincalc[i, j],
                                          M, xs0KEin, xs0Kvalues)

    # Return the angle-integrated cross section matrix:
    return XsMat


@nb.jit(nopython=True, parallel=True, cache=True, nogil=True)
def sigma1_NucInteract_Strict(Tcalc: np.ndarray, Eincalc: np.ndarray,
                             M: float, xs0KEin: np.ndarray,
                            xs0Kvalues: np.ndarray) -> np.ndarray:
    """
    Update the angle-integrated cross section matrix based on SIGMA1 model
    for the nuclear interaction in 4PCF model with strict calculation

    Parameters
    ----------
    Tcalc: np.ndarray, (N, M)
        The temperature grid in K
    Eincalc: np.ndarray, (N, M)
        The incoming energy grid in eV
    M: float
        The mass of the target in amu
    xs0Kvalues: np.ndarray, (Z,)
        The values of the cross section
    xs0KEin: np.ndarray, (Z,)
        The incident energy grid

    Returns
    -------
    XsMat : np.ndarray, (N, M)
        Angle-integrated cross section matrix.

    Examples
    --------
    # 0K xs data for U238:
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("xs0K.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> M = 238.05077040419212
    >>> xs0K = Xs0K.from_file("u238.0.2", M)
    >>> os.chdir(wd)

    # Example with a multiple temperatures and incident energies:
    >>> T = np.array([[500, 1000], [100, 300]])
    >>> EinMat = np.array([[2.0, 7.0], [3.0, 6.67]])

    # Calculate the angle-integrated cross section matrix:
    >>> XsMat = sigma1_NucInteract_Strict(T, EinMat, M, xs0K.EinGrid, xs0K.values)
    >>> pd.DataFrame(XsMat).round(6)
              0           1
    0  9.086010   20.673028
    1  8.844076  455.670534
    """
    # Allocate the output matrix:
    rows, cols = Eincalc.shape
    XsMat = np.zeros((rows, cols))

    # Calculate the angle-integrated cross section for each combination of
    # T and Einc
    for i in prange(rows):
        for j in prange(cols):
            if Tcalc[i, j] == 0.0:
                # If the temperature is 0K, use the 0K cross section values
                XsMat[i, j] = np.interp(Eincalc[i, j], xs0KEin, xs0Kvalues)
            else:
                XsMat[i, j] = calc_sigma1(Tcalc[i, j], Eincalc[i, j],
                                          M, xs0KEin, xs0Kvalues)

    # Return the angle-integrated cross section matrix:
    return XsMat

