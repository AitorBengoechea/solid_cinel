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

    def XsMat_from_sigma1(self, T: [float, np.ndarray], Ein: [float, np.ndarray] = None,
               values: bool = False) -> [pd.DataFrame, np.ndarray]:
        """
        Calculate the angle-integrated cross section matrix based on SIGMA1 model

        Parameters
        ----------
        T: float, np.ndarray
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
        >>> xs0K.XsMat_from_sigma1(T, EinGrid)
                       300
        2.00      9.086237
        6.67    455.670534

        >>> T = [300, 1000]
        >>> xs0K.XsMat_from_sigma1(T, EinGrid)
                    300         1000
        2.00    9.086237    9.086042
        6.67  455.670534  282.297098

        """
        # Convert the input to arrays
        Tcalc = to_arrays(T)
        EinCalc = to_arrays(Ein) if Ein is not None else self.EinGrid

        # Calculate the Eout grid matrix:
        EoutMatrix = EoutMat(EinCalc)

        # Calculate the Transfer function matrix:
        XsMat = np.zeros((len(EinCalc), len(Tcalc)))
        calc_XsMat_sigma1(Tcalc, EinCalc, EoutMatrix, self.M,
                          self.interpolate(EoutMatrix, values=True),
                          XsMat)
        if values:
            return XsMat
        else:
            return pd.DataFrame(XsMat, index=EinCalc, columns=Tcalc)

    def nuclearInteract_sigma1(self, Tinteract: np.ndarray,
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

        >>> T = np.array([100, 300])
        >>> EinMat = np.array([[2.0, 7.0], [3.0, 6.67]])
        >>> pd.DataFrame(xs0K.nuclearInteract_sigma1(T, EinMat), index=T).round(6)
                    0           1
        100  9.086957   19.893739
        300  8.843855  455.670534
        >>> T = np.array([[500, 1000], [100, 300]])
        >>> pd.DataFrame(xs0K.nuclearInteract_sigma1(T, EinMat)).round(6)
                  0           1
        0  9.086010   20.673028
        1  8.844076  455.670534
        """
        XsMat = np.zeros(EinMat.shape)
        if len(Tinteract.shape) == 2:
            NucInteractStrict_sigma1(Tinteract, EinMat, self.M, self.values,
                                     self.EinGrid, XsMat)
        else:
            NucInteractAprox_sigma1(Tinteract, EinMat, self.M, self.values,
                                    self.EinGrid, XsMat)
        return XsMat


def check_dx(data: [pd.DataFrame, pd.Series],
             dx: [float, np.ndarray, pd.DataFrame],
             axis: [str, int]) -> [float, pd.Series, pd.DataFrame]:
    """
    Check the dx value to shift the Double Scattering function and return the value
    in the correct format for the shift aplicattion.

    Parameters
    ----------
    data : pd.DataFrame, pd.Series
        Double or Single Scattering function data to shift
    dx : float, np.ndarray, pd.DataFrame
        Value to shift the data
    axis : str, int
        Axis to shift the data. Available options are "Eout", "mu" or 0, 1
        respectively.

    Returns
    -------
    float, pd.Series, pd.DataFrame
        Value to shift the data in the correct format

    Examples
    --------
    >>> data = pd.DataFrame([[1, 1, 1]] * 3, index=[-1, 0, 1], columns=[1, 2, 3])
    >>> check_dx(data, 0.1, "Eout")
    0.1

    >>> check_dx(data, pd.Series([1, 1], index=[[0, -1]]), "mu")
     0    1
    -1    1
    dtype: int64

    >>> check_dx(data, [1, 1, 1], "mu")
    -1    1
     0    1
     1    1
    dtype: int64

    >>> check_dx(data, [1, 1, 1], 1)
    1    1
    2    1
    3    1
    dtype: int64

    >>> check_dx(data, [[1, 1, 1]] * 3, 0)
        1	2	3
    -1	1	1	1
    0	1	1	1
    1	1	1	1
    """
    if isinstance(dx, float) or isinstance(dx, int) or isinstance(dx, pd.Series) or isinstance(dx, pd.DataFrame):
        return dx
    elif len(np.array(dx).shape) == 1:
        axis_ = 1 if axis == "Eout" else 0 if axis == "mu" else axis
        return pd.Series(dx, index=data.index if axis_ == 0 else data.columns)
    else:
        return pd.DataFrame(dx, index=data.index, columns=data.columns)


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
def EoutMat(Ein:np.ndarray) -> np.ndarray:
    """
    Generate the Eout matrix for the convolution

    Parameters
    ----------
    Ein: np.ndarray
        Incoming energy grid in eV

    Returns
    -------
    EoutMat: np.ndarray
        Outgoing energy grid matrix in eV

    Examples
    --------
    >>> Ein = np.array([1.0, 2.0])
    >>> pd.DataFrame(EoutMat(Ein)[::, ::2000], index=Ein)
           0         1         2         3
    1.0  0.0  0.990007  1.003344  2.251720
    2.0  0.0  1.980013  2.006689  3.181654
    """
    # Get the length of the incoming energy grid:
    EinLen = len(Ein)

    # 1 iteration to allocate the matrix:
    Eout = default_Eout(Ein[0])
    result = np.zeros((EinLen, len(Eout)))
    result[0, :] += Eout

    # Next iteration to fill the matrix:
    for i in range(1, EinLen):
        result[i, :] += default_Eout(Ein[i])
    return result


@nb.jit(nopython=True, cache=True)
def sigma1(Eout: float, Ein: float, T: float, M: float):
    """
    Sigma1 function for Energy differential Transfer function
    ..math::
           S(E, E^\prime, M, T) = \frac{1}{2}\sqrt{\frac{M}{m\pi k_BT}}\frac{\sqrt{E^\prime}}{E}\left(exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} - \sqrt{E^\prime}\right)^2 \right) - exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} + \sqrt{E^\prime}\right)^2 \right)\right)

    Parameters
    ----------
    Eout : np.array
        Outgoing energy grid in eV
    Ein : float
        Incoming energy in eV
    T : float
        Temperature in K
    M :
        Mass of the target in amu

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
    >>> transferFunc = sigma1(Eout, Ein, T, M)
    >>> pd.Series(transferFunc, index=Eout).round(6)
    6.7554    0.000000
    6.9050    0.001153
    7.0439    0.522804
    7.2000    5.501786
    7.3157    1.568599
    7.4480    0.017808
    dtype: float64

    >>> Ein = np.array([7.0, 7.2])
    >>> transferFunc = sigma1(Eout, Ein[::, np.newaxis], T, M)
    >>> pd.DataFrame(transferFunc, index=Ein, columns=Eout).round(6)
           6.7554    6.9050    7.0439    7.2000    7.3157    7.4480
    7.0  0.014191  2.278534  4.638397  0.119514  0.000412  0.000000
    7.2  0.000000  0.001153  0.522804  5.501786  1.568599  0.017808
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


@nb.jit(nopython=True, parallel=True, cache=True, nogil=True)
def calc_XsMat_sigma1(Tcalc: np.ndarray, Eincalc: np.ndarray, EoutMat: np.ndarray,
                      M: float, xsOkinterp: np.ndarray,
                      XsMat: np.ndarray) -> np.ndarray:
    """
    Calculate angle-integrated cross section matrix based on sigma1 model

    Parameters
    ----------
    Tcalc : np.ndarray, (N,)
        Temperature grid in K
    Eincalc : np.ndarray, (M,)
        Incoming energy grid in eV
    EoutMat : np.ndarray, (M, Z)
        Outgoing energy grid matrix in eV
    M : float
        Mass of the target in amu
    xsOkinterp : np.ndarray, (M, Z)
        Interpolated 0K cross section values
    XsMat : np.ndarray, (M, N)
        Angle-integrated cross section matrix

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
    >>> T = np.array([300, 1000])
    >>> EoutMat = EoutMat(Ein)
    >>> xsOkinterp = xs0K.interpolate(EoutMat, values=True)

    >>> XsMat = np.zeros((len(Ein), len(T)))
    >>> calc_XsMat_sigma1(T, Ein, EoutMat, M, xsOkinterp, XsMat)
    >>> pd.DataFrame(XsMat, index=Ein, columns=T).round(6)
                    300         1000
        2.00    9.086237    9.086042
        6.67  455.670534  282.297098
    """
    Eincalc = Eincalc[::, np.newaxis]
    for j in prange(len(Tcalc)):
        XsMat[:, j] += np.trapz(
            sigma1(EoutMat, Eincalc, Tcalc[j], M) * xsOkinterp, x=EoutMat
        )

@nb.jit(nopython=True, parallel=True, cache=True, nogil=True)
def NucInteractAprox_sigma1(Tcalc: np.ndarray, Eincalc: np.ndarray,
                            M: float, xs0Kvalues: np.ndarray, xs0KEin: np.ndarray,
                            XsMat: np.ndarray):
    """
    Update the angle-integrated cross section matrix based on SIGMA1 model
    for the nuclear interaction in 4PCF model

    Parameters
    ----------
    Tcalc: np.ndarray
        The temperature grid in K
    Eincalc: np.ndarray
        The incoming energy grid in eV
    M: float
        The mass of the target in amu
    xs0Kvalues: np.ndarray
        The values of the cross section
    xs0KEin: np.ndarray
        The incident energy grid
    XsMat: np.ndarray
        The angle-integrated cross section matrix

    Returns
    -------
    void
        The angle-integrated cross section matrix is updated

    Examples
    --------
    # 0K xs data for U238:
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("xs0K.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> M = 238.05077040419212
    >>> xs0K = Xs0K.from_file("u238.0.2", M)
    >>> os.chdir(wd)

    >>> T = np.array([100, 300])
    >>> EinMat = np.array([[2.0, 7.0], [3.0, 6.67]])
    >>> XsMat = np.zeros(EinMat.shape)
    >>> NucInteractAprox_sigma1(T, EinMat, M, xs0K.values, xs0K.EinGrid, XsMat)
    >>> pd.DataFrame(XsMat, index=T).round(6)
                0           1
    100  9.086957   19.893739
    300  8.843855  455.670534
    """
    # 1 iteration to allocate the matrix and modify in place:
    EoutMatrix = EoutMat(Eincalc[0])
    transferFunc = sigma1(EoutMatrix, Eincalc[0][::, np.newaxis], Tcalc[0], M)
    transferFunc *= np.interp(EoutMatrix, xs0KEin, xs0Kvalues)
    XsMat[0] += np.trapz(transferFunc, x=EoutMatrix)

    # Next interactions with the same Eout grid:
    for i in range(1, XsMat.shape[0]):
        EoutMatrix[:] = EoutMat(Eincalc[i])
        transferFunc[:] = sigma1(EoutMatrix, Eincalc[i][::, np.newaxis], Tcalc[i], M)
        transferFunc *= np.interp(EoutMatrix, xs0KEin, xs0Kvalues)
        XsMat[i] += np.trapz(transferFunc, x=EoutMatrix)


@nb.jit(nopython=True, parallel=True, cache=True, nogil=True)
def NucInteractStrict_sigma1(Tcalc: np.ndarray, Eincalc: np.ndarray,
                             M: float, xs0Kvalues: np.ndarray, xs0KEin: np.ndarray,
                             XsMat: np.ndarray):
    """
    Update the angle-integrated cross section matrix based on SIGMA1 model
    for the nuclear interaction in 4PCF model with strict calculation

    Parameters
    ----------
    Tcalc: np.ndarray
        The temperature grid in K
    Eincalc: np.ndarray
        The incoming energy grid in eV
    M: float
        The mass of the target in amu
    xs0Kvalues: np.ndarray
        The values of the cross section
    xs0KEin: np.ndarray
        The incident energy grid
    XsMat: np.ndarray
        The angle-integrated cross section matrix

    Returns
    -------
    void
        The angle-integrated cross section matrix is updated

    Examples
    --------
    # 0K xs data for U238:
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("xs0K.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> M = 238.05077040419212
    >>> xs0K = Xs0K.from_file("u238.0.2", M)
    >>> os.chdir(wd)

    >>> T = np.array([[500, 1000], [100, 300]])
    >>> EinMat = np.array([[2.0, 7.0], [3.0, 6.67]])
    >>> XsMat = np.zeros(EinMat.shape)
    >>> NucInteractStrict_sigma1(T, EinMat, M, xs0K.values, xs0K.EinGrid, XsMat)
    >>> pd.DataFrame(XsMat).round(6)
              0           1
    0  9.086010   20.673028
    1  8.844076  455.670534
    """
    for i in prange(XsMat.shape[0]):
        for j in range(XsMat.shape[1]):
            Eout = default_Eout(Eincalc[i, j])
            transferFunc = sigma1(Eout, Eincalc[i, j], Tcalc[i, j], M)
            transferFunc *= np.interp(Eout, xs0KEin, xs0Kvalues)
            XsMat[i, j] += np.trapz(transferFunc, x=Eout)

