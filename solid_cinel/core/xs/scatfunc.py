"""
Python for working with Diferential XS.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from scipy.constants import physical_constants as const

from solid_cinel import Xs, sigma1
from solid_cinel.core.scattering_function import DynamicStruc
from solid_cinel.core.scattering_function.alpha import get_alpha
from solid_cinel.core.generic import integrate, reshift, interpolation, reshape_differential
import os
from typing import Iterable

from solid_cinel.core.xs import default_Eout

# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]

# Avoid numba fast math:
nb.config.FASTMATH_DEFAULT = False

class Xs0K:
    def __init__(self, M:float, *args, **kwargs):
        self.M = M
        self.data = pd.Series(*args, **kwargs)

    @property
    def data(self) -> pd.Series:
        return self._data

    @data.setter
    def data(self, xs: pd.Series):
        # Sort the Series by index
        xs_ = pd.Series(xs, name="0").sort_index()
        xs_.index.name = "Ein"

        # Drop duplicates, keeping the first occurrence
        self._data = xs_.drop_duplicates(keep='first')

    @property
    def values(self) -> np.ndarray:
        return self.data.values
    @property
    def EinGrid(self) -> np.ndarray:
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
    def from_file(cls, filename: str, M: float, **kwargs):
        return cls(M, cls.read_xs(filename, **kwargs))

    def interpolate(self, EinSmall: [float, np.array], inplace: bool = False,
                    values: bool = False) -> ["Xs0K", pd.Series, np.ndarray]:
        """

        Parameters
        ----------
        EinSmall :
        inplace :
        values :

        Returns
        -------
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("scatfunc.py", ""))
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
        if values and inplace:
            raise ValueError("Values and inplace cannot be True at the same time")
        EinSmall = np.array(EinSmall) if hasattr(EinSmall, "__len__") else np.array([EinSmall])
        if values:
            return reshape_differential(self.data, EinSmall)
        else:
            xs0Kinterp = interpolation(self.data, EinSmall, values=values)
            return self.update(xs0Kinterp) if inplace else xs0Kinterp

    def update(self, dataNew: pd.Series) -> "Xs0K":
            self.data = dataNew
            return self


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

    Examples
    --------
    Test the default Eout with NJOY values:
    # 0K xs data for U238:
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("xs.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> xs0K = Xs.read_xs("u238.0.2")
    >>> os.chdir(wd)

    # Generate Broadening test results:
    >>> T = 1000
    >>> Ein = 2.0
    >>> Eout = default_Eout(Ein)
    >>> M = 238.05077040419212
    >>> float(round(ScatFunc.from_sigma1(xs0K, Ein, M, T, Eout).integral, 2))
    9.09
    """
    EoutSmall = np.linspace(0,
                             0.99 * Ein,
                             2000)
    EoutMid = np.linspace(0.99 * Ein,
                          Ein * 1.01,
                              3000)
    if Ein * 2 < 5.0:
        EoutGreat = np.logspace(np.log10(Ein * 1.01),
                                np.log10(5.0),
                                 2000)
    else:
        EoutGreat = np.logspace(np.log10(Ein * 1.01),
                                 np.log10(2 * Ein),
                                 2000)
    return np.unique(np.concatenate((EoutGreat, EoutSmall, EoutMid)))


@nb.jit(nopython=True, parallel=True, cache=True, nogil=True)
def XsMat_sigma1(Ein: float, T: float, M: float, xs0Kvalues: np.ndarray,
                 xs0KEinGrid: np.ndarray, XsMat: np.ndarray) -> np.ndarray:
    """

    Parameters
    ----------
    Ein :
    T :
    M :
    xs0Kvalues :
    xs0KEinGrid :

    Returns
    -------
    # 0K xs data for U238:
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("xs.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> xs0K = Xs0K.read_xs("u238.0.2")
    >>> os.chdir(wd)

    >>> Ein = np.array([1.0, 2.0])
    >>> T = np.array([300, 100])
    >>> M = 238.05077040419212

    >>> XsMat_sigma1(Ein, T, M, xs0K.values, xs0K.index)
    array([[9.27057255, 9.27182968],
           [9.08623706, 9.08695736]])
    """
    for i in prange(XsMat.shape[0]):
        for j in range(XsMat.shape[1]):
            Eout = default_Eout(Ein[i])
            scatFunc = sigma1(Eout, Ein[i], T[j], M)
            scatFunc *= np.interp(Eout, xs0KEinGrid, xs0Kvalues)
            XsMat[i, j] += np.trapz(scatFunc, Eout)
