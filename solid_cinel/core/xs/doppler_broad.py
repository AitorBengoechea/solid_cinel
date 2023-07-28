import numpy as np
import pandas as pd
import os
from scipy.constants import physical_constants as const
from solid_cinel.core.generic import reshape_differential, integrate
from solid_cinel.core.material.scattering_function.scatfunc import ScatFuncSD

# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]

def get_DB(*args, **kwargs) -> [pd.Series, pd.DataFrame]:
    """
    Calculate the Double differential or singe differential dopper broadened
    cross sections for elastic scattering at a given temperature and incident
    energy using one of the following formalism:
        - sigma1: sigma1 algorithm from NJOY2016 manual

    Parameters for get_DB
    ---------------------
    algorithm : 'str'
        The algorithm to use for the calculation of the dopper broadened elastic
        cross section. The available algorithms are:
            - sigma1: sigma1 algorithm from NJOY2016 manual

    Parameters for sigma1
    ---------------------
    xs_0K : pd.Series
        0K xs data for the given material
    Ein : float
        Incident energy of the neutron
    Eout : np.array
        Outgoing energy grid
    M : float
        Mass of the material in amu
    T : float
        Temperature of the material in K

    Returns
    -------
    pd.Series or pd.DataFrame
        Doppler broadened differential cross section for the given temperature,
        incident energy and mass

    Examples
    --------
    # 0K xs data for U238:
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("doppler_broad.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> xs_0K = pd.read_csv("u238.0.2", sep="    ", header=None, engine="python").set_index(0).drop([2], axis=1)
    >>> os.chdir(wd)
    >>> xs_0K = xs_0K[~xs_0K.index.duplicated(keep='first')]

    # Generate Broadening test results:
    >>> Ein = 36.68723
    >>> Eout = np.linspace(Ein * 0.98 , Ein * 1.02, 1000)
    >>> M = 238.05077040419212
    >>> T = 300

    # SIGMA1:
    >>> algorithm = "sigma1"
    >>> round(get_DB(xs_0K, Ein, M, T, Eout, algorithm=algorithm, integral=True), 2)
    7905.42

    """
    algorithm = kwargs.pop("algorithm")
    if algorithm == "sigma1":
        return sigma1(*args, **kwargs)
    return


def sigma1(xs_0K: pd.Series, Ein: float, M: float, T: float, Eout: np.array,
           integral: bool = False) -> pd.Series:
    """
    Calculate the outgoin energy defferential Doppler broadened cross section at
    a given temperature and for an incident energy and mass using sigma1
    algorithm from NJOY2016 manual:
    .. math::
        \frac{\sigma_T(E)}{dE^\prime} = \frac{1}{2}\sqrt{\frac{M}{m\pi k_BT}}\frac{\sqrt{E^\prime}}{E}\sigma_0(E^\prime)\left(exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} - \sqrt{E^\prime}\right)^2 \right) - exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} + \sqrt{E^\prime}\right)^2 \right)\right)

    Parameters
    ----------
    xs_0K : pd.Series
        0K xs data for the given material
    Ein : float
        Incident energy of the neutron
    Eout : np.array
        Outgoing energy grid
    M : float
        Mass of the material in amu
    T : float
        Temperature of the material in K

    Returns
    -------
    pd.Series
        Outgoing energy differential Doppler broadened cross section for the
        given temperature, incident energy and mass

    Examples
    --------
    # 0K xs data for U238:
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("doppler_broad.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> xs_0K = pd.read_csv("u238.0.2", sep="    ", header=None, engine="python").set_index(0).drop([2], axis=1)
    >>> os.chdir(wd)
    >>> xs_0K = xs_0K[~xs_0K.index.duplicated(keep='first')]

    # Generate Broadening test results:
    >>> Ein = 36.68723
    >>> Eout = np.linspace(Ein * 0.98 , Ein * 1.02, 1000)
    >>> M = 238.05077040419212
    >>> T = 300
    >>> xs_broad = sigma1(xs_0K, Ein, M, T, Eout)

    # Differential value test:
    >>> xs_broad.iloc[::100]
    Eout
    35.953485    7.227742e-14
    36.100381    3.567348e-08
    36.247277    1.191135e-03
    36.394173    3.113927e+00
    36.541069    9.254287e+02
    36.687964    1.079050e+05
    36.834860    1.182981e+03
    36.981756    6.837390e+00
    37.128652    4.644441e-03
    37.275548    2.789231e-07
    Name: 36.68723, dtype: float64

    # Integral value test:
    >>> round(sigma1(xs_0K, Ein, M, T, Eout, integral = True), 2)
    7905.42
    """
    return ScatFuncSD.from_MD(Ein, M, T, Eout).convolve(xs_0K, integral=integral)
