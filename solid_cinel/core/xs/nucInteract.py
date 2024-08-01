import numpy as np
import pandas as pd
import numba as nb
from scipy.constants import physical_constants as const
from numba import prange
from typing import Iterable
from solid_cinel.core.scattering_function.alpha import get_alphaRecoil


# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]


class InteractTemp:
    """
    Class to calculate the interaction temperature of a material.
    """
    def __init__(self, data: Iterable):
        """
        Initialize the class with the data.
        Parameters
        ----------
        data: Iterable
            The data to be stored in the class
        """
        self._data = data

    @property
    def data(self) -> np.ndarray:
        """
        Get the data stored in the class.

        Returns
        -------
        np.ndarray
            The data stored in the class
        """
        return self._data

    @data.setter
    def data(self, temp: Iterable):
        """
        Set the data stored in the class.

        Parameters
        ----------
        temp: Iterable
            The data to be stored in the class
        """
        self._data = np.array(temp)

    @staticmethod
    def from_approx4PCF(T: float, mu: np.ndarray) -> np.ndarray:
        """
        Approximation of the interaction temperature from the 4PCF model.

        Parameters
        ----------
        T: float
            The temperature of the material in Kelvin
        mu: np.ndarray
            The cosine of the angle between the incident and outgoing particles

        Returns
        -------
        np.ndarray
            The interaction temperature of the material in Kelvin

        Examples
        --------
        >>> T = 300
        >>> mu = np.array([-1.0, -0.5, 0.0, 0.5, 0.75])
        >>> values = InteractTemp.from_approx4PCF(T, mu)
        >>> pd.Series(values, index=mu)
        -1.00      0.0
        -0.50     75.0
         0.00    150.0
         0.50    225.0
         0.75    262.5
        dtype: float64
        """
        return T * (1 + mu) / 2

    @staticmethod
    def from_strict4PCF(T: float, mu: np.ndarray, Ein: float,
                        Eout: np.ndarray) -> np.ndarray:
        """
        Strict calculation of the interaction temperature from the 4PCF model.

        Parameters
        ----------
        T: float
            The temperature of the material in Kelvin
        mu: np.ndarray
            The cosine of the angle between the incident and outgoing particles
        Ein: float
            The energy of the incident particle in eV
        Eout: np.ndarray
            The energy of the outgoing particles in eV

        Returns
        -------
        np.ndarray
            The interaction temperature of the material in Kelvin

        Examples
        --------
        >>> T = 300
        >>> mu = np.array([-1.0, -0.5, 0.0, 0.5, 0.75])
        >>> Ein = 1
        >>> Eout = np.array([0.5, 0.9, 1.0, 1.1, 1.5, 2])
        >>> values = InteractTemp.from_strict4PCF(T, mu, Ein, Eout)
        >>> pd.DataFrame(values, index=mu, columns=Eout)
                      0.5         0.9    1.0         1.1         1.5         2.0
        -1.00    0.000000    0.000000    0.0    0.000000    0.000000    0.000000
        -0.50   50.971707   71.085473   75.0   78.601151   90.610233  101.943414
         0.00  100.000000  142.105263  150.0  157.142857  180.000000  200.000000
         0.50  141.885436  212.862866  225.0  235.447187  264.652925  283.770872
         0.75  149.371843  247.654462  262.5  274.067269  296.998250  298.743687
        """
        if len(mu.shape) == 1:
            mu = mu[:, np.newaxis]
        Tstar = T * (1 - mu ** 2) * Eout
        Tstar /= Ein + Eout - 2 * mu * np.sqrt(Ein * Eout)
        return Tstar

    @classmethod
    def from_4PCF(cls, T: float, theta: np.ndarray, *args,
                  approx: bool = True) -> "InteractTemp":
        """
        Calculate the interaction temperature from the 4PCF model.

        Parameters
        ----------
        T: float
            The temperature of the material in Kelvin
        theta: np.ndarray
            The angle between the incident and outgoing particles in degrees
        approx: bool
            Whether to use the approximation or strict calculation

        Parameters for strict calculation:
        ----------------------------------
        Ein: float
            The energy of the incident particle in eV
        Eout: np.ndarray
            The energy of the outgoing particles in eV

        Returns
        -------
        InteractTemp
            The interaction temperature of the material in Kelvin

        Examples
        --------
        >>> T = 300
        >>> theta = np.array([30, 60, 90, 120, 150])
        >>> values = InteractTemp.from_4PCF(T, theta)
        >>> pd.Series(values.data, index=theta)
        30      20.096189
        60      75.000000
        90     150.000000
        120    225.000000
        150    279.903811
        dtype: float64

        >>> Ein = 1
        >>> Eout = np.array([0.5, 0.9, 1.0, 1.1, 1.5, 2])
        >>> values = InteractTemp.from_4PCF(T, theta, Ein, Eout, approx=False)
        >>> pd.DataFrame(values.data, index=theta[::-1], columns=Eout)
                    0.5         0.9         1.0         1.1         1.5         2.0
        150   13.762756   19.050750   20.096189   21.064241   24.343692   27.525513
        120   50.971707   71.085473   75.000000   78.601151   90.610233  101.943414
        90   100.000000  142.105263  150.000000  157.142857  180.000000  200.000000
        60   141.885436  212.862866  225.000000  235.447187  264.652925  283.770872
        30   136.237244  262.817382  279.903811  291.097921  297.084879  272.474487
        """
        mu = np.sort(np.cos(np.deg2rad(theta)))
        if approx:
            Tinteract = cls.from_approx4PCF(T, mu)
        else:
            Tinteract = cls.from_strict4PCF(T, mu, *args)
        return cls(Tinteract)
class InteractEnergy:
    """
    Class to calculate the interaction energy of a material.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the class with the data.
        Parameters
        ----------
        data: Iterable
            The data to be stored in the class
        """
        self._data = pd.DataFrame(*args, **kwargs)

    @property
    def data(self) -> np.ndarray:
        """
        Get the data stored in the class.

        Returns
        -------
        np.ndarray
            The data stored in the class
        """
        return self._data

    @data.setter
    def data(self, EinMat: Iterable):
        """
        Set the data stored in the class.

        Parameters
        ----------
        temp: Iterable
            The data to be stored in the class
        """
        self._data = pd.DataFrame(EinMat).sort_index(axis=0).sort_index(axis=1)

    @staticmethod
    def original4PCFapprox(Ein: float, Eout: np.ndarray, mu: np.ndarray,
                           M: float) -> np.ndarray:
        """
        Approximation of the interaction energy from the original 4PCF model.

        Parameters
        ----------
        Ein: float
            The energy of the incident particle in eV
        Eout: np.ndarray
            The energy of the outgoing particles in eV
        mu: np.ndarray
            The cosine of the angle between the incident and outgoing particles
        M: float
            The mass of the target nucleus in u

        Returns
        -------
        np.ndarray
            The interaction energy of the material in eV

        Examples
        --------
        >>> Ein = 1
        >>> Eout = np.array([0.5, 0.9, 1.0, 1.1, 1.5, 2])
        >>> mu = np.array([-1.0, -0.5, 0.0, 0.5, 0.75])
        >>> M = 238.05077040419212
        >>> values = InteractEnergy.original4PCFapprox(Ein, Eout, mu, M)
        >>> pd.DataFrame(values, index=mu, columns=Eout)
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.754237  0.954237  1.004237  1.054237  1.254237  1.504237
        -0.50  0.752119  0.952119  1.002119  1.052119  1.252119  1.502119
         0.00  0.750000  0.950000  1.000000  1.050000  1.250000  1.500000
         0.50  0.747881  0.947881  0.997881  1.047881  1.247881  1.497881
         0.75  0.746822  0.946822  0.996822  1.046822  1.246822  1.496822
        """
        if len(mu.shape) == 1:
            mu = mu[:, np.newaxis]
        return (Eout + Ein) / 2 - Ein * mu * m / M

    @staticmethod
    def mod4PCFapprox(Ein: float, Eout: np.ndarray, mu: np.ndarray,
                      M: float) -> np.ndarray:
        """
        Approximation of the interaction energy from the modified 4PCF model.

        Parameters
        ----------
        Ein: float
            The energy of the incident particle in eV
        Eout: np.ndarray
            The energy of the outgoing particles in eV
        mu: np.ndarray
            The cosine of the angle between the incident and outgoing particles
        M: float
            The mass of the target nucleus in u

        Returns
        -------
        np.ndarray
            The interaction energy of the material in eV

        Examples
        --------
        >>> Ein = 1
        >>> Eout = np.array([0.5, 0.9, 1.0, 1.1, 1.5, 2])
        >>> mu = np.array([-1.0, -0.5, 0.0, 0.5, 0.75])
        >>> M = 238.05077040419212
        >>> values = InteractEnergy.mod4PCFapprox(Ein, Eout, mu, M)
        >>> pd.DataFrame(values, index=mu, columns=Eout)
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.757324  0.958260  1.008474  1.058684  1.259480  1.510411
        -0.50  0.755236  0.956142  1.006356  1.056566  1.257379  1.508353
         0.00  0.753178  0.954025  1.004237  1.054449  1.255296  1.506356
         0.50  0.751241  0.951912  1.002119  1.052335  1.253285  1.504601
         0.75  0.750545  0.950864  1.001059  1.051286  1.252440  1.504268
        """
        if len(mu.shape) == 1:
            mu = mu[:, np.newaxis]
        recoilMod = get_alphaRecoil(Eout, Ein, M, mu)
        recoilMod /= 2 * (1 - mu)
        return InteractEnergy.original4PCFapprox(Ein, Eout, mu, M) + recoilMod

    @classmethod
    def from_4PCF(cls, Ein: float, Eout: np.ndarray, mu: np.ndarray,
                  M: float, mod: bool = True, approx: bool = True) -> "InteractEnergy":
        """
        Calculate the interaction energy from the 4PCF model.

        Parameters
        ----------
        Ein: float
            The energy of the incident particle in eV
        Eout: np.ndarray
            The energy of the outgoing particles in eV
        mu: np.ndarray
            The cosine of the angle between the incident and outgoing particles
        M: float
            The mass of the target nucleus in amu
        mod: bool
            Whether to use the modified 4PCF model
        approx: bool
            Whether to use the approximation

        Returns
        -------
        InteractEnergy
            The interaction energy of the material in eV

        Examples
        --------
        >>> Ein = 1
        >>> Eout = np.array([0.5, 0.9, 1.0, 1.1, 1.5, 2])
        >>> mu = np.array([-1.0, -0.5, 0.0, 0.5, 0.75])
        >>> M = 238.05077040419212
        >>> EinMat = InteractEnergy.from_4PCF(Ein, Eout, mu, M)
        >>> EinMat.data
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.757324  0.958260  1.008474  1.058684  1.259480  1.510411
        -0.50  0.755236  0.956142  1.006356  1.056566  1.257379  1.508353
         0.00  0.753178  0.954025  1.004237  1.054449  1.255296  1.506356
         0.50  0.751241  0.951912  1.002119  1.052335  1.253285  1.504601
         0.75  0.750545  0.950864  1.001059  1.051286  1.252440  1.504268

        >>> EinMat = InteractEnergy.from_4PCF(Ein, Eout, mu, M, mod=False)
        >>> EinMat.data
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.754237  0.954237  1.004237  1.054237  1.254237  1.504237
        -0.50  0.752119  0.952119  1.002119  1.052119  1.252119  1.502119
         0.00  0.750000  0.950000  1.000000  1.050000  1.250000  1.500000
         0.50  0.747881  0.947881  0.997881  1.047881  1.247881  1.497881
         0.75  0.746822  0.946822  0.996822  1.046822  1.246822  1.496822
        """
        if approx:
            if mod:
                Einteract = cls.mod4PCFapprox(Ein, Eout, mu, M)
            else:
                Einteract = cls.original4PCFapprox(Ein, Eout, mu, M)
        return cls(Einteract, index=mu, columns=Eout)
class NucInteract:
    pass