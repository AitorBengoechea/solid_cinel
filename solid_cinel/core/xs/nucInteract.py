import numpy as np
import pandas as pd
import numba as nb
import os
import dask.bag as db
from scipy.constants import physical_constants as const
from typing import Iterable
from solid_cinel.core.scattering_function.alpha import get_alphaRecoil
from solid_cinel.core.xs.xs import Xs
from solid_cinel.core.generic import interpolation
from solid_cinel.core.xs.scatfunc import ScatFunc
from solid_cinel.core.scattering_function.dynamicStruc import DynamicStruc


# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]

# Modified 4PCF model decorator:
def apply_mod(func):
    """
    Decorator to apply the modification factor to the interaction energy of the
    4PCF model.
    Parameters
    ----------
    func: function
        4PCF model energy interaction function

    Returns
    -------
    function
        The 4PCF model energy interaction function with the modification factor
    """
    def wrapper(Ein: float, Eout: np.ndarray, mu: np.ndarray, M: float) -> np.ndarray:
        mu_ = InteractEnergy.check_mu(mu)
        recoilMod = InteractEnergy.get_4PCFmod(Ein, Eout, mu_, M)
        EinMat = func(Ein, Eout, mu_, M)
        return EinMat + recoilMod
    return wrapper


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
        mu_ = InteractEnergy.check_mu(mu)
        Tstar = T * (1 - mu_ ** 2) * Eout
        Tstar /= Ein + Eout - 2 * mu_ * np.sqrt(Ein * Eout)
        return Tstar

    @classmethod
    def from_4PCF(cls, T: float, mu: np.ndarray, *args,
                  approx: bool = True) -> "InteractTemp":
        """
        Calculate the interaction temperature from the 4PCF model.

        Parameters
        ----------
        T: float
            The temperature of the material in Kelvin
        mu: np.ndarray
            The cosine between the incident and outgoing particles
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
        >>> mu = np.sort(np.cos(np.deg2rad(theta)))
        >>> values = InteractTemp.from_4PCF(T, mu)
        >>> pd.Series(values.data, index=theta[::-1])
        150     20.096189
        120     75.000000
        90     150.000000
        60     225.000000
        30     279.903811
        dtype: float64

        >>> Ein = 1
        >>> Eout = np.array([0.5, 0.9, 1.0, 1.1, 1.5, 2])
        >>> values = InteractTemp.from_4PCF(T, mu, Ein, Eout, approx=False)
        >>> pd.DataFrame(values.data, index=theta[::-1], columns=Eout)
                    0.5         0.9         1.0         1.1         1.5         2.0
        150   13.762756   19.050750   20.096189   21.064241   24.343692   27.525513
        120   50.971707   71.085473   75.000000   78.601151   90.610233  101.943414
        90   100.000000  142.105263  150.000000  157.142857  180.000000  200.000000
        60   141.885436  212.862866  225.000000  235.447187  264.652925  283.770872
        30   136.237244  262.817382  279.903811  291.097921  297.084879  272.474487
        """
        if approx:
            Tinteraction = cls.from_approx4PCF(T, mu)
        else:
            Tinteraction = cls.from_strict4PCF(T, mu, *args)
        return cls(Tinteraction)


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
    def check_mu(mu: np.ndarray) -> np.ndarray:
        """
        Check the cosine of the angle between the incident and outgoing particles.

        Parameters
        ----------
        mu: np.ndarray
            The cosine of the angle between the incident and outgoing particles

        Returns
        -------
        np.ndarray
            The cosine of the angle between the incident and outgoing particles

        Examples
        --------
        >>> mu = np.array([-1.0, -0.5, 0.0, 0.5, 0.75])
        >>> InteractEnergy.check_mu(mu)
        array([[-1.  ],
               [-0.5 ],
               [ 0.  ],
               [ 0.5 ],
               [ 0.75]])
        """
        mu_ = np.array(mu) if isinstance(mu, Iterable) else np.array([mu])
        if len(mu.shape) == 1:
            mu_ = mu_[:, np.newaxis]
        return mu_

    @staticmethod
    def get_4PCFmod(Ein: float, Eout: np.ndarray, mu: np.ndarray,
                    M: float) -> np.ndarray:
        """
        Get the modification factor for the interaction energy of the 4PCF model.

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

        Returns
        -------
        np.ndarray
            The modification factor for the interaction energy
        """
        mu_ = InteractEnergy.check_mu(mu)
        recoilMod = get_alphaRecoil(Eout, Ein, M, mu_)
        recoilMod /= 2 * (1 - mu_)
        return recoilMod

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
            The mass of the target nucleus in amu

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
        mu_ = InteractEnergy.check_mu(mu)
        return (Eout + Ein) / 2 - Ein * mu_ * m / M

    @staticmethod
    def original4PCFstrict(Ein: float, Eout: np.ndarray, mu: np.ndarray,
                           M: float) -> np.ndarray:
        """
        Strict calculation of the interaction energy from the original 4PCF
        model.

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
        >>> values = InteractEnergy.original4PCFstrict(Ein, Eout, mu, M)
        >>> pd.DataFrame(values, index=mu, columns=Eout)
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.708592  0.950676  1.002100  1.051012  1.227317  1.417184
        -0.50  0.694107  0.949241  1.001050  1.049514  1.217727  1.388215
         0.00  0.666667  0.947368  1.000000  1.047619  1.200000  1.333333
         0.50  0.591607  0.943748  0.998950  1.044142  1.150694  1.183214
         0.75  0.464368  0.938023  0.998425  1.038856  1.059500  0.928737
        """
        mu_ = InteractEnergy.check_mu(mu)
        Esqrt = np.sqrt(Ein * Eout)
        Ediff = Eout - Ein
        EinMat = Ein - mu_ * Esqrt / (2 * M)
        EinMat -= Ediff * (mu_ * Esqrt - Ein) / (Ein + Eout - 2 * mu_ * Esqrt)
        return EinMat

    @staticmethod
    @apply_mod
    def mod4PCFapprox(Ein: float, Eout: np.ndarray, mu: np.ndarray,
                      M: float) -> np.ndarray:
        """
        Approximation of the interaction energy of the 4PCF model with the
        modification.

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
        return InteractEnergy.original4PCFapprox(Ein, Eout, mu, M)

    @staticmethod
    @apply_mod
    def mod4PCFstrict(Ein: float, Eout: np.ndarray, mu: np.ndarray,
                      M: float) -> np.ndarray:
        """
        Strict calculation of the interaction energy of the 4PCF model with the
        modification.

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
        >>> values = InteractEnergy.mod4PCFstrict(Ein, Eout, mu, M)
        >>> pd.DataFrame(values, index=mu, columns=Eout)
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.711679  0.954698  1.006338  1.055458  1.232560  1.423358
        -0.50  0.697225  0.953265  1.005287  1.053961  1.222988  1.394449
         0.00  0.669845  0.951394  1.004237  1.052068  1.205296  1.339689
         0.50  0.594967  0.947779  1.003187  1.048596  1.156098  1.189933
         0.75  0.468091  0.942065  1.002662  1.043320  1.065118  0.936183
        """
        return InteractEnergy.original4PCFstrict(Ein, Eout, mu, M)

    @staticmethod
    def correct_Eout(Eout: np.ndarray, Ein: np.ndarray, M: float,
                     mu: np.ndarray) -> np.ndarray:
        """
        Correct the energy of the outgoing particles by adding the recoil energy.

        Parameters
        ----------
        Eout: np.ndarray
            The energy of the outgoing particles in eV
        Ein: np.ndarray
            The energy of the incident particles in eV
        M: float
            The mass of the target nucleus in amu
        mu: np.ndarray
            The cosine of the angle between the incident and outgoing particles

        Returns
        -------
        np.ndarray
            The corrected energy of the outgoing particles in eV

        Examples
        --------
        >>> Ein = 1
        >>> Eout = np.array([0.5, 0.9, 1.0, 1.1, 1.5, 2])
        >>> mu = np.array([-1.0, -0.5, 0.0, 0.5, 0.75])
        >>> M = 238.05077040419212
        >>> values = InteractEnergy.correct_Eout(Eout, Ein, M, mu)
        >>> pd.DataFrame(values, index=mu, columns=Eout)
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.512348  0.916090  1.016949  1.117786  1.520972  2.024696
        -0.50  0.509352  0.912070  1.012712  1.113342  1.515782  2.018704
         0.00  0.506356  0.908051  1.008474  1.108898  1.510593  2.012712
         0.50  0.503360  0.904031  1.004237  1.104454  1.505403  2.006719
         0.75  0.501862  0.902021  1.002119  1.102232  1.502809  2.003723
        """
        mu_ = InteractEnergy.check_mu(mu)
        return Eout + get_alphaRecoil(Eout, Ein, M, mu_)

    @staticmethod
    def corr4PCFapprox(Ein: float, Eout: np.ndarray, mu: np.ndarray,
                       M: float) -> np.ndarray:
        """
        Approximation of the interaction energy from the corrected 4PCF model.

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
        >>> values = InteractEnergy.corr4PCFapprox(Ein, Eout, mu, M)
        >>> pd.DataFrame(values, index=mu, columns=Eout)
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.756174  0.958045  1.008474  1.058893  1.260486  1.512348
        -0.50  0.754676  0.956035  1.006356  1.056671  1.257891  1.509352
         0.00  0.753178  0.954025  1.004237  1.054449  1.255296  1.506356
         0.50  0.751680  0.952015  1.002119  1.052227  1.252702  1.503360
         0.75  0.750931  0.951011  1.001059  1.051116  1.251404  1.501862
        """
        mu_ = InteractEnergy.check_mu(mu)
        return (InteractEnergy.correct_Eout(Eout, Ein, M, mu_) + Ein) / 2

    @staticmethod
    def corr4PCFstrict(Ein: float, Eout: np.ndarray, mu: np.ndarray,
                       M: float) -> np.ndarray:
        """
        Strict calculation of the interaction energy from the corrected 4PCF
        model.

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
        >>> values = InteractEnergy.corr4PCFstrict(Ein, Eout, mu, M)
        >>> pd.DataFrame(values, index=mu, columns=Eout)
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.714340  0.956940  1.008474  1.057490  1.234172  1.424443
        -0.50  0.699100  0.954492  1.006356  1.054871  1.223273  1.393963
         0.00  0.670904  0.951606  1.004237  1.051856  1.204237  1.337571
         0.50  0.595089  0.946971  1.002119  1.047259  1.153623  1.185940
         0.75  0.467472  0.940740  1.001059  1.041413  1.061775  0.930707
        """
        mu_ = InteractEnergy.check_mu(mu)
        Esqrt = np.sqrt(Ein * Eout)
        Ediff = InteractEnergy.correct_Eout(Eout, Ein, M, mu_) - Ein
        EinMat = Ein - Ediff * (mu_ * Esqrt - Ein) / (Ein + Eout - 2 * mu_ * Esqrt)
        return EinMat

    @classmethod
    def from_4PCF(cls, Ein: float, Eout: np.ndarray, mu: np.ndarray,
                  M: float, approx: bool = True, kind: str = "corrected") -> "InteractEnergy":
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
        approx: bool
            Whether to use the approximation or strict calculation
        kind: str
            The type of calculation to be performed. The options are:
            - "original": Original 4PCF model
            - "modified": Modified original 4PCF model
            - "corrected": Corrected 4PCF model

        Returns
        -------
        InteractEnergy
            The interaction energy of the material in eV

        Examples
        --------
        # Example data:
        >>> Ein = 1
        >>> Eout = np.array([0.5, 0.9, 1.0, 1.1, 1.5, 2])
        >>> mu = np.array([-1.0, -0.5, 0.0, 0.5, 0.75])
        >>> M = 238.05077040419212

        # Approximate original 4PCF model
        >>> EinMat = InteractEnergy.from_4PCF(Ein, Eout, mu, M, kind="original")
        >>> EinMat.data
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.754237  0.954237  1.004237  1.054237  1.254237  1.504237
        -0.50  0.752119  0.952119  1.002119  1.052119  1.252119  1.502119
         0.00  0.750000  0.950000  1.000000  1.050000  1.250000  1.500000
         0.50  0.747881  0.947881  0.997881  1.047881  1.247881  1.497881
         0.75  0.746822  0.946822  0.996822  1.046822  1.246822  1.496822

        # Approximate modified 4PCF model
        >>> EinMat = InteractEnergy.from_4PCF(Ein, Eout, mu, M, kind="modified")
        >>> EinMat.data
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.757324  0.958260  1.008474  1.058684  1.259480  1.510411
        -0.50  0.755236  0.956142  1.006356  1.056566  1.257379  1.508353
         0.00  0.753178  0.954025  1.004237  1.054449  1.255296  1.506356
         0.50  0.751241  0.951912  1.002119  1.052335  1.253285  1.504601
         0.75  0.750545  0.950864  1.001059  1.051286  1.252440  1.504268

        # Approximate corrected 4PCF model
        >>> EinMat = InteractEnergy.from_4PCF(Ein, Eout, mu, M, kind="corrected")
        >>> EinMat.data
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.756174  0.958045  1.008474  1.058893  1.260486  1.512348
        -0.50  0.754676  0.956035  1.006356  1.056671  1.257891  1.509352
         0.00  0.753178  0.954025  1.004237  1.054449  1.255296  1.506356
         0.50  0.751680  0.952015  1.002119  1.052227  1.252702  1.503360
         0.75  0.750931  0.951011  1.001059  1.051116  1.251404  1.501862

        # Strict original 4PCF model
        >>> EinMat = InteractEnergy.from_4PCF(Ein, Eout, mu, M, approx=False, kind="original")
        >>> EinMat.data
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.708592  0.950676  1.002100  1.051012  1.227317  1.417184
        -0.50  0.694107  0.949241  1.001050  1.049514  1.217727  1.388215
         0.00  0.666667  0.947368  1.000000  1.047619  1.200000  1.333333
         0.50  0.591607  0.943748  0.998950  1.044142  1.150694  1.183214
         0.75  0.464368  0.938023  0.998425  1.038856  1.059500  0.928737

        # Strict modified 4PCF model
        >>> EinMat = InteractEnergy.from_4PCF(Ein, Eout, mu, M, approx=False, kind="modified")
        >>> EinMat.data
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.711679  0.954698  1.006338  1.055458  1.232560  1.423358
        -0.50  0.697225  0.953265  1.005287  1.053961  1.222988  1.394449
         0.00  0.669845  0.951394  1.004237  1.052068  1.205296  1.339689
         0.50  0.594967  0.947779  1.003187  1.048596  1.156098  1.189933
         0.75  0.468091  0.942065  1.002662  1.043320  1.065118  0.936183

        # Strict corrected 4PCF model
        >>> EinMat = InteractEnergy.from_4PCF(Ein, Eout, mu, M, approx=False, kind="corrected")
        >>> EinMat.data
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.714340  0.956940  1.008474  1.057490  1.234172  1.424443
        -0.50  0.699100  0.954492  1.006356  1.054871  1.223273  1.393963
         0.00  0.670904  0.951606  1.004237  1.051856  1.204237  1.337571
         0.50  0.595089  0.946971  1.002119  1.047259  1.153623  1.185940
         0.75  0.467472  0.940740  1.001059  1.041413  1.061775  0.930707
        """
        method_dict = {
                       (True, "corrected"): cls.corr4PCFapprox,
                       (True, "modified"): cls.mod4PCFapprox,
                       (True, "original"): cls.original4PCFapprox,
                       (False, "corrected"): cls.corr4PCFstrict,
                       (False, "modified"): cls.mod4PCFstrict,
                       (False, "original"): cls.original4PCFstrict
        }
        Einteract = method_dict[(approx, kind.lower())](Ein, Eout, mu, M)
        return cls(Einteract, index=mu, columns=Eout)


class NucInteract:
    """
    Class to calculate the nuclear interaction of the material with the neutron.
    """
    def __init__(self, M: float, T: float, theta: np.ndarray):
        """
        Initialize the class with the data.
        Parameters
        ----------
        data: Iterable
            The data to be stored in the class
        """
        self.M = M
        self.T = T
        self.mu = np.sort(np.cos(np.deg2rad(theta)))
        self.is0KinTinteraction = 180 in np.array(theta)

    def get_interactTemp(self, *args, approx: bool = True) -> np.ndarray:
        """
        Get the interaction temperature of the material in Kelvin.

        Parameters
        ----------
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
        np.ndarray
            The interaction temperature of the material in Kelvin in ascending
            order of mu

        Examples
        --------
        # Example data:
        >>> T = 300
        >>> M = 238.05077040419212
        >>> theta = np.array([30, 60, 90, 120, 150])
        >>> Ein = 1
        >>> Eout = np.array([0.5, 0.9, 1.0, 1.1, 1.5, 2])

        # Initialize the class
        >>> nuclearInteraction = NucInteract(M, T, theta)

        # Nuclear interaction temperature from aproximate 4PCF model
        >>> pd.Series(nuclearInteraction.get_interactTemp(), index=theta[::-1])
        150     20.096189
        120     75.000000
        90     150.000000
        60     225.000000
        30     279.903811
        dtype: float64

        # Nuclear interaction temperature from strict 4PCF model
        >>> values = nuclearInteraction.get_interactTemp(Ein, Eout, approx=False)
        >>> pd.DataFrame(values, index=theta[::-1], columns=Eout)
                    0.5         0.9         1.0         1.1         1.5         2.0
        150   13.762756   19.050750   20.096189   21.064241   24.343692   27.525513
        120   50.971707   71.085473   75.000000   78.601151   90.610233  101.943414
        90   100.000000  142.105263  150.000000  157.142857  180.000000  200.000000
        60   141.885436  212.862866  225.000000  235.447187  264.652925  283.770872
        30   136.237244  262.817382  279.903811  291.097921  297.084879  272.474487
        """
        # Define the method to calculate the interaction temperature in the
        # material from InteractTemp class
        method = InteractTemp.from_4PCF
        return method(self.T, self.mu, *args, approx=approx).data

    def get_interactEnergy(self, Ein: float, Eout: np.ndarray, approx: bool = True,
                           kind: str = "corrected") -> pd.DataFrame:
        """
        Get the interaction energy of the material in eV.

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
        approx: bool
            Whether to use the approximation or strict calculation
        kind: str
            The type of calculation to be performed. The options are:
            - "original": Original 4PCF model
            - "modified": Modified original 4PCF model
            - "corrected": Corrected 4PCF model

        Returns
        -------
        pd.DataFrame
            The interaction energy of the material in eV

        Examples
        --------
        # Example data:
        >>> T = 300
        >>> M = 238.05077040419212
        >>> theta = np.array([30, 60, 90, 120, 150])
        >>> Ein = 1
        >>> Eout = np.array([0.5, 0.9, 1.0, 1.1, 1.5, 2])

        # Initialize the class
        >>> nuclearInteraction = NucInteract(M, T, theta)

        # Nuclear interaction energy from original 4PCF model
        >>> nuclearInteraction.get_interactEnergy(Ein, Eout, kind="original").set_axis(theta[::-1], axis=0)
                  0.5       0.9       1.0       1.1       1.5       2.0
        150  0.753670  0.953670  1.003670  1.053670  1.253670  1.503670
        120  0.752119  0.952119  1.002119  1.052119  1.252119  1.502119
        90   0.750000  0.950000  1.000000  1.050000  1.250000  1.500000
        60   0.747881  0.947881  0.997881  1.047881  1.247881  1.497881
        30   0.746330  0.946330  0.996330  1.046330  1.246330  1.496330

        # Approximate modified 4PCF model
        >>> nuclearInteraction.get_interactEnergy(Ein, Eout, kind="modified").set_axis(theta[::-1], axis=0)
                  0.5       0.9       1.0       1.1       1.5       2.0
        150  0.756763  0.957692  1.007907  1.058116  1.258916  1.509857
        120  0.755236  0.956142  1.006356  1.056566  1.257379  1.508353
        90   0.753178  0.954025  1.004237  1.054449  1.255296  1.506356
        60   0.751241  0.951912  1.002119  1.052335  1.253285  1.504601
        30   0.750683  0.950392  1.000568  1.050812  1.252319  1.505036

        # Approximate corrected 4PCF model
        >>> nuclearInteraction.get_interactEnergy(Ein, Eout, kind="corrected").set_axis(theta[::-1], axis=0)
                  0.5       0.9       1.0       1.1       1.5       2.0
        150  0.755773  0.957507  1.007907  1.058298  1.259791  1.511545
        120  0.754676  0.956035  1.006356  1.056671  1.257891  1.509352
        90   0.753178  0.954025  1.004237  1.054449  1.255296  1.506356
        60   0.751680  0.952015  1.002119  1.052227  1.252702  1.503360
        30   0.750583  0.950544  1.000568  1.050600  1.250802  1.501166

        # Strict original 4PCF model
        >>> nuclearInteraction.get_interactEnergy(Ein, Eout, approx=False, kind="original").set_axis(theta[::-1], axis=0)
                  0.5       0.9       1.0       1.1       1.5       2.0
        150  0.705410  0.950314  1.001819  1.050631  1.225179  1.410821
        120  0.694107  0.949241  1.001050  1.049514  1.217727  1.388215
        90   0.666667  0.947368  1.000000  1.047619  1.200000  1.333333
        60   0.591607  0.943748  0.998950  1.044142  1.150694  1.183214
        30   0.294590  0.928806  0.998181  1.030450  0.917678  0.589179

        # Strict modified 4PCF model
        >>> nuclearInteraction.get_interactEnergy(Ein, Eout, approx=False, kind="modified").set_axis(theta[::-1], axis=0)
                  0.5       0.9       1.0       1.1       1.5       2.0
        150  0.708504  0.954337  1.006056  1.055078  1.230426  1.417008
        120  0.697225  0.953265  1.005287  1.053961  1.222988  1.394449
        90   0.669845  0.951394  1.004237  1.052068  1.205296  1.339689
        60   0.594967  0.947779  1.003187  1.048596  1.156098  1.189933
        30   0.298942  0.932868  1.002418  1.034932  0.923666  0.597885

        # Strict corrected 4PCF model
        >>> nuclearInteraction.get_interactEnergy(Ein, Eout, approx=False, kind="corrected").set_axis(theta[::-1], axis=0)
                  0.5       0.9       1.0       1.1       1.5       2.0
        150  0.710956  0.956307  1.007907  1.056809  1.231683  1.417675
        120  0.699100  0.954492  1.006356  1.054871  1.223273  1.393963
        90   0.670904  0.951606  1.004237  1.051856  1.204237  1.337571
        60   0.595089  0.946971  1.002119  1.047259  1.153623  1.185940
        30   0.297518  0.931288  1.000568  1.032746  0.919649  0.590799
        """
        # Define the method to calculate the interaction energy in the material
        # from InteractEnergy class
        method = InteractEnergy.from_4PCF
        return method(Ein, Eout, self.mu, self.M, approx, kind).data

    def get_interactComb(self, Ein: float, Eout: np.ndarray,
                         approx: bool = True, kind: str = "corrected") -> np.ndarray:
        """
        Get the incident energy and temperature combinations

        Parameters
        ----------
        Tnew: Iterable
            The new temperatures to calculate
        EinGrid: Iterable, None
            The incident energy grid in eV. If not provided, it will be taken
            from the class attribute.

        Returns
        -------
        list
            The incident energy and temperature combinations

        Examples
        --------
        >>> T = 300
        >>> M = 238.05077040419212
        >>> theta = np.array([90, 180])
        >>> Ein = 1
        >>> Eout = np.array([0.5, 0.9, 1.0, 1.1, 1.5, 2])

        # Define the data type for a structured array
        >>> dtype = [('first', 'f8'), ('second', 'f8')]
        # Initialize the class
        >>> nuclearInteraction = NucInteract(M, T, theta)
        >>> comb = nuclearInteraction.get_interactComb(Ein, Eout)
        >>> np.array(comb, dtype=dtype).reshape(EinMat.shape)
        array([[(  0., 0.75617403), (  0., 0.95804507), (  0., 1.00847437),
                (  0., 1.05889304), (  0., 1.26048595), (  0., 1.51234806)],
               [(150., 0.75317789), (150., 0.95402532), (150., 1.00423718),
                (150., 1.05444904), (150., 1.25529648), (150., 1.50635578)]],
              dtype=[('first', '<f8'), ('second', '<f8')])

        # Approximate modified 4PCF model
        >>> comb = nuclearInteraction.get_interactComb(Ein, Eout, approx=False)
        >>> np.array(comb, dtype=dtype).reshape(EinMat.shape)
        array([[(  0.        , 0.71434011), (  0.        , 0.95694023),
                (  0.        , 1.00847437), (  0.        , 1.05749003),
                (  0.        , 1.23417152), (  0.        , 1.42444303)],
               [(100.        , 0.67090385), (142.10526316, 0.95160561),
                (150.        , 1.00423718), (157.14285714, 1.05185623),
                (180.        , 1.20423718), (200.        , 1.33757052)]],
              dtype=[('first', '<f8'), ('second', '<f8')])
        """
        # Get the interaction temperature of the material
        Tinteraction = self.get_interactTemp(Ein, Eout, approx=approx)

        # Get the interaction energy of the material
        EinMat = self.get_interactEnergy(Ein, Eout, approx=approx, kind=kind).values

        # Get the number of dimensions of Tinteraction
        Tinteract_dim = Tinteraction.ndim

        # Case 1: Both are 2D arrays
        if Tinteract_dim == 2:
            return  [(Tinteraction[i, j], EinMat[i, j]) for i in range(EinMat.shape[0])
                    for j in range(EinMat.shape[1])]

        # Case 2: EinGrid is 2D and Tnew is 1D
        else:
            return [(Tinteraction[i], EinMat[i, j]) for i in range(EinMat.shape[0])
                    for j in range(EinMat.shape[1])]

    def calc_interactionXs(self, xs: Xs, Tinteraction: np.ndarray,
                           EinMat: np.ndarray) -> np.ndarray:
        """
        Calculate the interaction cross section of the material.

        Parameters
        ----------
        xs: Xs
            The cross section class with 0K data in barns
        Tinteraction: np.ndarray
            The interaction temperature of the material in Kelvin
        EinMat: np.ndarray
            The interaction energy of the material in eV

        Returns
        -------
        np.ndarray
            The interaction cross section of the material in barns

        Examples
        --------
        # 0K xs data for U238:
        >>> import os
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("nucInteract.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs = Xs.from_xs0K("u238.0.2", M)
        >>> os.chdir(wd)

        # Example data:
        >>> T = 300
        >>> M = 238.05077040419212
        >>> theta = np.array([30, 60, 90, 120, 150, 180])
        >>> Ein = 1
        >>> Eout = np.array([0.5, 0.9, 1.0, 1.1, 1.5, 2])

        # Initialize the class
        >>> nuclearInteraction = NucInteract(M, T, theta)

        # Get the interaction temperature of the material
        >>> Tinteraction = nuclearInteraction.get_interactTemp(Ein, Eout)

        # Get the interaction energy of the material
        >>> EinMat = nuclearInteraction.get_interactEnergy(Ein, Eout).values

        # Calculate the cross section interaction:
        >>> values = nuclearInteraction.calc_interactionXs(xs, Tinteraction, EinMat)
        >>> pd.DataFrame(values, index=nuclearInteraction.mu, columns=Eout)
                            0.5       0.9       1.0       1.1       1.5       2.0
        -1.000000e+00  9.308729  9.276303  9.268018  9.259652  9.225313  9.180326
        -8.660254e-01  9.312016  9.278274  9.269748  9.261169  9.226225  9.180848
        -5.000000e-01  9.312039  9.279556  9.271239  9.262833  9.228322  9.183091
         6.123234e-17  9.310983  9.278747  9.270498  9.262162  9.227947  9.183081
         5.000000e-01  9.310813  9.278633  9.270400  9.262083  9.227958  9.183225
         8.660254e-01  9.310884  9.278735  9.270511  9.262204  9.228123  9.183458
        """
        if self.is0KinTinteraction:
            values0K = interpolation(xs.xs0Kcomplete, EinMat[0], values=True)
            valuesTstar = xs.compute(Tinteraction[1:], EinMat[1:])
            return np.vstack((values0K, valuesTstar))
        else:
            return xs.compute(Tinteraction, EinMat)

    def from_sigma(self, xs: Xs, Ein: float, Eout: np.ndarray,
                   approx: bool = True, kind: str = "corrected") -> pd.DataFrame:
        """
        Calculate the cross section from the nuclear interaction of the material.

        Parameters
        ----------
        xs: Xs
            The cross section class with 0K data in barns
        Ein: float
            The energy of the incident particle in eV
        T: float
            The temperature of the material in Kelvin
        Eout: np.ndarray
            The energy of the outgoing particles in eV
        theta: np.ndarray
            The angle of the outgoing particles in degrees
        approx: bool
            Whether to use the approximation or strict calculation
        kind: str
            The type of calculation to be performed. The options are:
            - "original": Original 4PCF model
            - "modified": Modified original 4PCF model
            - "corrected": Corrected 4PCF model

        Returns
        -------
        pd.DataFrame
            The cross section of the material in barns

        Examples
        --------
        # 0K xs data for U238:
        >>> import os
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("nucInteract.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs = Xs.from_xs0K("u238.0.2", M)
        >>> os.chdir(wd)

        # Example data:
        >>> T = 300
        >>> M = 238.05077040419212
        >>> theta = np.array([30, 60, 90, 120, 150, 180])
        >>> Ein = 1
        >>> Eout = np.array([0.5, 0.9, 1.0, 1.1, 1.5, 2])

        # Initialize the class
        >>> nuclearInteraction = NucInteract(M, T, theta)
        >>> nuclearInteraction.from_sigma(xs, Ein, Eout)
                            0.5       0.9       1.0       1.1       1.5       2.0
        -1.000000e+00  9.308729  9.276303  9.268018  9.259652  9.225313  9.180326
        -8.660254e-01  9.312016  9.278274  9.269748  9.261169  9.226225  9.180848
        -5.000000e-01  9.312039  9.279556  9.271239  9.262833  9.228322  9.183091
         6.123234e-17  9.310983  9.278747  9.270498  9.262162  9.227947  9.183081
         5.000000e-01  9.310813  9.278633  9.270400  9.262083  9.227958  9.183225
         8.660254e-01  9.310884  9.278735  9.270511  9.262204  9.228123  9.183458
        """
        # Get the interaction temperature of the material
        Tinteraction = self.get_interactTemp(Ein, Eout, approx=approx)

        # Get the interaction energy of the material
        EinMat = self.get_interactEnergy(Ein, Eout, approx=approx, kind=kind)

        # Calculate the cross section interaction:
        values = self.calc_interactionXs(xs, Tinteraction, EinMat.values)
        return pd.DataFrame(values, index=self.mu, columns=Eout)

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
