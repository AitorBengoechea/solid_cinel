import numpy as np
import pandas as pd
from scipy.constants import physical_constants as const
from typing import Iterable
from solid_cinel.core.scattering_function.alpha import get_alphaRecoil


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
                  M: float, original: bool = False, mod: bool = True,
                  approx: bool = True) -> "InteractEnergy":
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
        -1.00  0.756174  0.958045  1.008474  1.058893  1.260486  1.512348
        -0.50  0.754676  0.956035  1.006356  1.056671  1.257891  1.509352
         0.00  0.753178  0.954025  1.004237  1.054449  1.255296  1.506356
         0.50  0.751680  0.952015  1.002119  1.052227  1.252702  1.503360
         0.75  0.750931  0.951011  1.001059  1.051116  1.251404  1.501862

        >>> EinMat = InteractEnergy.from_4PCF(Ein, Eout, mu, M, approx=False)
        >>> EinMat.data
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.714340  0.956940  1.008474  1.057490  1.234172  1.424443
        -0.50  0.699100  0.954492  1.006356  1.054871  1.223273  1.393963
         0.00  0.670904  0.951606  1.004237  1.051856  1.204237  1.337571
         0.50  0.595089  0.946971  1.002119  1.047259  1.153623  1.185940
         0.75  0.467472  0.940740  1.001059  1.041413  1.061775  0.930707

        >>> EinMat = InteractEnergy.from_4PCF(Ein, Eout, mu, M, original=True, approx= True, mod=False)
        >>> EinMat.data
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.754237  0.954237  1.004237  1.054237  1.254237  1.504237
        -0.50  0.752119  0.952119  1.002119  1.052119  1.252119  1.502119
         0.00  0.750000  0.950000  1.000000  1.050000  1.250000  1.500000
         0.50  0.747881  0.947881  0.997881  1.047881  1.247881  1.497881
         0.75  0.746822  0.946822  0.996822  1.046822  1.246822  1.496822

        >>> EinMat = InteractEnergy.from_4PCF(Ein, Eout, mu, M, original=True, approx= True, mod=True)
        >>> EinMat.data
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.757324  0.958260  1.008474  1.058684  1.259480  1.510411
        -0.50  0.755236  0.956142  1.006356  1.056566  1.257379  1.508353
         0.00  0.753178  0.954025  1.004237  1.054449  1.255296  1.506356
         0.50  0.751241  0.951912  1.002119  1.052335  1.253285  1.504601
         0.75  0.750545  0.950864  1.001059  1.051286  1.252440  1.504268

        >>> EinMat = InteractEnergy.from_4PCF(Ein, Eout, mu, M, original=True, approx= False, mod=True)
        >>> EinMat.data
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.711679  0.954698  1.006338  1.055458  1.232560  1.423358
        -0.50  0.697225  0.953265  1.005287  1.053961  1.222988  1.394449
         0.00  0.669845  0.951394  1.004237  1.052068  1.205296  1.339689
         0.50  0.594967  0.947779  1.003187  1.048596  1.156098  1.189933
         0.75  0.468091  0.942065  1.002662  1.043320  1.065118  0.936183

        >>> EinMat = InteractEnergy.from_4PCF(Ein, Eout, mu, M, original=True, approx= False, mod=True)
        >>> EinMat.data
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.711679  0.954698  1.006338  1.055458  1.232560  1.423358
        -0.50  0.697225  0.953265  1.005287  1.053961  1.222988  1.394449
         0.00  0.669845  0.951394  1.004237  1.052068  1.205296  1.339689
         0.50  0.594967  0.947779  1.003187  1.048596  1.156098  1.189933
         0.75  0.468091  0.942065  1.002662  1.043320  1.065118  0.936183
        """
        if original:
            if approx:
                if mod:
                    Einteract = cls.mod4PCFapprox(Ein, Eout, mu, M)
                else:
                    Einteract = cls.original4PCFapprox(Ein, Eout, mu, M)
            else:
                if mod:
                    Einteract = cls.mod4PCFstrict(Ein, Eout, mu, M)
                else:
                    Einteract = cls.original4PCFstrict(Ein, Eout, mu, M)
        else:
            if approx:
                Einteract = cls.corr4PCFapprox(Ein, Eout, mu, M)
            else:
                Einteract = cls.corr4PCFstrict(Ein, Eout, mu, M)
        return cls(Einteract, index=mu, columns=Eout)
class NucInteract:
    pass