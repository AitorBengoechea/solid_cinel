import numpy as np
import pandas as pd
from scipy.constants import physical_constants as const
from typing import Iterable
from solid_cinel.core.dynamic_structure.alpha import calc_alphaRecoil
from solid_cinel.core.xs.xs0K import Xs0K
from solid_cinel.core.generic import to_arrays
from solid_cinel.core.dynamic_structure.dynamicStruc import DoubleDiffData
from dataclasses import dataclass


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
        return func(Ein, Eout, mu, M) + InteractEnergy.get_4PCFmod(Ein, Eout, mu, M)
    return wrapper

class InteractMu:

    def __init__(self, mu: np.ndarray):
        self.mu = np.unique(to_arrays(mu))
        self.mu2D = self.mu[::, np.newaxis]


@dataclass
class InteractTemp(InteractMu):
    """
    Class to calculate the interaction temperature of a material.
    """
    T: float
    mu: np.ndarray

    def __post_init__(self):
        super().__init__(self.mu)

    @property
    def approx4PCF(self) -> np.ndarray:
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
        >>> Tarno = InteractTemp(T, mu)
        >>> pd.Series(Tarno.approx4PCF, index=mu)
        -1.00      0.0
        -0.50     75.0
         0.00    150.0
         0.50    225.0
         0.75    262.5
        dtype: float64
        """
        return self.T * (1 + self.mu) / 2


    def strict4PCF(self, Ein: float, Eout: np.ndarray) -> np.ndarray:
        """
        Strict calculation of the interaction temperature from the 4PCF model.

        Parameters
        ----------
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
        >>> Tarno = InteractTemp(T, mu)
        >>> pd.DataFrame(Tarno.strict4PCF(Ein, Eout), index=mu, columns=Eout)
                      0.5         0.9    1.0         1.1         1.5         2.0
        -1.00    0.000000    0.000000    0.0    0.000000    0.000000    0.000000
        -0.50   50.971707   71.085473   75.0   78.601151   90.610233  101.943414
         0.00  100.000000  142.105263  150.0  157.142857  180.000000  200.000000
         0.50  141.885436  212.862866  225.0  235.447187  264.652925  283.770872
         0.75  149.371843  247.654462  262.5  274.067269  296.998250  298.743687
        """
        Tstar = self.T * (1 - self.mu2D ** 2) * Eout
        Tstar /= Ein + Eout - 2 * self.mu2D * np.sqrt(Ein * Eout)
        return Tstar

    def to_4PCF(self, *args) -> [pd.DataFrame, pd.Series]:
        """
        Calculate the interaction temperature from the 4PCF model.

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
        >>> Tarno = InteractTemp(T, mu)
        >>> pd.Series(Tarno.to_4PCF(), index=theta[::-1])
        150     20.096189
        120     75.000000
        90     150.000000
        60     225.000000
        30     279.903811
        dtype: float64

        >>> Ein = 1
        >>> Eout = np.array([0.5, 0.9, 1.0, 1.1, 1.5, 2])
        >>> pd.DataFrame(Tarno.to_4PCF(Ein, Eout), index=theta[::-1], columns=Eout)
                    0.5         0.9         1.0         1.1         1.5         2.0
        150   13.762756   19.050750   20.096189   21.064241   24.343692   27.525513
        120   50.971707   71.085473   75.000000   78.601151   90.610233  101.943414
        90   100.000000  142.105263  150.000000  157.142857  180.000000  200.000000
        60   141.885436  212.862866  225.000000  235.447187  264.652925  283.770872
        30   136.237244  262.817382  279.903811  291.097921  297.084879  272.474487
        """
        return self.approx4PCF if len(args) == 0 else self.strict4PCF(*args)


@dataclass
class InteractEnergy(InteractMu):
    """
    Class to calculate the interaction energy of a material.
    """
    Ein: float
    M: float
    Eout: np.ndarray
    mu: np.ndarray

    def __post_init__(self):
        self.Eout = np.unique(self.Eout)
        super().__init__(self.mu)

    @property
    def A(self) -> np.ndarray:
        return self.M / m

    @property
    def Esqrt(self) -> np.ndarray:
        return np.sqrt(self.Ein * self.Eout)

    @property
    def recoil(self) -> np.ndarray:
        return calc_alphaRecoil(self.Ein, self.M, self.Eout, self.mu2D)

    @property
    def recoilMod(self) -> np.ndarray:
        return self.recoil / (2 * (1 - self.mu2D))

    @property
    def correctEout(self) -> np.ndarray:
        return self.Eout + self.recoil

    @property
    def original4PCFapprox(self) -> np.ndarray:
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
        >>> EinMat = InteractEnergy(Ein, M, Eout, mu)
        >>> pd.DataFrame(EinMat.original4PCFapprox, index=mu, columns=Eout)
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.752996  0.954020  1.004237  1.054444  1.255189  1.505992
        -0.50  0.751498  0.952010  1.002119  1.052222  1.252595  1.502996
         0.00  0.750000  0.950000  1.000000  1.050000  1.250000  1.500000
         0.50  0.748502  0.947990  0.997881  1.047778  1.247405  1.497004
         0.75  0.747753  0.946985  0.996822  1.046667  1.246108  1.495506
        """
        return (self.Eout + self.Ein) / 2 - self.mu2D * self.Esqrt / self.A

    @property
    def original4PCFstrict(self) -> np.ndarray:
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
        >>> EinMat = InteractEnergy(Ein, M, Eout, mu)
        >>> pd.DataFrame(EinMat.original4PCFstrict, index=mu, columns=Eout)
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.708592  0.950676  1.002100  1.051012  1.227317  1.417184
        -0.50  0.694107  0.949241  1.001050  1.049514  1.217727  1.388215
         0.00  0.666667  0.947368  1.000000  1.047619  1.200000  1.333333
         0.50  0.591607  0.943748  0.998950  1.044142  1.150694  1.183214
         0.75  0.464368  0.938023  0.998425  1.038856  1.059500  0.928737
        """
        recoilCalc = (self.Eout - self.Ein) * (self.mu2D * self.Esqrt - self.Ein)
        recoilCalc /= self.Ein + self.Eout - 2 * self.mu2D * self.Esqrt
        return self.Ein - self.mu2D * self.Esqrt / (2 * self.M) - recoilCalc

    def apply_mod(self, originalFunc: np.ndarray):
        return originalFunc + self.recoilMod

    @property
    def mod4PCFapprox(self) -> np.ndarray:
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
        >>> EinMat = InteractEnergy(Ein, M, Eout, mu)
        >>> pd.DataFrame(EinMat.mod4PCFapprox, index=mu, columns=Eout)
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.756083  0.958042  1.008474  1.058891  1.260432  1.512166
        -0.50  0.754615  0.956033  1.006356  1.056669  1.257856  1.509231
         0.00  0.753178  0.954025  1.004237  1.054449  1.255296  1.506356
         0.50  0.751862  0.952021  1.002119  1.052232  1.252809  1.503723
         0.75  0.751476  0.951027  1.001059  1.051131  1.251725  1.502952
        """
        return self.apply_mod(self.original4PCFapprox)

    @property
    def mod4PCFstrict(self) -> np.ndarray:
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
        >>> EinMat = InteractEnergy(Ein, M, Eout, mu)
        >>> pd.DataFrame(EinMat.mod4PCFstrict, index=mu, columns=Eout)
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.711679  0.954698  1.006338  1.055458  1.232560  1.423358
        -0.50  0.697225  0.953265  1.005287  1.053961  1.222988  1.394449
         0.00  0.669845  0.951394  1.004237  1.052068  1.205296  1.339689
         0.50  0.594967  0.947779  1.003187  1.048596  1.156098  1.189933
         0.75  0.468091  0.942065  1.002662  1.043320  1.065118  0.936183
        """
        return self.apply_mod(self.original4PCFstrict)

    @property
    def corr4PCFapprox(self) -> np.ndarray:
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
        >>> EinMat = InteractEnergy(Ein, M, Eout, mu)
        >>> pd.DataFrame(EinMat.corr4PCFapprox, index=mu, columns=Eout)
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.756174  0.958045  1.008474  1.058893  1.260486  1.512348
        -0.50  0.754676  0.956035  1.006356  1.056671  1.257891  1.509352
         0.00  0.753178  0.954025  1.004237  1.054449  1.255296  1.506356
         0.50  0.751680  0.952015  1.002119  1.052227  1.252702  1.503360
         0.75  0.750931  0.951011  1.001059  1.051116  1.251404  1.501862
        """
        return (self.correctEout + self.Ein) / 2

    @property
    def corr4PCFstrict(self) -> np.ndarray:
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
        >>> EinMat = InteractEnergy(Ein, M, Eout, mu)
        >>> pd.DataFrame(EinMat.corr4PCFstrict, index=mu, columns=Eout)
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.715785  0.957126  1.008439  1.057254  1.233277  1.422918
        -0.50  0.700530  0.954687  1.006329  1.054644  1.222392  1.392475
         0.00  0.672292  0.951810  1.004219  1.051637  1.203375  1.336146
         0.50  0.596334  0.947184  1.002110  1.047049  1.152803  1.184681
         0.75  0.468456  0.940956  1.001055  1.041208  1.061022  0.929720
        """
        Eout_ = self.correctEout
        Esqrt = np.sqrt(self.Ein * Eout_)
        Ediff = Eout_ - self.Ein
        return self.Ein - Ediff * (self.mu2D * Esqrt - self.Ein) / (self.Ein + Eout_ - 2 * self.mu2D * Esqrt)

    def to_4PCF(self, approx: bool = True,
                  kind: str = "corrected") -> "InteractEnergy":
        """
        Calculate the interaction energy from the 4PCF model.

        Parameters
        ----------
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
        >>> EinMat = InteractEnergy(Ein, M, Eout, mu)

        # Approximate original 4PCF model
        >>> pd.DataFrame(EinMat.to_4PCF(kind="original"), index=mu, columns=Eout)
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.752996  0.954020  1.004237  1.054444  1.255189  1.505992
        -0.50  0.751498  0.952010  1.002119  1.052222  1.252595  1.502996
         0.00  0.750000  0.950000  1.000000  1.050000  1.250000  1.500000
         0.50  0.748502  0.947990  0.997881  1.047778  1.247405  1.497004
         0.75  0.747753  0.946985  0.996822  1.046667  1.246108  1.495506

        # Approximate modified 4PCF model
        >>> pd.DataFrame(EinMat.to_4PCF(kind="modified"), index=mu, columns=Eout)
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.756083  0.958042  1.008474  1.058891  1.260432  1.512166
        -0.50  0.754615  0.956033  1.006356  1.056669  1.257856  1.509231
         0.00  0.753178  0.954025  1.004237  1.054449  1.255296  1.506356
         0.50  0.751862  0.952021  1.002119  1.052232  1.252809  1.503723
         0.75  0.751476  0.951027  1.001059  1.051131  1.251725  1.502952

        # Approximate corrected 4PCF model
        >>> pd.DataFrame(EinMat.to_4PCF(kind="corrected"), index=mu, columns=Eout)
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.756174  0.958045  1.008474  1.058893  1.260486  1.512348
        -0.50  0.754676  0.956035  1.006356  1.056671  1.257891  1.509352
         0.00  0.753178  0.954025  1.004237  1.054449  1.255296  1.506356
         0.50  0.751680  0.952015  1.002119  1.052227  1.252702  1.503360
         0.75  0.750931  0.951011  1.001059  1.051116  1.251404  1.501862

        # Strict original 4PCF model
        >>> pd.DataFrame(EinMat.to_4PCF(approx=False, kind="original"), index=mu, columns=Eout)
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.708592  0.950676  1.002100  1.051012  1.227317  1.417184
        -0.50  0.694107  0.949241  1.001050  1.049514  1.217727  1.388215
         0.00  0.666667  0.947368  1.000000  1.047619  1.200000  1.333333
         0.50  0.591607  0.943748  0.998950  1.044142  1.150694  1.183214
         0.75  0.464368  0.938023  0.998425  1.038856  1.059500  0.928737

        # Strict modified 4PCF model
        >>> pd.DataFrame(EinMat.to_4PCF(approx=False, kind="modified"), index=mu, columns=Eout)
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.711679  0.954698  1.006338  1.055458  1.232560  1.423358
        -0.50  0.697225  0.953265  1.005287  1.053961  1.222988  1.394449
         0.00  0.669845  0.951394  1.004237  1.052068  1.205296  1.339689
         0.50  0.594967  0.947779  1.003187  1.048596  1.156098  1.189933
         0.75  0.468091  0.942065  1.002662  1.043320  1.065118  0.936183

        # Strict corrected 4PCF model
        >>> pd.DataFrame(EinMat.to_4PCF(approx=False, kind="corrected"), index=mu, columns=Eout)
                    0.5       0.9       1.0       1.1       1.5       2.0
        -1.00  0.715785  0.957126  1.008439  1.057254  1.233277  1.422918
        -0.50  0.700530  0.954687  1.006329  1.054644  1.222392  1.392475
         0.00  0.672292  0.951810  1.004219  1.051637  1.203375  1.336146
         0.50  0.596334  0.947184  1.002110  1.047049  1.152803  1.184681
         0.75  0.468456  0.940956  1.001055  1.041208  1.061022  0.929720
        """
        kind = kind.lower()
        if kind not in ["original", "modified", "corrected"]:
            raise ValueError("kind must be one of 'original', 'modified', or 'corrected'")
        else:
            kind_dict = {
                            "original": "original4PCF",
                            "modified": "mod4PCF",
                            "corrected": "corr4PCF"
            }
        approx_dict = {
                        True: "approx",
                        False: "strict"
        }
        return getattr(self, "".join([kind_dict[kind], approx_dict[approx]]))



class NucInteract(DoubleDiffData):
    """
    Class to calculate the nuclear interaction of the material with the neutron.
    """
    def __init__(self, xs0K: Xs0K, EinMat: InteractEnergy, *args,
                 Tinteract: InteractTemp = None, **kwargs):
        """
        Initialize the class with the data.
        Parameters
        ----------
        data: Iterable
            The data to be stored in the class
        """
        self.xs0K = xs0K
        self.EinMat = EinMat
        self.Tinteract = Tinteract
        super().__init__(*args, **kwargs)

    @staticmethod
    def XsMat_sigma1(xs0K: Xs0K, EinMat: np.ndarray, Tinteraction: np.ndarray,
                     is0K: bool) -> np.ndarray:
        """
        Calculate the cross section of the material from the nuclear interaction.

        Parameters
        ----------
        xs0K: Xs0K
            The cross section class with 0K data in barns
        EinMat: np.ndarray
            The interaction energy of the material in eV
        Tinteraction: np.ndarray
            The interaction temperature of the material in Kelvin
        is0K: bool
            Whether the data is at 0K or not

        Returns
        -------
        np.ndarray
            The cross section of the material in barns
        """
        if is0K:
            values0K = xs0K.interpolate(EinMat[0], values=True)
            valuesTstar = xs0K.sigma1(Tinteraction[1:], EinMat[1:], values=True)
            return np.vstack((values0K, valuesTstar))
        else:
            return xs0K.sigma1(Tinteraction, EinMat, values=True)

    @classmethod
    def from_sigma(cls, xs0K: Xs0K, Ein: float, T: float, Eout: np.ndarray,
                   theta: np.ndarray, approx: bool = True,
                   kind: str = "corrected") -> "NucInteract":
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
        >>> xs0K = Xs0K.from_file("u238.0.2", M)
        >>> os.chdir(wd)

        # Example data:
        >>> T = 300
        >>> M = 238.05077040419212
        >>> theta = np.array([30, 60, 90, 120, 150, 180])
        >>> Ein = 6.67
        >>> Eout = np.array([6.5, 6.6, 6.67, 6.8, 6.9])

        # Initialize the class
        >>> NucInteract.from_sigma(xs0K, Ein, T, Eout, theta).data
        Eout                 6.50        6.60        6.67        6.80       6.90
        mu
        -1.000000e+00  109.429067  578.174610  132.000620   47.804039  33.258262
        -8.660254e-01  114.424855  797.111411  158.423511   50.115934  34.138420
        -5.000000e-01  111.529220  749.375088  314.215063   58.499214  36.945805
         6.123234e-17   85.702705  542.581865  491.664735   82.203465  42.255244
         5.000000e-01   62.495084  387.510240  511.456837  140.088731  51.371004
         8.660254e-01   49.379282  304.044914  474.211771  201.620717  64.069957
        """
        # Calculate mu:
        theta_ = to_arrays(theta)
        mu = np.sort(np.cos(np.deg2rad(theta_)))

        # Get the interaction temperature of the material
        Tinteraction = InteractTemp.from_4PCF(T, mu, Ein, Eout, approx=approx)

        # Get the interaction energy of the material
        EinMat = InteractEnergy.from_4PCF(Ein, Eout, mu,  xs0K.M, approx=approx,
                                          kind=kind)

        # Calculate the cross section interaction:
        interactValues = cls.XsMat_sigma1(xs0K, EinMat.values, Tinteraction,
                                          180 in theta_)

        return cls(xs0K, EinMat, interactValues, index=mu, columns=Eout,
                   Tinteract=Tinteraction)
