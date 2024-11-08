import numpy as np
import pandas as pd
from scipy.constants import physical_constants as const
from typing import Iterable
from solid_cinel.core.dynamic_structure.alpha import calc_alphaRecoil
from solid_cinel.core.xs.xs0K import Xs0K
from solid_cinel.core.generic import to_arrays
from solid_cinel.core.dynamic_structure.dynamicStruc import DoubleDiffData, DoubleDiff
from dataclasses import dataclass


# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]


@dataclass
class InteractTemp:
    """
    Class to calculate the interaction temperature of a material.
    """
    T: float
    mu: np.ndarray

    def __post_init__(self):
        self.mu = np.unique(to_arrays(self.mu))
        self.mu2D = self.mu[::, np.newaxis]

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

    @property
    def p0(self) -> np.ndarray:
        """
        The p0 value for the strict and approx calculation of the interaction
        temperature.

        Returns
        -------
        np.ndarray
            The p0 value for the strict calculation of the interaction temperature

        Examples
        --------
        >>> T = 300
        >>> mu = np.array([-1.0, -0.5, 0.0, 0.5, 0.75])
        >>> Tarno = InteractTemp(T, mu)
        >>> pd.Series(Tarno.p0, index=mu)
        -1.00      0.00
        -0.50     37.50
         0.00     75.00
         0.50    112.50
         0.75    131.25
        dtype: float64
        """
        return self.approx4PCF / 2

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
        >>> mu = np.array([-1.0, -0.5, 0.0, 0.5, 0.75, 1.0])
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
         1.00    0.000000    0.000000    0.0    0.000000    0.000000    0.000000
        """
        Tstar = self.T * (1 - self.mu2D ** 2) * Eout
        Tstar /= Ein + Eout - 2 * self.mu2D * np.sqrt(Ein * Eout)
        return np.nan_to_num(Tstar, nan=0.0)

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


class InteractEnergy(DoubleDiff):
    """
    Class to calculate the interaction energy of a material.
    """

    def __init__(self, Ein: float, M: float, Eout: np.ndarray, mu: np.ndarray):
        """
        Initialize the Sab_to_DynamicStruc class.

        Parameters
        ----------
        Ein : float
            The incident energy of the neutron in eV
        M : float
            The mass of the target material in amu
        T : float
            Temperature of the material in K
        Eout : np.array
            The neutron outgoing energy
        mu : np.ndarray
            The cosine of the angle of the distribution in degrees
        """
        super().__init__(Ein, M, Eout, mu)

    @property
    def Esqrt(self) -> np.ndarray:
        """
        Square root of the product of the incident and outgoing energies for
        avoiding code duplication

        Returns
        -------
        np.ndarray
            The square root of the product of the incident and outgoing energies
            in eV
        """
        return np.sqrt(self.Ein * self.Eout)

    @property
    def Eplus(self) -> np.ndarray:
        """
        Sum of the incident and outgoing energies for aboiding code duplication

        Returns
        -------
        np.ndarray
            The sum of the incident and outgoing energies in eV
        """
        return self.Ein + self.Eout

    @property
    def recoilMod(self) -> np.ndarray:
        """
        Recoil energy modification

        Returns
        -------
        np.ndarray
            The recoil energy modification in eV
        """
        return self.recoil / (2 * (1 - self.mu2D))

    @property
    def correctEout(self) -> np.ndarray:
        """
        Corrected outgoing energy with the recoil energy

        Returns
        -------
        np.ndarray
            The corrected outgoing energy in eV with the recoil energy
        """
        return self.Eout + self.recoil

    @property
    def original4PCFapprox(self) -> np.ndarray:
        """
        Approximation of the interaction energy from the original 4PCF model.

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
        return self.Eplus / 2 - self.mu2D * self.Esqrt / self.A

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
        # Save the value in local variable for avoiding repeated calculation:
        Esqrt = self.Esqrt

        # Do the rest of the calculation:
        recoilCalc = self.dE * (self.mu2D * Esqrt - self.Ein)
        recoilCalc /= self.Ein + self.Eout - 2 * self.mu2D * Esqrt
        return self.Ein - self.mu2D * Esqrt / (2 * self.M) - recoilCalc

    def apply_mod(self, originalFunc: np.ndarray) -> np.ndarray:
        """
        Apply the modification to the original function.

        Parameters
        ----------
        originalFunc : np.ndarray

        Returns
        -------
        np.ndarray
            Original function modified by the recoil
        """
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
                    0.5       0.9  1.0       1.1       1.5       2.0
        -1.00  0.708588  0.948905  1.0  1.048608  1.223886  1.412725
        -0.50  0.694822  0.948467  1.0  1.048211  1.215587  1.385269
         0.00  0.668073  0.947590  1.0  1.047418  1.199156  1.331927
         0.50  0.593604  0.944966  1.0  1.045043  1.151169  1.183448
         0.75  0.466470  0.939738  1.0  1.040308  1.060682  0.929981
        """
        Eout_ = self.correctEout
        Esqrt = np.sqrt(self.Ein * Eout_)
        return self.Ein - self.dE * (self.mu2D * Esqrt - self.Ein) / (self.Ein + Eout_ - 2 * self.mu2D * Esqrt)

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
                    0.5       0.9  1.0       1.1       1.5       2.0
        -1.00  0.708588  0.948905  1.0  1.048608  1.223886  1.412725
        -0.50  0.694822  0.948467  1.0  1.048211  1.215587  1.385269
         0.00  0.668073  0.947590  1.0  1.047418  1.199156  1.331927
         0.50  0.593604  0.944966  1.0  1.045043  1.151169  1.183448
         0.75  0.466470  0.939738  1.0  1.040308  1.060682  0.929981
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

    @property
    def originalP0(self) -> np.ndarray:
        """
        Calculate the original P0 of the material.

        Returns
        -------
        float
            The original P0 of the material

        Examples
        --------
        # Example data:
        >>> Ein = 1
        >>> Eout = np.array([0.5, 0.9, 1.0, 1.1, 1.5, 2])
        >>> mu = np.array([-1.0, -0.5, 0.0, 0.5, 0.75])
        >>> M = 238.05077040419212
        >>> EinMat = InteractEnergy(Ein, M, Eout, mu)
        >>> pd.Series(EinMat.originalP0, index=mu)
        -1.00    0.995763
        -0.50    0.997881
         0.00    1.000000
         0.50    1.002119
         0.75    1.003178
        dtype: float64
        """
        return self.Ein * (self.A + self.mu) / self.A
    @property
    def correctP0(self) -> np.ndarray:
        """
        Calculate the original P0 of the material.

        Returns
        -------
        float
            The original P0 of the material

        Examples
        --------
        # Example data:
        >>> Ein = 1
        >>> Eout = np.array([0.5, 0.9, 1.0, 1.1, 1.5, 2])
        >>> mu = np.array([-1.0, -0.5, 0.0, 0.5, 0.75])
        >>> M = 238.05077040419212
        >>> EinMat = InteractEnergy(Ein, M, Eout, mu)
        >>> pd.Series(EinMat.correctP0, index=mu)
        -1.00    1.000000
        -0.50    1.002119
         0.00    1.004237
         0.50    1.006356
         0.75    1.007415
        dtype: float64
        """
        return self.originalP0 + self.Ein / self.A

    def p0(self, kind: str = "corrected") -> np.ndarray:
        """
        Calculate the P0 of the material.
        Parameters
        ----------
        kind: str
            The type of calculation to be performed. The options are:
            - "original": Original P0
            - "modified": Modified original P0
            - "corrected": Corrected P0

        Returns
        -------
        np.ndarray
            The P0 of the material

        Examples
        --------
        # Example data:
        >>> Ein = 1
        >>> Eout = np.array([0.5, 0.9, 1.0, 1.1, 1.5, 2])
        >>> mu = np.array([-1.0, -0.5, 0.0, 0.5, 0.75])
        >>> M = 238.05077040419212
        >>> EinMat = InteractEnergy(Ein, M, Eout, mu)
        >>> pd.Series(EinMat.p0(), index=mu)
        -1.00    1.000000
        -0.50    1.002119
         0.00    1.004237
         0.50    1.006356
         0.75    1.007415
        dtype: float64

        >>> pd.Series(EinMat.p0("original"), index=mu)
        -1.00    0.995763
        -0.50    0.997881
         0.00    1.000000
         0.50    1.002119
         0.75    1.003178
        dtype: float64
        """
        kind = kind.lower()
        if kind not in ["original", "modified", "corrected"]:
            raise ValueError("kind must be one of 'original', 'modified', or 'corrected'")
        else:
            kind_dict = {
                            "original": "originalP0",
                            "modified": "correctP0",
                            "corrected": "correctP0"
            }
        return getattr(self, "".join([kind_dict[kind]]))


class NucInteractBase:
    """
    Class to calculate the nuclear interaction of the material with the neutron.
    """
    def __init__(self, xs0K: Xs0K, Tinteract: InteractTemp,
                 EinMat: InteractEnergy) -> "NucInteractBase":
        """
        Initialize the class with the data.
        Parameters
        ----------
        data: Iterable
            The data to be stored in the class
        """
        self._xs0K = xs0K
        self._Tinteract = Tinteract
        self._EinMat = EinMat

    @classmethod
    def from_theta(cls, xs0K: Xs0K, Ein: float, T: float, Eout: np.ndarray,
                   theta: np.ndarray) -> "NucInteractBase":
        """
        Create the class from the data.

        Parameters
        ----------
        xs0K : Xs0K
            The 0K cross section data of the material
        Ein : float
            Incident energy of the neutron in eV
        T : float
            Temperature of the material in Kelvin
        Eout : np.ndarray
            Outgoing energy of the neutron in eV
        theta : np.ndarray
            The angle of the distribution in degrees

        Returns
        -------
        NucInteractBase
            The class with the data
        """
        # Calculate mu:
        theta_ = to_arrays(theta)
        mu = np.sort(np.cos(np.deg2rad(theta_)))

        return cls(xs0K, InteractTemp(T, mu),
                   InteractEnergy(Ein, xs0K.M, Eout, mu))

    def alphaCapt(self, approx: bool = True, kind: str = "corrected"):
        """
        Calculate the alpha capture of the material.

        Parameters
        ----------
        approx
        kind

        Returns
        -------

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
        >>> mu = np.sort(np.cos(np.deg2rad(theta)))
        >>> Ein = 6.67
        >>> Eout = np.array([6.5, 6.6, 6.67, 6.8, 6.9])
        >>> nuc = NucInteractBase.from_theta(xs0K, Ein, T, Eout, theta)
        >>> pd.DataFrame(nuc.alphaCapt(), index=mu, columns=Eout)
                              6.50         6.60         6.67         6.80         6.90
        -1.000000e+00          inf          inf          inf          inf          inf
        -8.660254e-01  3832.563612  3861.665299  3882.035998  3919.866269  3948.965588
        -5.000000e-01  1025.352263  1033.137929  1038.587820  1048.708888  1056.494187
         6.123234e-17   511.596930   515.481492   518.200686   523.250618   527.135181
         5.000000e-01   340.345152   342.929347   344.738309   348.097862   350.682179
         8.660254e-01   273.162249   275.236304   276.688177   279.384586   281.458811
        """
        # Get the interaction temperature of the material
        if approx:
            Tinteraction = self._Tinteract.to_4PCF()[::, np.newaxis]
        else:
            Tinteraction = self._Tinteract.to_4PCF(self._EinMat.Ein, self._EinMat.Eout)

        # Get the alpha capture of the material
        return self._EinMat.to_4PCF(approx, kind) / (kb * Tinteraction)

    def calc_sigma1(self, approx: bool = True, kind: str = "corrected") -> np.ndarray:
        """
        Calculate the cross section from the nuclear interaction of the material.

        Parameters
        ----------
        approx: bool
            Whether to use the approximation or strict calculation. Default is
            True.
        kind: str
            The type of calculation to be performed. The options are:
            - "original": Original 4PCF model
            - "modified": Modified original 4PCF model
            - "corrected": Corrected 4PCF model

        Returns
        -------
        np.ndarray
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
        >>> mu = np.sort(np.cos(np.deg2rad(theta)))
        >>> Ein = 6.67
        >>> Eout = np.array([6.5, 6.6, 6.67, 6.8, 6.9])
        >>> nuc = NucInteractBase.from_theta(xs0K, Ein, T, Eout, theta)

        # Calculate the values
        >>> pd.DataFrame(nuc.calc_sigma1(), index=mu, columns=Eout)
                             6.50        6.60        6.67        6.80       6.90
        -1.000000e+00  109.429067  578.174610  132.000620   47.804039  33.258262
        -8.660254e-01  114.424855  797.111411  158.423511   50.115934  34.138420
        -5.000000e-01  111.529220  749.375088  314.215063   58.499214  36.945805
         6.123234e-17   85.702705  542.581865  491.664735   82.203465  42.255244
         5.000000e-01   62.495084  387.510240  511.456837  140.088731  51.371004
         8.660254e-01   49.379282  304.044914  474.211771  201.620717  64.069957
        """
        # Get the interaction temperature of the material
        args = [] if approx else [self._EinMat.Ein, self._EinMat.Eout]
        Tinteraction = self._Tinteract.to_4PCF(*args)

        # Get the interaction energy of the material
        EinMat = self._EinMat.to_4PCF(approx, kind)

        # Calculate the interaction cross section:
        if approx and Tinteraction[0] == 0:
            return np.vstack(
                (
                    self._xs0K.interpolate(EinMat[0], values=True),
                    self._xs0K.db_sigma1(Tinteraction[1:], EinMat[1:])
                )
            )
        elif not approx:
            result = []

            # Check if the first value is zero (mu = -1):
            if Tinteraction[0, 0] == 0:
                start = 1
                result.append(self._xs0K.interpolate(EinMat[0], values=True))
            else:
                start = 0

            # Check if the last value is zero (mu = 1):
            ismu1 = Tinteraction[-1, 0] == 0
            end = len(Tinteraction[0, :]) - 2 if ismu1 else len(Tinteraction[0, :]) - 1

            # Sigma1 algorithm calculation for intermediate values
            result.append(
                self._xs0K.db_sigma1(Tinteraction[start:end], EinMat[start:end]))

            # Add the last value if mu = 1
            if ismu1:
                result.append(self._xs0K.interpolate(EinMat[-1], values=True))
            return np.vstack(result)
        else:
            return self._xs0K.db_sigma1(Tinteraction, EinMat)

    def calc_alpha0(self) -> np.ndarray:
        """
        Not implemented yet.

        Returns
        -------
        np.ndarray
            The Doppler broadening with the alpha0 methodology
        """
        return


class NucInteract(DoubleDiffData):
    def __init__(self, approx: bool, kind: str,
                 *args, nuc: NucInteractBase = None, **kwargs):
        """
        Initialize the class with the data.
        Parameters
        ----------
        data: Iterable
            The data to be stored in the class
        """
        # Atributes of the Nuclear Interaction class:
        self.approx = approx
        self.kind = kind

        # Initialize the parent class:
        super().__init__(*args, **kwargs)

        # Llamar a __post_init__ para inicialización adicional
        if nuc is not None:
            self.__post_init__(nuc)

    def __post_init__(self, nuc: NucInteractBase):
        # Extract the values
        self.Ein = nuc._EinMat.Ein
        self.T = nuc._Tinteract.T

        # Extract the class data:
        self._xs0K = nuc._xs0K
        self._EinMat = nuc._EinMat.to_4PCF(self.approx, self.kind)
        if not self.approx:
            self._Tinteract = nuc._Tinteract.to_4PCF(self.Ein, self.Eout)
        else:
            self._Tinteract = nuc._Tinteract.to_4PCF()

    @classmethod
    def from_sigma(cls, xs0K: Xs0K, Ein: float, T: float, Eout: np.ndarray,
                   theta: np.ndarray, approx: bool = True,
                   kind: str = "corrected") -> "NucInteract":
        """
        Initialize the class with the data.

        Parameters
        ----------
        xs0K : Xs0K
            The 0K cross section data of the material
        Ein : float
            Incident energy of the neutron in eV
        T : float
            Temperature of the material in Kelvin
        Eout : np.ndarray
            Outgoing energy of the neutron in eV
        theta : np.ndarray
            The angle of the distribution in degrees
        approx : bool
            Whether to use the approximation or strict calculation. Default is
            True.
        kind : str
            The type of calculation to be performed. The options are:
                - "original": Original 4PCF model
                - "modified": Modified original 4PCF model
                - "corrected": Corrected 4PCF model

        Returns
        -------
        NucInteract
            The class with the data

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
        >>> mu = np.sort(np.cos(np.deg2rad(theta)))
        >>> Ein = 6.67
        >>> Eout = np.array([6.5, 6.6, 6.67, 6.8, 6.9])
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
        # Initialize the class with the data:
        nuc = NucInteractBase.from_theta(xs0K, Ein, T, Eout, theta)

        # Calculate the cross section:
        return cls(approx, kind, nuc.calc_sigma1(approx, kind),
                   index=nuc._EinMat.mu, columns=nuc._EinMat.Eout, nuc=nuc)

    @property
    def norm(self) -> float:
        """
        Normalization of the Dynamic Structure Factor.

        Returns
        -------
        float
            Normalization of the Dynamic Structure Factor
        """
        return super().doubleIntegral

    @property
    def transferFunc(self) -> pd.Series:
        """
        Return the Transference function of the Nuclear interaction.

        Returns
        -------
        pd.Series
            The transfer function

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
        >>> mu = np.sort(np.cos(np.deg2rad(theta)))
        >>> Ein = 6.67
        >>> Eout = np.array([6.5, 6.6, 6.67, 6.8, 6.9])
        >>> nuc = NucInteract.from_sigma(xs0K, Ein, T, Eout, theta)
        >>> nuc.transferFunc.round(6)
        Eout
        6.50     163.179694
        6.60    1057.229009
        6.67     738.593808
        6.80     179.723234
        6.90      81.858024
        dtype: float64
        """
        return super().columsIntegral

    @property
    def angularDistr(self) -> pd.Series:
        """
        Return the angle distribution of the Nuclear interaction.

        Returns
        -------
        pd.Series
            The angle distribution

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
        >>> mu = np.sort(np.cos(np.deg2rad(theta)))
        >>> Ein = 6.67
        >>> Eout = np.array([6.5, 6.6, 6.67, 6.8, 6.9])
        >>> nuc = NucInteract.from_sigma(xs0K, Ein, T, Eout, theta)
        >>> nuc.angularDistr.round(6)
        mu
        -1.000000e+00     74.976735
        -8.660254e-01     96.788317
        -5.000000e-01    109.269550
         6.123234e-17    111.137228
         5.000000e-01    105.887563
         8.660254e-01    102.123839
        dtype: float64
        """
        return super().rowIntegral
