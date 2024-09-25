"""
Python file for working with scattering functions.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
from numba import float64
from scipy.constants import physical_constants as const
from solid_cinel.core.generic import integrate, interp_multyParallel
from solid_cinel.core.scattering_function.beta import get_AbsBeta, calc_Beta
from solid_cinel.core.scattering_function.alpha import get_alphaMat, get_alphaMatMod, get_alphaFromEout, get_expansionOrder, Alpha
from solid_cinel.core.scattering_function.sab import get_SabSct, phonon_expansion
from solid_cinel.core.material.pdos import Pdos
from solid_cinel.core.material.tau import get_tauNbeta
from typing import Iterable
from math import pi
import warnings
from functools import cached_property

# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]

class DoubleDiffData:
    """
    Abstract class for Double Differential data.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the DoubleDiffData class.

        Parameters
        ----------
        args: Iterable
            The values for the pd.DataFrame
        kwargs: dict
            Optional arguments for the construction of the pd.DataFrame

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> dd.data
        Eout  1  2  3
        mu
        1     1  2  3
        2     4  5  6
        """
        self.data = pd.DataFrame(*args, **kwargs)

    @property
    def data(self) -> pd.DataFrame:
        """
        DataFrame

        Returns
        -------
        pd.Series
            Transfer function data
        """
        return self._data

    @data.setter
    def data(self, dd_pdf: Iterable):
        """
        Set the data and check the normalization.

        Parameters
        ----------
        dd_pdf : pd.Series
            Transfer function data

        """
        # Sort and define the style of the dataframe:
        dd_pdf_ = pd.DataFrame(dd_pdf).sort_index(axis=0).sort_index(axis=1)
        dd_pdf_.index.name = "mu"
        dd_pdf_.columns.name = "Eout"

        # Save the data:
        self._data = dd_pdf_

    @property
    def values(self) -> np.ndarray:
        """
        The values of the Double Differential data.

        Returns
        -------
        np.ndarray
            The values of the Double Differential data
        """
        return self.data.values

    @property
    def Eout(self) -> np.ndarray:
        """
        The outgoing energy grid.

        Returns
        -------
        np.ndarray
            The outgoing energy grid
        """
        return self.data.columns.values

    @property
    def mu(self) -> np.ndarray:
        """
        The cosine of the angle of the distribution in degrees.

        Returns
        -------
        np.ndarray
            The cosine of the angle of the distribution in degrees
        """
        return self.data.index.values

    def integrate(self, axis: int) -> pd.Series:
        """
        Integrate the Double Differential data.

        Parameters
        ----------
        axis: int
            The axis to integrate

        Returns
        -------
        pd.Series
            The integrated Double Differential data

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> dd.integrate(axis=0)
        Eout
        1    2.5
        2    3.5
        3    4.5
        dtype: float64

        >>> dd.integrate(axis=1)
        mu
        1     4.0
        2    10.0
        dtype: float64
        """
        return self.data.apply(integrate, axis=axis)

    @cached_property
    def rowIntegral(self) -> pd.Series:
        """
        Integral of the Double Differential data by row.

        Returns
        -------
        pd.Series
            Integral of the Double Differential data by row

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> dd.rowIntegral
                mu
        1     4.0
        2    10.0
        dtype: float64
        """
        return self.integrate(axis=1)

    @cached_property
    def columsIntegral(self) -> pd.Series:
        """
        Integral of the Double Differential data by column.

        Returns
        -------
        pd.Series
            Integral of the Double Differential data by column

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> dd.columsIntegral
        Eout
        1    2.5
        2    3.5
        3    4.5
        dtype: float64
        """
        return self.integrate(axis=0)

    @cached_property
    def doubleIntegral(self) -> float:
        """
        Double integral of the Double Differential data.

        Returns
        -------
        float
            Double integral of the Double Differential data

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> assert dd.doubleIntegral == 7.0
        """
        return integrate(self.rowIntegral)

    @property
    def pdf(self) -> pd.DataFrame:
        """
        Probability density function of the Double Differential data.

        Returns
        -------
        pd.Series
            Probability density function of the Double Differential data

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> dd.pdf
        Eout         1         2         3
        mu
        1     0.142857  0.285714  0.428571
        2     0.571429  0.714286  0.857143
        """
        return self.data / self.doubleIntegral

    @property
    def rowPdf(self) -> pd.Series:
        """
        Probability density function of the Double Differential data by row.

        Returns
        -------
        pd.Series
            Probability density function of the Double Differential data by row

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> dd.rowPdf
        mu
        1    0.571429
        2    1.428571
        dtype: float64
        """
        return self.integrate(axis=1) / self.doubleIntegral

    @property
    def columsPdf(self) -> pd.Series:
        """
        Probability density function of the Double Differential data by column.

        Returns
        -------
        pd.Series
            Probability density function of the Double Differential data by column

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> dd.columsPdf
        Eout
        1    0.357143
        2    0.500000
        3    0.642857
        dtype: float64
        """
        return self.integrate(axis=0) / self.doubleIntegral

    @property
    def cdf(self) -> pd.DataFrame:
        """
        Cumulative distribution function of the Dynamic Structure Factor.

        Returns
        -------
        pd.Series
            Cumulative distribution function of the Dynamic Structure Factor

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> dd.cdf
        Eout         1         2         3
        mu
        1     0.047619  0.142857  0.285714
        2     0.238095  0.571429  1.000000

        """
        cdf = self.data.cumsum(axis=0).cumsum(axis=1)
        return cdf / cdf.iloc[-1, -1]


class DynamicStruc(DoubleDiffData):
    """
    Dynamic structure factor class.
    """

    def __init__(self, Ein: float, T: float, M: float,  *args, **kwargs):
        """
        Initialize the DynamicStruc class.

        Parameters
        ----------
        Ein : float
            The neutron incident energy in eV
        T : float
            Temperature of the material in K
        M : float
            Mass of the material in amu
        args : Iterable, (N, M)
            The Transfer function data for the pd.DataFrame
        kwargs : dict
            Optional arguments for the construction of the pd.DataFrame
        """
        # Atributes of the Transfer function (Change in these parameters will
        # change the Transfer function):
        self.Ein = Ein
        self.T = T
        self.M = M

        # The Dynamic Structure data:
        super().__init__(*args, **kwargs)

        self.alphaMat = DoubleDiffData(
            get_alphaMat(self.Eout, self.Ein, self.T, self.M, self.mu)
        )


    def get_alphaMat(self, values=True) -> DoubleDiffData:
        """
        Return the $\alpha$ matrix for the Dynamic Structure Factor.

        Parameters
        ----------
        values: bool, optional
            If True return the values of the $\alpha$ matrix. If False return
            the $\alpha$ matrix as a pd.DataFrame. The default is True.

        Returns
        -------
        np.ndarray or pd.DataFrame
            The $\alpha$ matrix for the Dynamic Structure Factor
        """
        return self.alphaMat.values if values else self.alphaMat.data

    def recoil(self, values=True) -> [np.ndarray, pd.DataFrame]:
        """
        Return the recoil energy for the Dynamic Structure Factor.

        Parameters
        ----------
        values: bool, optional
            If True return the values of the recoil energy. If False return
            the recoil energy as a pd.DataFrame. The default is True.

        Returns
        -------
        np.ndarray or pd.DataFrame
            The recoil energy for the Dynamic Structure Factor
        """
        return self.get_alphaMat(values) * kb * self.T

    @property
    def alpha0(self) -> float:
        """
        The $\alpha_0$ parameter of the Dynamic Structure Factor.

        Returns
        -------
        float
            The $\alpha_0$ parameter of the Dynamic Structure Factor

        Examples
        --------
        >>> from solid_cinel.data.examples.UO2 import rho_in_energy_U238, interv_in_energy_U238
        >>> Ein = 7.2
        >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
        >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([40, 80, 120, 160])
        >>> mu = np.cos(np.deg2rad(theta))
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> float(round(DynamicStruc.from_pdos(Ein, M, T, Eout, mu, pdos, threshold=1.0e-14).alpha0, 6))
        0.328006
        """
        # Get the alpha0 parameter:
        return integrate((self.data * self.alphaMat.values).apply(integrate)) / 2

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
        Return the Transference function of the Dynamic Structure Factor.

        Returns
        -------
        pd.Series
            The transfer function

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
        >>> mu = np.cos(np.deg2rad(theta))
        >>> dynamicStructure = DynamicStruc.from_fgm(Ein, M, T, Eout, mu)
        >>> dynamicStructure.transferFunc.round(6)
        Eout
        6.7554    0.031423
        6.9050    0.404638
        7.0439    1.888728
        7.2000    4.340047
        7.3157    0.688071
        7.4480    0.045257
        dtype: float64
        """
        return super().columsIntegral

    @property
    def angleDist(self):
        """
        Return the angle distribution of the Dynamic Structure Factor.

        Returns
        -------
        pd.Series
            The angle distribution

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
        >>> mu = np.cos(np.deg2rad(theta))
        >>> dynamicStructure = DynamicStruc.from_fgm(Ein, M, T, Eout, mu)
        >>> dynamicStructure.angleDist.round(6)
        mu
        -9.659258e-01    0.480322
        -8.660254e-01    0.482041
        -7.071068e-01    0.484205
        -5.000000e-01    0.485923
        -2.588190e-01    0.486289
         6.123234e-17    0.484720
         2.588190e-01    0.481267
         5.000000e-01    0.479932
         7.071068e-01    0.515987
         8.660254e-01    0.714575
         9.659258e-01    1.435551
        dtype: float64
        """
        return super().rowIntegral

    @classmethod
    def from_fgm(cls, Ein: float, M: float, T: float, Eout: np.ndarray,
                 mu: np.ndarray, ws: float = 1.0):
        """
        Generate the Dynamic Structure Factor from a S(alpha, -beta) table based
        on Free Gas Model.

        Parameters
        ----------
        Ein : float
            The incident energy of the neutron in eV
        M : float
            The mass of the target material in amu
        T : float
            Temperature of the material in K
        Eout : np.array
            The neutron outgoing energy grid in eV
        mu : np.ndarray
            The cosine of the angle of the distribution in degrees
        ws: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Returns
        -------
        DynamicStruc
            Dynamic Structure Factor from a S(alpha, -beta) table based on Free
            Gas Model

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
        >>> mu = np.cos(np.deg2rad(theta))
        >>> DynamicStruc.from_fgm(Ein, M, T, Eout, mu).data.round(6)
        Eout             6.7554    6.9050    7.0439     7.2000    7.3157    7.4480
        mu
        -9.659258e-01  0.093290  0.635800  1.344517   0.987905  0.366598  0.054415
        -8.660254e-01  0.074800  0.591841  1.360299   1.032095  0.376520  0.052584
        -7.071068e-01  0.049539  0.515196  1.379332   1.109853  0.392419  0.049014
        -5.000000e-01  0.024994  0.404827  1.387900   1.228207  0.412767  0.043015
        -2.588190e-01  0.008241  0.268643  1.360778   1.399190  0.433942  0.033969
         6.123234e-17  0.001317  0.132279  1.255634   1.643445  0.447804  0.022111
         2.588190e-01  0.000054  0.036774  1.013814   1.998435  0.436944  0.009862
         5.000000e-01  0.000000  0.002991  0.600838   2.539266  0.368245  0.001932
         7.071068e-01  0.000000  0.000010  0.155387   3.441598  0.204433  0.000045
         8.660254e-01  0.000000  0.000000  0.002062   5.233842  0.024125  0.000000
         9.659258e-01  0.000000  0.000000  0.000000  10.563289  0.000000  0.000000
        """
        # Get the Dynamic Structure Factor values:
        scatfunc = get_ScatSctAngular(Eout, mu, Ein, T, M, T, ws)

        return cls(Ein, T, M, scatfunc, index=mu, columns=Eout)

    @classmethod
    def from_sct(cls, Ein: float, M: float, T: float, Eout: np.ndarray,
                 mu: np.ndarray, pdos: Pdos, ws: float = 1.0):
        """
        Generate tDynamic Structure Factor from a S(alpha, -beta) table based on
        Short Collision Time model.

        Parameters
        ----------
        Ein : float
            The incident energy of the neutron in eV
        M : float
            The mass of the target material in amu
        T : float
            Temperature of the material in K
        Eout : np.array
            The neutron outgoing energy grid in eV
        mu : np.ndarray
            The cosine of the angle of the distribution in degrees
        pdos: 'solid_cinel.core.material.Pdos'
            Pdos object.
        ws: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Returns
        -------
        DynamicStruc
            Dynamic Structure Factor from a S(alpha, -beta) table based on Short
            Collision Time model

        Examples
        --------
        >>> from solid_cinel.data.examples.UO2 import rho_in_energy_U238, interv_in_energy_U238
        >>> Ein = 7.2
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
        >>> mu = np.cos(np.deg2rad(theta))
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> DynamicStruc.from_sct(Ein, M, T, Eout, mu, pdos).data.round(6)
        Eout             6.7554    6.9050    7.0439     7.2000    7.3157    7.4480
        mu
        -9.659258e-01  0.094001  0.636412  1.342345   0.987382  0.367669  0.054937
        -8.660254e-01  0.075434  0.592611  1.358169   1.031486  0.377620  0.053100
        -7.071068e-01  0.050039  0.516194  1.377318   1.109089  0.393570  0.049515
        -5.000000e-01  0.025312  0.406041  1.386155   1.227206  0.413997  0.043483
        -2.588190e-01  0.008381  0.269913  1.359573   1.397842  0.435292  0.034377
         6.123234e-17  0.001348  0.133285  1.255372   1.641602  0.449328  0.022419
         2.588190e-01  0.000056  0.037238  1.014880   1.995877  0.438696  0.010033
         5.000000e-01  0.000000  0.003057  0.602973   2.535640  0.370193  0.001978
         7.071068e-01  0.000000  0.000011  0.156817   3.436247  0.206125  0.000047
         8.660254e-01  0.000000  0.000000  0.002116   5.225195  0.024538  0.000000
         9.659258e-01  0.000000  0.000000  0.000000  10.545191  0.000000  0.000000
        """
        # Get the effective temperature:
        Teff = pdos.fix_T(T).Teff

        # Get the Dynamic Structure Factor values:
        scatfunc = get_ScatSctAngular(Eout, mu, Ein, T, M, Teff, ws)

        return cls(Ein, T, M, scatfunc, index=mu, columns=Eout)

    @classmethod
    def from_tau(cls, Ein: float, M: float, T: float, Eout: np.ndarray,
                 mu: np.ndarray, tauN: np.ndarray, tauNbeta: np.ndarray,
                 DebyeWallerCoeff: float):
        """
        Generate the Dynamic Structure Factor from tauN function using the
        phonon expansion model.

        Parameters
        ----------
        Ein : float
            The incident energy of the neutron in eV
        M : float
            The mass of the target material in amu
        T : float
            Temperature of the material in K
        Eout : np.ndarray, (M,)
            The neutron outgoing energy grid in eV
        theta : np.ndarray, (N,)
            Grid of angle of the scattering angle
        tauN: np.ndarray, (Z, T)
            tauN function. Z is the number of phonon expansion order and T is
            the number of beta grid points.
        tauNbeta: np.ndarray, (T,)
            Beta grid for the tauN function
        DebyeWallerCoeff: float
            Debye-Waller coefficient in LEAPR formalism

        Returns
        -------
        DynamicStruc
            Dynamic Structure Factor

        Examples
        --------
        >>> from solid_cinel.data.examples.UO2 import rho_in_energy_U238, interv_in_energy_U238
        >>> Ein = 7.2
        >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
        >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([40, 80, 120, 160])
        >>> mu = np.cos(np.deg2rad(theta))
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> DebyeWallerCoeff = pdos.DebyeWallerCoeff
        >>> nphonon = get_expansionOrder(get_alphaFromEout(Eout, Ein, M, T, mu.min()), DebyeWallerCoeff, 1.0e-6, 5000)
        >>> tauN = pdos.tauN(nphonon, 1.0e-14, values=True)
        >>> tauNbeta = get_tauNbeta(pdos.beta.data, tauN.shape[1])
        >>> DynamicStruc.from_tau(Ein, M, T, Eout, mu, tauN, tauNbeta, DebyeWallerCoeff).data.loc[::, Eout_test].round(6)
        Eout         6.7554    6.9050    7.0439    7.2000    7.3157    7.4480
        mu
        -0.939693  0.109061  0.644157  1.346117  1.029210  0.373643  0.053219
        -0.500000  0.034511  0.426488  1.383082  1.262613  0.415630  0.042074
         0.173648  0.000519  0.073364  1.103240  1.912878  0.440892  0.013328
         0.766044  0.000000  0.000012  0.077506  4.022814  0.127645  0.000019
        """
        # Get the Dynamic Structure Factor values with correct type:
        scatfunc = calc_DynStrucClm(Ein, M, T, Eout, mu, tauN, tauNbeta, DebyeWallerCoeff)

        return cls(Ein, T, M, scatfunc, index=mu, columns=Eout)

    @classmethod
    def from_pdos(cls, Ein: float, M: float, T: float, Eout: np.ndarray,
                  mu: np.ndarray, pdos: Pdos, nphonon: int = None,
                  decimal: float = 1.0e-6,
                  order_max: int = 5000, threshold: float = 0.0):
        """
        Generate the Dynamic Structure Factor from a S(alpha, -beta) table based
        on Phonon expansion model.

        Parameters
        ----------
        Ein : float
            The incident energy of the neutron in eV
        M : float
            The mass of the target material in amu
        T : float
            Temperature of the material in K
        Eout : np.array
            The neutron outgoing energy grid in eV
        mu : np.ndarray
            The cosine of the angle of the distribution in degrees
        pdos: 'solid_cinel.core.material.Pdos'
            Pdos object.
        nphonon: 'int', optional
            Phonon expansion order. The default is None and the order is
            calculated using the get_expansionOrder function.
        decimal: 'float', optional
            Decimal precision for the calculation of the expansion order.
            The default is 1.0e-6.
        order_max: 'int', optional
            Maximun expansion order. The default is 5000.
        threshold: 'float', optional
            Minimun value to take into account in the creation of tauN
            functions

        Returns
        -------
        DynamicStruc
            Dynamic Structure Factor from a S(alpha, -beta) table based on
            Phonon expansion model.

        Examples
        --------
        >>> from solid_cinel.data.examples.UO2 import rho_in_energy_U238, interv_in_energy_U238
        >>> Ein = 7.2
        >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
        >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([40, 80, 120, 160])
        >>> mu = np.cos(np.deg2rad(theta))
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> DynamicStruc.from_pdos(Ein, M, T, Eout, mu, pdos, threshold=1.0e-14).data.loc[::, Eout_test].round(6)
        Eout         6.7554    6.9050    7.0439    7.2000    7.3157    7.4480
        mu
        -0.939693  0.109061  0.644157  1.346117  1.029210  0.373643  0.053219
        -0.500000  0.034511  0.426488  1.383082  1.262613  0.415630  0.042074
         0.173648  0.000519  0.073364  1.103240  1.912878  0.440892  0.013328
         0.766044  0.000000  0.000012  0.077506  4.022814  0.127645  0.000019
        """
        # Get Tpdos:
        Tpdos = pdos.fix_T(T)

        # Get the Debye-Waller coefficient:
        DebyeWallerCoeff = Tpdos.DebyeWallerCoeff

        # Get the expansion order:
        if nphonon:
            warnings.warn(
                "Is posible that the expansion order is not enough to get the correct results")
        else:
            alphaMax = get_alphaFromEout(Eout, Ein, M, T, mu.min())
            nphonon = get_expansionOrder(alphaMax, DebyeWallerCoeff, decimal, order_max)

        # Get tauN function:
        tauN = Tpdos.tauN(nphonon, threshold, values=True)
        tauNbeta = get_tauNbeta(Tpdos.beta.data, tauN.shape[1])

        # Get the Dynamic Structure Factor values:
        return cls.from_tau(Ein, M, T, Eout, mu, tauN, tauNbeta, DebyeWallerCoeff)

    @classmethod
    def from_model(cls, Ein: float, M: float, T: float, Eout: np.ndarray,
                   theta: np.ndarray, *args, model: str = "fgm", **kwargs):
        """
        Generate Dynamic Structure Factor from a S(alpha, -beta) table.
        ..math::
        S(\theta, E^\prime, E, M, T) = \frac{1}{2 * k_B * T}\sqrt{\frac{^\prime}{E}} S(\alpha(\theta, E^\prime, E, M, T), \beta( E^\prime, E, T))

        Parameters
        ----------
        Ein : float
            The incident energy of the neutron in eV
        M : float
            The mass of the target material in amu
        T : float
            Temperature of the material in K
        Eout : np.ndarray
            The neutron outgoing energy grid in eV
        theta : np.ndarray
            Grid of angle of the scattering angle
        model: str
            The model used to generate the S(alpha, beta) table. The available
            models are:
                - "pdos": Phonon expansion model
                - "fgm" : Free Gas Model (Default)
                - "sct" : Short Collision Time model

        Parameters for SCT model
        ------------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        ws: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.
        twt: 'float', optional
            twt for the effective temperature. For solid is 1.

        Parameters for PDOS model
        -------------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        nphonon: 'int', optional
            Phonon expansion order. The default is None and the order is
            calculated using the get_expansionOrder function.
        decimal: 'float', optional
            Decimal precision for the calculation of the expansion order.
            The default is 1.0e-6.
        order_max: 'int', optional
            Maximun expansion order. The default is 5000.
        threshold: 'float', optional
            Minimun value to take into account in the creation of tauN
            functions

        Returns
        -------
        DynamicStruc
            Dynamic Structure Factor from a S(alpha, -beta) table.

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])

        # Using the Free Gas Model:
        >>> DynamicStruc.from_model(Ein, M, T, Eout, theta, model="fgm").data.round(6)
        Eout             6.7554    6.9050    7.0439     7.2000    7.3157    7.4480
        mu
        -9.659258e-01  0.093290  0.635800  1.344517   0.987905  0.366598  0.054415
        -8.660254e-01  0.074800  0.591841  1.360299   1.032095  0.376520  0.052584
        -7.071068e-01  0.049539  0.515196  1.379332   1.109853  0.392419  0.049014
        -5.000000e-01  0.024994  0.404827  1.387900   1.228207  0.412767  0.043015
        -2.588190e-01  0.008241  0.268643  1.360778   1.399190  0.433942  0.033969
         6.123234e-17  0.001317  0.132279  1.255634   1.643445  0.447804  0.022111
         2.588190e-01  0.000054  0.036774  1.013814   1.998435  0.436944  0.009862
         5.000000e-01  0.000000  0.002991  0.600838   2.539266  0.368245  0.001932
         7.071068e-01  0.000000  0.000010  0.155387   3.441598  0.204433  0.000045
         8.660254e-01  0.000000  0.000000  0.002062   5.233842  0.024125  0.000000
         9.659258e-01  0.000000  0.000000  0.000000  10.563289  0.000000  0.000000

        # Using the Short Collision Time model:
        >>> from solid_cinel.data.examples.UO2 import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> DynamicStruc.from_model(Ein, M, T, Eout, theta, pdos, model="sct").data.round(6)
        Eout             6.7554    6.9050    7.0439     7.2000    7.3157    7.4480
        mu
        -9.659258e-01  0.094001  0.636412  1.342345   0.987382  0.367669  0.054937
        -8.660254e-01  0.075434  0.592611  1.358169   1.031486  0.377620  0.053100
        -7.071068e-01  0.050039  0.516194  1.377318   1.109089  0.393570  0.049515
        -5.000000e-01  0.025312  0.406041  1.386155   1.227206  0.413997  0.043483
        -2.588190e-01  0.008381  0.269913  1.359573   1.397842  0.435292  0.034377
         6.123234e-17  0.001348  0.133285  1.255372   1.641602  0.449328  0.022419
         2.588190e-01  0.000056  0.037238  1.014880   1.995877  0.438696  0.010033
         5.000000e-01  0.000000  0.003057  0.602973   2.535640  0.370193  0.001978
         7.071068e-01  0.000000  0.000011  0.156817   3.436247  0.206125  0.000047
         8.660254e-01  0.000000  0.000000  0.002116   5.225195  0.024538  0.000000
         9.659258e-01  0.000000  0.000000  0.000000  10.545191  0.000000  0.000000


        # Using the Phonon expansion model:
        >>> Ein = 7.2
        >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
        >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([40, 80, 120, 160])

        >>> DynamicStruc.from_model(Ein, M, T, Eout, theta, pdos, threshold=1.0e-14, model="pdos").data.loc[::, Eout_test].round(6)
        Eout         6.7554    6.9050    7.0439    7.2000    7.3157    7.4480
        mu
        -0.939693  0.109061  0.644157  1.346117  1.029210  0.373643  0.053219
        -0.500000  0.034511  0.426488  1.383082  1.262613  0.415630  0.042074
         0.173648  0.000519  0.073364  1.103240  1.912878  0.440892  0.013328
         0.766044  0.000000  0.000012  0.077506  4.022814  0.127645  0.000019
        """
        # Get the cosine of the angle of the distribution:
        mu = np.cos(np.deg2rad(theta))

        # Get the Transfer function:
        if model.lower() == "pdos":
            return cls.from_pdos(Ein, M, T, Eout, mu, *args, **kwargs)
        elif model.lower() == "sct":
            return cls.from_sct(Ein, M, T, Eout, mu, *args, **kwargs)
        else:
            return cls.from_fgm(Ein, M, T, Eout, mu)



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
    return exponetials * np.sqrt(AkbT / pi) * EoutSqrt / Ein / 2


@nb.jit(float64[:](float64[:], float64, float64, float64),
        nopython=True, cache=True)
def normFactor(Eout: np.ndarray, Ein: float, T: float, M: float) -> np.ndarray:
    """
    Normalization factor for the Transfer function calculation.

    Parameters
    ----------
    Eout: 'np.ndarray', (N,)
        Outgoing energy grid in eV.
    Ein: 'float'
        Incident energy in eV.
    T: 'float'
        Temperature in K.
    M: 'float'
        Mass of the target in amu.

    Returns
    -------
    'np.ndarray', (N,)
        Normalization factor for the Transfer function calculation.
    """
    M_div_m = M / m
    aws = ((M_div_m + 1) / M_div_m) ** 2
    two_kb_T = 2 * kb * T
    return aws * np.sqrt(Eout / Ein) / two_kb_T


@nb.jit(float64[:, :](float64[:], float64[:], float64, float64,
                      float64, float64, float64),
    nopython=True, cache=True)
def get_ScatSctAngular(Eout: np.ndarray, mu: [float, np.ndarray], Ein: float,
                       T: float, M: float, Teff: float, ws: float) -> np.ndarray:
    """
    Calculate the Transfer function from the Short Collision Time model using
    a single angle.
    ..math::
        S(\theta, E^\prime, E, M, T) = \frac{1}{2 * k_B * T}\sqrt{\frac{E^\prime}{E}} \frac{1}{\sqrt{4 \pi w_s \alpha T_{eff} / T}} exp\left(\frac{(w_s\alpha +\beta)^2}{4 \alpha w_s T_{eff}/T}\right)

    Parameters
    ----------
    Eout : np.ndarray, (N,)
        The neutron outgoing energy grid in eV
    mu : np.ndarray, (M,)
        Cosine of the angle between the incident neutron direction and
        the outgoing neutron direction
    Ein : float
        The incident energy of the neutron in eV
    T : float
        Temperature of the material in K
    M : float
        The mass of the target material in amu
    Teff : float
        Effective temperature of the material in K
    ws : float
        Normalization for continuous (vibrational) part. For solid is 1.

    Returns
    -------
    np.ndarray, (M, N)
        The Transfer function values for a single angle
    """
    # Get the beta grid:
    beta = calc_Beta(Eout, Ein, T)

    # Get the temperature ratio:
    Tratio = Teff / T

    # Get the alpha grid:
    if isinstance(mu, (int, float)):
        alpha = get_alphaFromEout(Eout, Ein, T, M, mu)[::, np.newaxis]
    else:
        alpha = get_alphaMat(Eout, Ein, T, M, mu)

    # Get the Transfer function values and apply normalization:
    return get_SabSct(alpha, beta, Tratio, ws) * normFactor(Eout, Ein, T, M)


@nb.jit(float64[:, :](float64, float64, float64, float64[:], float64[:],
                      float64[:, :], float64[:], float64),
        nopython=True, cache=True)
def calc_DynStrucClm(Ein: float, M: float, T: float, Eout: np.ndarray, mu: np.ndarray,
                    tauN: np.ndarray, tauNbeta: np.ndarray,
                    DebyeWallerCoeff: float) -> np.ndarray:
    """
    Generate the Transfer function from a S(alpha, -beta) table based on
    the phonon expansion model.

    Parameters
    ----------
    Ein : float
        The incident energy of the neutron in eV
    M : float
        The mass of the target material in amu
    T : float
        Temperature of the material in K
    Eout : np.ndarray, (N,)
        The neutron outgoing energy grid in eV
    mu : float
        Cosine of the scattering angle
    tauN : 'np.ndarray', (M, T)
        all tau n functions in one array.
    tauNbeta : 'np.ndarray', (M,)
        Space between beta grid points of tau n functions.
    DebyeWallerCoeff : float
        Debye Waller coefficient

    Returns
    -------
    S_diag : 'np.ndarray', (N,)
        Transfer function values for a single angle.

    Examples
    --------
    >>> from solid_cinel.data.examples.UO2 import rho_in_energy_U238, interv_in_energy_U238
    >>> Ein = 7.2
    >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
    >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
    >>> T = 1000.0
    >>> M = 238.05077040419212
    >>> mu = np.cos(np.deg2rad([120]))
    >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
    >>> DebyeWallerCoeff = pdos.DebyeWallerCoeff
    >>> nphonon = get_expansionOrder(get_alphaFromEout(Eout, Ein, M, T, mu), DebyeWallerCoeff, 1.0e-6, 5000)
    >>> tauN = pdos.tauN(nphonon, 1.0e-14, values=True)
    >>> tau1beta = pdos.beta.data
    >>> tauNbeta = get_tauNbeta(tau1beta, tauN.shape[1])
    >>> sd_pdf = calc_DynStrucClm(Ein, M, T, Eout, mu, tauN, tauNbeta, DebyeWallerCoeff)
    >>> pd.Series(sd_pdf[0], index=Eout).loc[Eout_test].round(6)
    6.7554    0.034510
    6.9050    0.426488
    7.0439    1.383081
    7.2000    1.262613
    7.3157    0.415630
    7.4480    0.042074
    dtype: float64
    """
    # Get the beta grid:
    betaAbs = get_AbsBeta(Eout, Ein, T)

    # Eout calculation for the absulote beta values:
    EoutCalc = np.sort(Ein + np.concatenate((-betaAbs[::-1], betaAbs[1::])) * kb * T)

    # Ensure the Eout values are positive:
    positiveMask = EoutCalc > 0
    EoutCalc = EoutCalc[positiveMask]

    # Interpolation of tauN functions to reduce the number of calculations:
    tauNinterp = interp_multyParallel(betaAbs, tauNbeta, tauN)

    # Get the S(alpha, -beta) values for the alpha and beta combinations:
    # Correct alpha values for absolute beta values:
    sabValues = phonon_expansion(get_alphaMat(Eout, Ein, T, M, mu),
                                 tauN.shape[0],  # number of phonons
                                 tauNinterp, DebyeWallerCoeff)

    # Full Dynamic Structure factor values calculation:
    dynamicStruc = np.concatenate(
        (sabValues[::, ::-1], sabValues[::, 1:] * np.exp(-betaAbs[1:])), axis=1
    )[::, positiveMask]

    # Normalization constant
    dynamicStruc *= normFactor(EoutCalc, Ein, T, M)

    # Interpolation for avoiding numerical fluctuations:
    return interp_multyParallel(Eout, EoutCalc, dynamicStruc)

@nb.jit(float64[:, :](float64, float64, float64, float64[:], float64[:],
                      float64[:, :], float64[:], float64, float64),
        nopython=True, cache=True)
def get_ScatFuncClm(Ein: float, M: float, T: float, Eout: np.ndarray, mu: np.ndarray,
                    tauN: np.ndarray, tauNbeta: np.ndarray,
                    DebyeWallerCoeff: float, alpha0: float) -> np.ndarray:
    """
    Generate the Transfer function from a S(alpha, -beta) table based on
    the phonon expansion model.

    Parameters
    ----------
    Ein : float
        The incident energy of the neutron in eV
    M : float
        The mass of the target material in amu
    T : float
        Temperature of the material in K
    Eout : np.ndarray, (N,)
        The neutron outgoing energy grid in eV
    mu : float
        Cosine of the scattering angle
    tauN : 'np.ndarray', (M, T)
        all tau n functions in one array.
    tauNbeta : 'np.ndarray', (M,)
        Space between beta grid points of tau n functions.
    DebyeWallerCoeff : float
        Debye Waller coefficient

    Returns
    -------
    S_diag : 'np.ndarray', (N,)
        Transfer function values for a single angle.

    Examples
    --------
    >>> from solid_cinel.data.examples.UO2 import rho_in_energy_U238, interv_in_energy_U238
    >>> Ein = 7.2
    >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
    >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
    >>> T = 1000.0
    >>> M = 238.05077040419212
    >>> mu = np.cos(np.deg2rad([120]))
    >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
    >>> DebyeWallerCoeff = pdos.DebyeWallerCoeff
    >>> nphonon = get_expansionOrder(get_alphaFromEout(Eout, Ein, M, T, mu), DebyeWallerCoeff, 1.0e-6, 5000)
    >>> tauN = pdos.tauN(nphonon, 1.0e-14, values=True)
    >>> tau1beta = pdos.beta.data
    >>> tauNbeta = get_tauNbeta(tau1beta, tauN.shape[1])

    # Using the alpha0 parameter:
    >>> alpha0 = Ein / M / (kb * T)
    >>> sd_pdf = get_ScatFuncClm(Ein, M, T, Eout, mu, tauN, tauNbeta, DebyeWallerCoeff, alpha0)
    >>> pd.Series(sd_pdf[0], index=Eout).loc[Eout_test].round(6)
    6.7554    0.000002
    6.9050    0.005037
    7.0439    0.605736
    7.2000    2.560414
    7.3157    0.360692
    7.4480    0.002042
    dtype: float64
    """
    # Get the beta grid:
    beta = get_AbsBeta(Eout, Ein, T)

    # Eout calculation
    EoutCalc = np.sort(Ein + np.concatenate((-beta[::-1], beta[1::])) * kb * T)

    # Ensure the Eout values are positive:
    positiveMask = EoutCalc > 0
    EoutCalc = EoutCalc[positiveMask]

    # Interpolation of tauN functions to reduce the number of calculations:
    tauNinterp = interp_multyParallel(beta, tauNbeta, tauN)

    # Get the alpha matrix for the Dynamic Structure Factor with the maximun
    # outgoing energy:
    Eout_ = beta * kb * T + Ein if len(beta) < len(Eout) else Eout

    # Get the S(alpha, -beta) values for the alpha and beta combinations:
    sabValues = phonon_expansion(get_alphaMatMod(Eout_, Ein, M, T, mu, DebyeWallerCoeff, alpha0),
                                 tauN.shape[0], tauNinterp, DebyeWallerCoeff)

    # Full Dynamic Structure factor values calculation:
    dynamicStruc = np.concatenate((sabValues[::, ::-1], sabValues[::, 1:] * np.exp(-beta[1:])), axis=1)[::, positiveMask]

    # Normalization constant
    dynamicStruc *= normFactor(EoutCalc, Ein, T, M)

    # Interpolation for avoiding numerical fluctuations:
    return interp_multyParallel(Eout, EoutCalc, dynamicStruc)