"""
Python file for working with scattering functions.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
from numba import vectorize
from scipy.constants import physical_constants as const
from solid_cinel.core.generic import integrate, interp_multyParallel
from solid_cinel.core.scattering_function.beta import get_beta, Beta
from solid_cinel.core.scattering_function.alpha import get_alphaMat, get_alphaMatMod, get_alphaFromEout, get_expansionOrder, Alpha
from solid_cinel.core.scattering_function.sab import get_SabSct, get_SabSctAlpha, Sab
from solid_cinel.core.material.vibration.pdos import Pdos
from solid_cinel.core.material.vibration.tau import get_tauNbeta
from typing import Iterable
from math import sqrt, pi, exp
import warnings

# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]


class ScatFunc:
    """
    Double Differencial (angle, Outgoing energy) scattering function base
    class.
    """

    def __init__(self, Ein: float, T: float, M: float,  *args, **kwargs):
        """
        Initialize the TransferFunc class.

        Parameters
        ----------
        Ein : float
            The neutron incident energy in eV
        T : float
            Temperature of the material in K
        M : float
            Mass of the material in amu
        args : Iterable, (N, M)
            The scattering function data for the pd.DataFrame
        kwargs : dict
            Optional arguments for the construction of the pd.DataFrame
        """
        # Atributes of the scattering function (Change in these parameters will
        # change the scattering function):
        self.Ein = Ein
        self.T = T
        self.M = M

        # The scattering function data:
        self.data = pd.DataFrame(*args, **kwargs)

    @property
    def data(self) -> pd.DataFrame:
        """
        Scattering function data.

        Returns
        -------
        pd.Series
            The scattering function data
        """
        return self._data

    @data.setter
    def data(self, dd_pdf: Iterable):
        """
        Set the scattering function data and check the normalization.

        Parameters
        ----------
        dd_pdf : pd.Series
            Double differential scattering function data

        """
        # Sort and define the style of the dataframe:
        dd_pdf_ = pd.DataFrame(dd_pdf).sort_index(axis=0).sort_index(axis=1)
        dd_pdf_.index.name = "mu"
        dd_pdf_.columns.name = "Eout"

        # Erase the columns with all zeros:
        dd_pdf_ = dd_pdf_.loc[::, ~dd_pdf_.eq(0).all()]

        # Save the data:
        self._data = dd_pdf_

    @classmethod
    def from_model(cls, Ein: float, M: float, T: float, Eout: np.ndarray,
                   theta: np.ndarray, *args, model: str = "fgm", **kwargs):
        """
        Generate the double differential scattering function from a
         S(alpha, -beta) table.
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
        TransferFunc
            Double differential scattering scattering function

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])

        # Using the Free Gas Model:
        >>> ScatFunc.from_model(Ein, M, T, Eout, theta, model="fgm").data.round(6)
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
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> ScatFunc.from_model(Ein, M, T, Eout, theta, pdos, model="sct").data.round(6)
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

        >>> ScatFunc.from_model(Ein, M, T, Eout, theta, pdos, threshold=1.0e-14, model="pdos").data.loc[::, Eout_test].round(6)
        Eout         6.7554    6.9050    7.0439    7.2000    7.3157    7.4480
        mu
        -0.939693  0.109061  0.644157  1.346117  1.029210  0.373643  0.053219
        -0.500000  0.034511  0.426488  1.383082  1.262613  0.415630  0.042074
         0.173648  0.000519  0.073364  1.103240  1.912878  0.440892  0.013328
         0.766044  0.000000  0.000012  0.077506  4.022814  0.127645  0.000019
        """
        # Get the cosine of the angle of the distribution:
        mu = np.cos(np.deg2rad(theta))

        # Get the scattering function:
        if model.lower() == "pdos":
            return cls.from_pdos(Ein, M, T, Eout, mu, *args, **kwargs)
        elif model.lower() == "sct":
            return cls.from_sct(Ein, M, T, Eout, mu, *args, **kwargs)
        else:
            return cls.from_fgm(Ein, M, T, Eout, mu)

    @classmethod
    def from_pdos(cls, Ein: float, M: float, T: float, Eout: np.ndarray,
                  mu: np.ndarray, pdos: Pdos, nphonon: int = None,
                  decimal: float = 1.0e-6,
                  order_max: int = 5000, threshold: float = 0.0):
        """
        Generate the double differential scattering function from a
        S(alpha, -beta) table based on Phonon expansion model.

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
        ScatFunc
            Double differential scattering function from a S(alpha, -beta) table
            based on Phonon expansion model.

        Examples
        --------
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> Ein = 7.2
        >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
        >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([40, 80, 120, 160])
        >>> mu = np.cos(np.deg2rad(theta))
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> ScatFunc.from_pdos(Ein, M, T, Eout, mu, pdos, threshold=1.0e-14).data.loc[::, Eout_test].round(6)
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

        # Get the scattering fucntion values:
        return cls.from_tau(Ein, M, T, Eout, mu, tauN, tauNbeta,
                            DebyeWallerCoeff)

    @classmethod
    def from_sct(cls, Ein: float, M: float, T: float, Eout: np.ndarray,
                 mu: np.ndarray, pdos: Pdos, ws: float = 1.0):
        """
        Generate the double differential scattering function from a
        S(alpha, -beta) table based on Short Collision Time model.

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
        ScatFunc
            Double differential scattering function from a S(alpha, -beta) table
            based on Short Collision Time model

        Examples
        --------
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> Ein = 7.2
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
        >>> mu = np.cos(np.deg2rad(theta))
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> ScatFunc.from_sct(Ein, M, T, Eout, mu, pdos).data.round(6)
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

        # Get the scattering fucntion values:
        scatfunc = get_ScatSctAngular(Eout, mu, Ein, T, M, Teff, ws)

        return cls(Ein, T, M, scatfunc, index=mu, columns=Eout)

    @classmethod
    def from_fgm(cls, Ein: float, M: float, T: float, Eout: np.ndarray,
                 mu: np.ndarray, ws: float = 1.0):
        """
        Generate the double differential scattering function from a
        S(alpha, -beta) table based on Free Gas Model.

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
        ScatFunc
            Double differential scattering function from a S(alpha, -beta) table
            based on Free Gas Model

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
        >>> mu = np.cos(np.deg2rad(theta))
        >>> ScatFunc.from_fgm(Ein, M, T, Eout, mu).data.round(6)
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
        # Get the scattering fucntion values:
        scatfunc = get_ScatSctAngular(Eout, mu, Ein, T, M, T, ws)

        return cls(Ein, T, M, scatfunc, index=mu, columns=Eout)

    @classmethod
    def from_tau(cls, Ein: float, M: float, T: float, Eout: np.ndarray,
                 mu: np.ndarray, tauN: np.ndarray, tauNbeta: np.ndarray,
                 DebyeWallerCoeff: float):
        """
        Generate the double differential scattering function from tauN function
        using the phonon expansion model.

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
        TransferFunc
            Double differential scattering function

        Examples
        --------
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
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
        >>> ScatFunc.from_tau(Ein, M, T, Eout, mu, tauN, tauNbeta, DebyeWallerCoeff).data.loc[::, Eout_test].round(6)
        Eout         6.7554    6.9050    7.0439    7.2000    7.3157    7.4480
        mu
        -0.939693  0.109061  0.644157  1.346117  1.029210  0.373643  0.053219
        -0.500000  0.034511  0.426488  1.383082  1.262613  0.415630  0.042074
         0.173648  0.000519  0.073364  1.103240  1.912878  0.440892  0.013328
         0.766044  0.000000  0.000012  0.077506  4.022814  0.127645  0.000019
        """
        # Get the scattering fucntion values:
        scatfunc = get_ScatFuncClm(Ein, M, T, Eout, mu, tauN, tauNbeta,
                                   DebyeWallerCoeff)

        return cls(Ein, T, M, scatfunc, index=mu, columns=Eout)

    @property
    def to_transferFunc(self):
        """
        Return the TransferFunc object.

        Returns
        -------
        TransferFunc
            The TransferFunc object

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
        >>> mu = np.cos(np.deg2rad(theta))
        >>> scatfunc = ScatFunc.from_fgm(Ein, M, T, Eout, mu)
        >>> scatfunc.to_transferFunc.data.round(6)
        Eout
        6.7554    0.031423
        6.9050    0.404638
        7.0439    1.888728
        7.2000    4.340047
        7.3157    0.688071
        7.4480    0.045257
        dtype: float64
        """
        return TransferFunc(self.Ein, self.T, self.M, self.data.apply(integrate))
    @property
    def alpha0(self) -> float:
        """
        The alpha0 parameter of the scattering function.

        Returns
        -------
        float
            The alpha0 parameter of the scattering function

        Examples
        --------
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> Ein = 7.2
        >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
        >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([40, 80, 120, 160])
        >>> mu = np.cos(np.deg2rad(theta))
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> float(round(ScatFunc.from_pdos(Ein, M, T, Eout, mu, pdos, threshold=1.0e-14).alpha0, 6))
        0.328006
        """
        # Get the scattering fucntion:
        scatfunc = self.data

        # Get the alpha matrix:
        alphaMat = get_alphaMat(scatfunc.columns.values, self.Ein, self.T, self.M,
                                scatfunc.index.values)

        # Get the alpha0 parameter:
        return integrate((scatfunc * alphaMat).apply(integrate)) / 2

    @property
    def norm(self) -> float:
        """
        Normalization of the scattering function.

        Returns
        -------
        float
            Normalization of the scattering function
        """
        return integrate(self.data.apply(integrate))

    @property
    def pdf(self) -> pd.DataFrame:
        """
        Probability density function of the scattering function.

        Returns
        -------
        pd.Series
            Probability density function of the scattering function
        """
        return self.data / self.norm

    @property
    def cdf(self) -> pd.DataFrame:
        """
        Cumulative distribution function of the scattering function.

        Returns
        -------
        pd.Series
            Cumulative distribution function of the scattering function
        """
        cdf = self.data.cumsum(axis=0).cumsum(axis=1)
        return cdf / cdf.iloc[-1, -1]

class TransferFunc:
    """
    Single Differencial (angle or Outgoing energy) scattering function base
    class.
    """

    def __init__(self, Ein: float, T: float, M: float,  *args, **kwargs):
        """
        Initialize the TransferFunc class.

        Parameters
        ----------
        Ein : float
            The neutron incident energy in eV
        T : float
            Temperature of the material in K
        M : float
            Mass of the material in amu
        args : Iterable, (N,)
            The scattering function data for the pd.Series
        kwargs : dict
            Optional arguments for the construction of the pd.Series
        """
        # Atributes of the scattering function (Change in these parameters will
        # change the scattering function):
        self.Ein = Ein
        self.T = T
        self.M = M

        # The scattering function data:
        self.data = pd.Series(*args, **kwargs)

    @property
    def data(self) -> pd.Series:
        """
        Scattering function data.

        Returns
        -------
        pd.Series
            The scattering function data
        """
        return self._data

    @data.setter
    def data(self, pdf: Iterable):
        """
        Set the scattering function data and check the normalization.

        Parameters
        ----------
        pdf : pd.Series
            The scattering function data

        """
        pdf_ = pd.Series(pdf).sort_index()

        # Set index name:
        pdf_.index.name = "Eout"

        # Save the data:
        self._data = pdf_

    @classmethod
    def from_sigma1(cls, Ein: float, M: float, T: float, Eout: np.array):
        """
        Calculate the scattering function using Maxwellian velocity distribution
        and angular integration
        .. math::
            S(E, E^\prime, M, T) = \frac{1}{2}\sqrt{\frac{M}{m\pi k_BT}}\frac{\sqrt{E^\prime}}{E}\left(exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} - \sqrt{E^\prime}\right)^2 \right) - exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} + \sqrt{E^\prime}\right)^2 \right)\right)

        Parameters
        ----------
        Ein : float
            The incident energy of the neutron in eV
        M : float
            Mass of the material in amu
        T : float
            Temperature of the material in K
        Eout : np.array
            The neutron outgoing energy grid in eV

        Returns
        -------
        TransferFunc
            The scattering function for the given temperature, incident energy
            and mass using Maxwellian velocity distribution and angular
            integration

        Examples
        --------
        # Generate Broadening test results:
        >>> Ein = 36.68723
        >>> Eout = np.linspace(Ein * 0.98 , Ein * 1.02, 1000)
        >>> M = 238.05077040419212
        >>> T = 300
        >>> pdf = TransferFunc.from_sigma1(Ein, M, T, Eout)
        >>> pdf.data.iloc[::100]
        Eout
        35.953485    8.937086e-15
        36.100381    1.841784e-09
        36.247277    2.425252e-05
        36.394173    2.074937e-02
        36.541069    1.172637e+00
        36.687964    4.449812e+00
        36.834860    1.152331e+00
        36.981756    2.069367e-02
        37.128652    2.618312e-05
        37.275548    2.371152e-09
        dtype: float64
        """
        return cls(Ein, T, M, sigma1(Eout, Ein, T, M), index=Eout)

    @classmethod
    def from_theta(cls, Ein: float, M: float, T: float, Eout: np.array,
                   theta: float, *args, model: str = "fgm", **kwargs):
        """
        Generate the Transfer function from a
        S(alpha, -beta) table.

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
        theta : float
            The angle of the scattering in degrees
        model : str
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
        TransferFunc
            Double differential scattering scattering function

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.array([7.10, 7.15, 7.2, 7.25, 7.3157])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = 15

        # Using the Free Gas Model:
        >>> TransferFunc.from_theta(Ein, M, T, Eout, theta, model="fgm").data.round(6)
        Eout
        7.1000     0.000030
        7.1500     0.851083
        7.2000    21.126578
        7.2500     0.489767
        7.3157     0.000000
        Name: 15, dtype: float64

        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> TransferFunc.from_theta(Ein, M, T, Eout, theta, pdos, model="sct").data.round(6)
        Eout
        7.1000     0.000031
        7.1500     0.859129
        7.2000    21.090382
        7.2500     0.495350
        7.3157     0.000000
        Name: 15, dtype: float64

        # Using the Phonon expansion model:
        >>> TransferFunc.from_theta(Ein, M, T, Eout, theta, pdos, threshold=1.0e-14, model="pdos").data.round(6)
        Eout
        7.1000     0.007187
        7.1500     1.036528
        7.2000    22.969304
        7.2500     0.584262
        7.3157     0.000400
        Name: 15, dtype: float64
        """
        # Get the scattering function values to the given angle:
        scatFunc = ScatFunc.from_model(Ein, M, T, Eout, [theta],  *args,
                                       model=model, **kwargs).data

        # Erase angular normalization
        scatFunc *= 2

        return cls(Ein, T, M, scatFunc.iloc[0], name=theta)

    @classmethod
    def from_alpha(cls, alpha: float, Ein: float, M: float, T: float, Eout: np.array,
                   *args, model: str = "fgm", **kwargs):
        """
        Generate the Transfer function from S(alpha, -beta) tables.

        Parameters
        ----------
        alpha: float
            The alpha parameter of the scattering function
        Ein: float
            The incident energy of the neutron in eV
        M: float
            The mass of the target material in amu
        T: float
            Temperature of the material in K
        Eout: np.array
            The neutron outgoing energy grid in eV
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
        threshold : 'float', optional
            Minimun value to take into account in the creation of tauN
            functions. For T>200 is convenient to set into 1.0e-14 to speed up
            the calculations. The default is 0.0.
        decimal: 'float'
            Decimal precision for the calculation of the expansion order.
            The default is 1.0e-6.
        order_max: 'int'
            Maximun expansion order. The default is 5000.

        Returns
        -------
        TransferFunc
            Transfer function using S(alpha, -beta) tables

        Examples
        --------
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> Ein = 7.2
        >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
        >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> alpha = Ein / M / (kb * T)
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> TransferFunc.from_alpha(alpha, Ein, M, T, Eout, model="fgm").data.loc[Eout_test].round(6)
        Eout
        6.7554    0.000000
        6.9050    0.006646
        7.0439    1.209391
        7.2000    5.061384
        7.3157    0.716275
        7.4480    0.003292
        dtype: float64

        >>> TransferFunc.from_alpha(alpha, Ein, M, T, Eout, pdos, model="sct").data.loc[Eout_test].round(6)
        Eout
        6.7554    0.000000
        6.9050    0.006791
        7.0439    1.213667
        7.2000    5.054144
        7.3157    0.716771
        7.4480    0.003338
        dtype: float64

        >>> TransferFunc.from_alpha(alpha, Ein, M, T, Eout, pdos, model="pdos").data.loc[Eout_test].round(6)
        Eout
        6.7554    0.000003
        6.9050    0.009647
        7.0439    1.196693
        7.2000    5.103854
        7.3157    0.705293
        7.4480    0.003823
        dtype: float64
        """
        # Get the S(alpha, -beta) values to the given alpha:
        sab = Sab.from_model(alpha, Beta.from_default(T), T, *args,
                             model=model, **kwargs).full

        # Interpolate to the outgoing energy grid:
        EoutCalc = Ein + sab.index.values * kb * T
        scatfunc = np.interp(Eout, EoutCalc, sab.values)

        # Normalize the scattering function to Eout:
        scatfunc /= kb * T

        return cls(Ein, T, M, scatfunc, index=Eout)

    @classmethod
    def from_alpha0(cls, Ein: float, M: float, T: float, Eout: np.array,
                    *args, model: str = "fgm", **kwargs):
        """
        Generate the Transfer function from gressier
        recoil energy

        Parameters
        ----------
        Ein: float
            The incident energy of the neutron in eV
        M: float
            The mass of the target material in amu
        T: float
            Temperature of the material in K
        Eout: np.array
            The neutron outgoing energy grid in eV
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
        threshold : 'float', optional
            Minimun value to take into account in the creation of tauN
            functions. For T>200 is convenient to set into 1.0e-14 to speed up
            the calculations. The default is 0.0.
        decimal: 'float'
            Decimal precision for the calculation of the expansion order.
            The default is 1.0e-6.
        order_max: 'int'
            Maximun expansion order. The default is 5000.

        Returns
        -------
        TransferFunc
            Transfer function using S(alpha, -beta) table
            based on the gressier recoil energy

        Examples
        --------
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> Ein = 7.2
        >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
        >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> TransferFunc.from_alpha0(Ein, M, T, Eout, model="fgm").data.loc[Eout_test].round(6)
        Eout
        6.7554    0.000000
        6.9050    0.005971
        7.0439    1.180445
        7.2000    5.102312
        7.3157    0.709375
        7.4480    0.003059
        dtype: float64

        >>> TransferFunc.from_alpha0(Ein, M, T, Eout, pdos, model="sct").data.loc[Eout_test].round(6)
        Eout
        6.7554    0.000000
        6.9050    0.006103
        7.0439    1.184746
        7.2000    5.094991
        7.3157    0.709907
        7.4480    0.003103
        dtype: float64

        >>> TransferFunc.from_alpha0(Ein, M, T, Eout, pdos, model="pdos").data.loc[Eout_test].round(6)
        Eout
        6.7554    0.000003
        6.9050    0.008817
        7.0439    1.168504
        7.2000    5.145645
        7.3157    0.698297
        7.4480    0.003581
        dtype: float64
        """
        beta = Beta.from_default(T)
        sab = Sab.from_alpha0(Ein, T, M, beta, *args, model=model,
                              **kwargs).full
        EoutCalc = Ein + sab.index.values * kb * T
        scatfunc = np.interp(Eout, EoutCalc, sab.values)
        scatfunc /= kb * T
        return cls(Ein, T, M, scatfunc, index=Eout)

    @staticmethod
    def get_alpha0(EinGrid: np.ndarray, M: float, T: float, *args,
                   model: str = "fgm", **kwargs) -> pd.DataFrame:

        """
        Calculate the alpha0 scattering function.

        Parameters
        ----------
        EinGrid: np.ndarray
            The incident energy grid in eV
        M: float
            The mass of the target material in amu
        T: float
            Temperature of the material in K
        model: str
            The model used to generate the S(alpha, beta) table. The available
            models are:
                - "pdos": Phonon expansion model
                - "fgm" : Free Gas Model (Default)
                - "sct" : Short Collision Time model
        display: bool
            If True, return a pd.DataFrame for visualization.
            If False, return a xp.ndarray for computation.

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
        threshold : 'float', optional
            Minimun value to take into account in the creation of tauN
            functions. For T>200 is convenient to set into 1.0e-14 to speed up
            the calculations. The default is 0.0.
        decimal: 'float'
            Decimal precision for the calculation of the expansion order.
            The default is 1.0e-6.
        order_max: 'int'
            Maximun expansion order. The default is 5000.

        Returns
        -------
        pd.DataFrame
            The alpha0 scattering function

        Examples
        --------
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> Ein = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> index = pd.Index(Ein, name="Ein")
        >>> T = 300
        >>> M = 238.05077040419212
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> TransferFunc.get_alpha0(Ein, M, T, model="fgm").iloc[::, 1000::1000].round(6)
        beta    -2.662634  -1.114591   0.433452   1.981495   9.902464
        Ein
        6.7554   0.153967   0.269409   0.158014   0.031065        0.0
        6.9050   0.156781   0.266477   0.155477   0.031139        0.0
        7.0439   0.159258   0.263776   0.153185   0.031192        0.0
        7.2000   0.161892   0.260769   0.150679   0.031233        0.0
        7.3157   0.163744   0.258559   0.148868   0.031253        0.0
        7.4480   0.165761   0.256053   0.146842   0.031264        0.0
        >>> TransferFunc.get_alpha0(Ein, M, T, pdos, model="sct").iloc[::, 1000::1000].round(6)
        beta    -2.668826  -1.120783   0.427260   1.975303   9.739857
        Ein
        6.7554   0.153590   0.264466   0.156371   0.030960        0.0
        6.9050   0.156257   0.261604   0.153885   0.031015        0.0
        7.0439   0.158601   0.258970   0.151640   0.031051        0.0
        7.2000   0.161088   0.256037   0.149185   0.031075        0.0
        7.3157   0.162834   0.253884   0.147411   0.031082        0.0
        7.4480   0.164732   0.251443   0.145427   0.031080        0.0
        >>> TransferFunc.get_alpha0(Ein, M, T, pdos, model="pdos").iloc[::, 1000::1000].round(6)
        beta    -2.998559  -1.450516   0.097527   1.645570   4.033176
        Ein
        6.7554   0.111171   0.257399   0.201964   0.047321   0.000699
        6.9050   0.114250   0.256102   0.198472   0.047219   0.000738
        7.0439   0.117031   0.254833   0.195325   0.047108   0.000774
        7.2000   0.120064   0.253339   0.191893   0.046964   0.000815
        7.3157   0.122248   0.252190   0.189417   0.046847   0.000846
        7.4480   0.124680   0.250838   0.186655   0.046702   0.000881
        """
        Ein = np.unique(EinGrid) if hasattr(EinGrid, '__len__') else np.array([EinGrid])
        # Scattering function calculation
        alpha, beta = Alpha.from_recoil(Ein, T, M), Beta.from_default(T)
        scatfunc = Sab.from_model(alpha, beta, T, *args,
                   model=model, **kwargs).full
        # Erase the columns with all zeros
        scatfunc = scatfunc.loc[::, ~scatfunc.eq(0).all()]
        return scatfunc.set_axis(pd.Index(Ein, name="Ein"), axis=0)


    @property
    def norm(self) -> float:
        """
        Normalization of the scattering function.

        Returns
        -------
        float
            Normalization of the transference function

        Examples
        --------
        # Generate Broadening test results:
        >>> Ein = 36.68723
        >>> Eout = np.linspace(Ein * 0.98 , Ein * 1.02, 1000)
        >>> M = 238.05077040419212
        >>> T = 300
        >>> pdf = TransferFunc.from_sigma1(Ein, M, T, Eout)
        >>> float(round(pdf.norm, 6))
        1.000001
        """
        return integrate(self.data)

    @property
    def pdf(self) -> pd.Series:
        """
        Probability density function of the scattering function.

        Returns
        -------
        pd.Series
            Probability density function of the scattering function
        """
        return self.data / self.norm
    @property
    def cdf(self) -> pd.Series:
        """
        Cumulative distribution function of the scattering function.

        Returns
        -------
        pd.Series
            Cumulative distribution function of the scattering function
        """
        cdf = self.data.cumsum()
        return cdf / cdf.iloc[-1]


@vectorize(['float64(float64, float64, float64, float64)'],
           target='parallel', cache=True)
def sigma1(Eout: float, Ein: float, T: float, M: float):
    """
    Sigma1 function for Energy differential scattering function
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
        Scattering function based on sigma1 model

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
    # Define teh constants:
    AkbT = M / (m * kb * T)

    # Get the negative exponetiial part:
    expNegative = exp(- AkbT * (sqrt(Ein) - sqrt(Eout)) ** 2)

    # Get the positive exponetiial part:
    expPositive = exp(- AkbT * (sqrt(Ein) + sqrt(Eout)) ** 2)

    # Calculate the scattering function:
    transferFunc = 0.5 * (expNegative - expPositive)
    transferFunc *= sqrt(AkbT / pi) * sqrt(Eout) / Ein

    return transferFunc


@nb.jit(nopython=True, cache=True)
def get_ScatSctAngular(Eout: np.ndarray, mu: np.ndarray, Ein: float, T: float,
                       M: float, Teff: float, ws: float) -> np.ndarray:
    """
    Calculate the scattering function from the Short Collision Time model using
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
        The scattering function values for a single angle
    """
    # Get the beta grid:
    beta = get_beta(Eout, Ein, T, abs=False)

    # Get the temperature ratio:
    Tratio = Teff / T

    # Get the alpha grid and the scattering function:
    if isinstance(mu, (int, float)):
        alpha = get_alphaFromEout(Eout, Ein, T, M, mu)
        sabValues = get_SabSctAlpha(alpha, beta, Tratio, ws)
    else:
        alpha = get_alphaMat(Eout, Ein, T, M, mu)
        sabValues = get_SabSct(alpha, beta, Tratio, ws)

    # Apply normalization to the scattering function:
    return sabValues * normFactor(Eout, Ein, T, M)


@nb.jit(nopython=True, cache=True)
def get_SabClm(alpha: np.ndarray, nphonon: int,  tauNinterp: np.ndarray,
               DebyeWallerCoeff: float) -> np.ndarray:
    """
    Generate the scattering function from a S(alpha, -beta) table based on
    the phonon expansion model using a single angle.

    Parameters
    ----------
    alpha : 'np.ndarray', (Z, N)
        alpha grid values.
    nphonon : 'int', optional
        Phonon expansion order.
    tauNinterp : 'np.ndarray', (Z, N)
        tauN function for the phonon expansion interpolated to the beta grid.
    DebyeWallerCoeff : 'float'
        Debye Waller Coefficient in LEAPR formalism.

    Returns
    -------
    S_diag : 'np.ndarray', (Z, N)
        S(alpha, -beta) values for the alpha and beta combinations.

    Examples
    --------
    >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
    >>> Ein = 7.2
    >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
    >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> mu = np.cos(np.deg2rad([120]))
    >>> beta = get_beta(Eout, Ein, T)
    >>> alpha_mat = get_alphaMat(beta * kb * T + Ein, Ein, T, M, mu)
    >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
    >>> DebyeWallerCoeff = pdos.DebyeWallerCoeff
    >>> nphonon = get_expansionOrder(alpha_mat, DebyeWallerCoeff, 1.0e-6, 5000)
    >>> tauN = pdos.tauN(nphonon, 1.0e-14, values=True)
    >>> tau1beta = pdos.beta.data
    >>> tauNbeta = get_tauNbeta(tau1beta, tauN.shape[1])
    >>> tauNinterp = interp_multyParallel(beta, tauNbeta, tauN)
    >>> sabValues = get_SabClm(alpha_mat, nphonon, tauNinterp, DebyeWallerCoeff)
    >>> pd.DataFrame(sabValues, index=[120], columns=beta).T.iloc[::100].round(6)
                   120
    0.000000  0.210641
    0.399957  0.247226
    0.802224  0.269120
    1.204491  0.271591
    1.603331  0.254529
    2.000979  0.221921
    2.403246  0.179661
    2.805512  0.135357
    3.526166  0.068344
    4.330699  0.024589
    5.135233  0.006799
    """
    # Zero phonon expansion:
    IterSum = np.log(alpha * DebyeWallerCoeff)
    alphaMul = np.exp(- alpha * DebyeWallerCoeff + IterSum)
    S_diag = alphaMul * tauNinterp[0]

    # Higher phonon expansion (nphonon >= 1):
    for n in range(1, nphonon):
        # Compute S(alpha, -beta) for tauN reshape
        IterSum += np.log(alpha * DebyeWallerCoeff / (n + 1))
        alphaMul = np.exp(- alpha * DebyeWallerCoeff + IterSum)
        S_diag += alphaMul * tauNinterp[n]
    return S_diag


@nb.jit(nopython=True, cache=True)
def normFactor(Eout: np.ndarray, Ein: float, T: float, M: float) -> np.ndarray:
    """
    Normalization factor for the scattering function calculation.

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
        Normalization factor for the scattering function calculation.
    """
    M_div_m = M / m
    aws = ((M_div_m + 1) / M_div_m) ** 2
    two_kb_T = 2 * kb * T
    return aws * np.sqrt(Eout / Ein) / two_kb_T


@nb.jit(nopython=True, cache=True)
def get_ScatFuncClm(Ein: float, M: float, T: float, Eout: np.ndarray,
                    mu: np.ndarray, tauN: np.ndarray, tauNbeta: np.ndarray,
                    DebyeWallerCoeff: float,  alpha0: float = None) -> np.ndarray:
    """
    Generate the scattering function from a S(alpha, -beta) table based on
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
        Scattering function values for a single angle.

    Examples
    --------
    >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
    >>> Ein = 7.2
    >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
    >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> mu = np.cos(np.deg2rad([120]))
    >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
    >>> DebyeWallerCoeff = pdos.DebyeWallerCoeff
    >>> nphonon = get_expansionOrder(get_alphaFromEout(Eout, Ein, M, T, mu), DebyeWallerCoeff, 1.0e-6, 5000)
    >>> tauN = pdos.tauN(nphonon, 1.0e-14, values=True)
    >>> tau1beta = pdos.beta.data
    >>> tauNbeta = get_tauNbeta(tau1beta, tauN.shape[1])
    >>> sd_pdf = get_ScatFuncClm(Ein, M, T, Eout, mu, tauN, tauNbeta, DebyeWallerCoeff)
    >>> pd.Series(sd_pdf[0], index=Eout).loc[Eout_test].round(6)
    6.7554    0.034510
    6.9050    0.426488
    7.0439    1.383081
    7.2000    1.262613
    7.3157    0.415630
    7.4480    0.042074
    dtype: float64

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
    beta = get_beta(Eout, Ein, T)

    # Eout calculation
    EoutCalc = np.sort(Ein + np.concatenate((-beta[::-1], beta[1::])) * kb * T)

    # Ensure the Eout values are positive:
    positiveMask = EoutCalc > 0
    EoutCalc = EoutCalc[positiveMask]

    # Get the number of phonon expansion:
    nphonon = tauN.shape[0]

    # Interpolation of tauN functions to reduce the number of calculations:
    tauNinterp = interp_multyParallel(beta, tauNbeta, tauN)

    # Get the alpha matrix for the scattering function with the maximun outgoing energy:
    Eout_ = beta * kb * T + Ein if len(beta) < len(Eout) else Eout
    if alpha0 is None:
        alphaMat = get_alphaMat(Eout_, Ein, T, M, mu)
    else:
        alphaMat = get_alphaMatMod(Eout_, Ein, M, T, mu, DebyeWallerCoeff,
                                   alpha0)

    # Get the S(alpha, -beta) values for the alpha and beta combinations:
    sabValues = get_SabClm(alphaMat, nphonon, tauNinterp, DebyeWallerCoeff)

    # Full Scattering function values calculation:
    scatFuncValues = np.concatenate((sabValues[::, ::-1], sabValues[::, 1:] * np.exp(-beta[1:])), axis=1)[::, positiveMask]

    # Normalization constant
    scatFuncValues *= normFactor(EoutCalc, Ein, T, M)

    # Interpolation for avoiding numerical fluctuations:
    return interp_multyParallel(Eout, EoutCalc, scatFuncValues)