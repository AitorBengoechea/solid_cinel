"""
Python for working with Diferential XS.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
from scipy.constants import physical_constants as const
from solid_cinel.core.scattering_function import TransferFunc, ScatFunc
from solid_cinel.core.scattering_function.alpha import get_alpha, get_alphaMat
from solid_cinel.core.generic import integrate, reshift, interpolation
import os
from typing import Iterable

# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]

# Avoid numba fast math:
nb.config.FASTMATH_DEFAULT = False


class Dxs:
    """
    Class for the differential cross section for elastic scattering
    """
    def __init__(self, Ein: float, T: float, M: float, *args, **kwargs):
        """
        Class for the Double differential cross section for elastic scattering

        Parameters
        ----------
        xs : pd.Series or pd.DataFrame, (N,) or (M, N)
            0K xs data for the given material in barns. If the cross
            section is a matrix, the scattering function is convolved directly
            with xs. If the cross section is a vector, the scattering function
            is convolved with the cross section for each outgoing energy or
            with the Exs introduced by the user.
        Exs : np.ndarray, optional, (N,) or (M, N)
            Displazed Energy grid of the cross section. If not provided, the
            energy grid of the scattering function is used.
        theta : np.ndarray, (M,)
            The neutron outgoing angle grid in degrees (0, 180]
        kwargs : dict
            Extra parameters for the selected algorithm
        """
        # Atributes of the scattering function (Change in these parameters will
        # change the scattering function):
        self.Ein = Ein
        self.T = T
        self.M = M
        # The dxs data:
        self.data = pd.Series(*args, **kwargs)

    @property
    def data(self) -> pd.Series:
        """
        Diferential xs data.

        Returns
        -------
        pd.Series
            Diferential xs data
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
        pdf_.index.name = "Eout"
        self._data = pdf_


    @classmethod
    def from_sigma1(cls, xs0K: pd.Series, Ein: float, M: float, T: float, Eout: np.ndarray):
        """
        Generate the Differential xs for elastic scattering from sigma1
        ..math::
            \frac{d\sigma_T(E)}{dE^\prime} = \frac{1}{2}\sqrt{\frac{M}{m\pi k_BT}}\frac{\sqrt{E^\prime}}{E}\sigma_0(E^\prime)\left(exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} - \sqrt{E^\prime}\right)^2 \right) - exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} + \sqrt{E^\prime}\right)^2 \right)\right)

        Parameters
        ----------
        xs0K : pd.Series, (Z,)
            0K xs data for the given material in barns
        Ein : float
        The incident energy of the neutron in eV
        M : float
            Mass of the material in amu
        T : float
            Temperature of the material in K
        Eout : np.ndarray, (N,)
            The neutron outgoing energy grid in eV

        Returns
        -------
        Dxs
            Differential cross section for elastic scattering

        Examples
        --------
        # 0K xs data for U238:
        >>> from solid_cinel.core.xs.xs import Xs
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("dxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = Xs.read_xs("u238.0.2")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212

        # SIGMA1 algorithm:
        >>> Dxs.from_sigma1(xs0K, Ein, M, T, Eout).data.iloc[::100]
        Eout
        1.80000     0.000049
        1.84004     0.009909
        1.88008     0.575486
        1.92012    10.018740
        1.96016    54.281606
        2.00020    94.844029
        2.04024    55.278649
        2.08028    11.098885
        2.12032     0.791546
        2.16036     0.020645
        dtype: float64
        """
        # Get the transfer function:
        transferFunc = TransferFunc.from_sigma1(Ein, M, T, Eout).data

        # Get the differential cross section:
        dxs = transferFunc * cls.interp_xs0K(xs0K, transferFunc.index.values)

        return cls(Ein, T, M, dxs)

    @classmethod
    def from_alpha(cls, xs0K: pd.Series, alpha: float, Ein: float, M: float,
                   T: float, Eout: np.ndarray, *args, model: str = "fgm", **kwargs):
        """
        Generate the Differential xs for elastic scattering from the
        S(alpha, beta) tables distribution

        Parameters
        ----------
        xs0K : pd.Series, (Z,)
            0K xs data for the given material in barns
        Ein : float
            The incident energy of the neutron in eV
        M : float
            Mass of the material in amu
        T : float
            Temperature of the material in K
        Eout : np.ndarray, (N,)
            The neutron outgoing energy grid in eV
        model : str
            The model used to calculate the S(alpha, beta) distribution. The available models are:
                - "fgm": Free Gas Model (default)
                - "sct": Short Collision Time
                - "pdos": Phonon Density of States

        Parameters for sct
        ------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.

        Parameters for pdos
        -------------------
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
        Dxs
            Differential cross section for elastic scattering

        Examples
        --------
        # 0K xs data for U238:
        >>> from solid_cinel.core.xs.xs import Xs
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("dxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = Xs.read_xs("u238.0.2")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> M = 238.05077040419212
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> alpha = Ein / (kb * T) / M

        # alpha0 algorithm:
        >>> Dxs.from_alpha(xs0K, alpha, Ein, M, T, Eout, model="fgm").data.iloc[::100]
        Eout
        1.80000     0.000299
        1.84004     0.034346
        1.88008     1.303107
        1.92012    16.344437
        1.96016    67.757179
        2.00020    92.829654
        2.04024    42.029652
        2.08028     6.289029
        2.12032     0.311053
        2.16036     0.005082
        dtype: float64

        # Using the Short Collision Time model:
        >>> from solid_cinel.core.material.vibration import Pdos
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> Dxs.from_alpha(xs0K, alpha, Ein, M, T, Eout, pdos, model="sct").data.iloc[::100]
        Eout
        1.80000     0.000312
        1.84004     0.035241
        1.88008     1.320349
        1.92012    16.416150
        1.96016    67.719720
        2.00020    92.676130
        2.04024    42.007690
        2.08028     6.317055
        2.12032     0.315203
        2.16036     0.005216
        dtype: float64
        """
        # Get the transfer function:
        transferFunc = TransferFunc.from_alpha(alpha, Ein, M, T, Eout, *args,
                                               model=model, **kwargs).data

        # Get the recoil energy:
        recoil = alpha * kb * T

        # Get the differential cross section:
        dxs = transferFunc * cls.interp_xs0K(xs0K, Eout, recoil)

        return cls(Ein, T, M, dxs)

    @classmethod
    def from_alpha0(cls, xs0K: pd.Series, Ein: float, M: float, T: float, Eout: np.ndarray,
                   theta: np.ndarray, *args, model: str = "fgm", **kwargs):
        """
        Generate the Differential xs for elastic scattering from the most similar distribution of the S(alpha, -beta)
        tables and sigma1 algorithm
        ..math::
            \frac{d\sigma_T(E)}{dE^\prime} = \frac{\sigma(E^\prime + R)}{2 * k_B * T}\sqrt{\frac{E^\prime}{E}} S(\alpha(\theta, E^\prime, E, M, T), \beta( E^\prime, E, T))

        Parameters for fgm, sct and pdos models
        ----------------------------------------
        xs0K : pd.Series, (Z,)
            0K xs data for the given material in barns
        Ein : float
        The incident energy of the neutron in eV
        M : float
            Mass of the material in amu
        T : float
            Temperature of the material in K
        Eout : np.ndarray, (N,)
            The neutron outgoing energy grid in eV
        theta : np.ndarray, (M,)
            The neutron outgoing angle grid in degrees (0, 180]
        model : str
            The model used to calculate the S(alpha, beta) distribution. The available models are:
                - "fgm": Free Gas Model (default)
                - "sct": Short Collision Time
                - "pdos": Phonon Density of States

        Parameters for sct
        ------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.

        Parameters for pdos
        -------------------
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

        Examples
        --------
        # 0K xs data for U238:
        >>> from solid_cinel.core.xs.xs import Xs
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("dxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = Xs.read_xs("u238.0.2")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> theta = np.arange(1, 180, 1)
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212

        # alpha0 algorithm:
        >>> Dxs.from_alpha0(xs0K, Ein, M, T, Eout, theta, model="fgm").data.iloc[::100]
        Eout
        1.80000     0.000287
        1.84004     0.033460
        1.88008     1.285044
        1.92012    16.258397
        1.96016    67.751989
        2.00020    92.982545
        2.04024    42.024971
        2.08028     6.255488
        2.12032     0.306710
        2.16036     0.004950
        dtype: float64

        # Using the Short Collision Time model:
        >>> from solid_cinel.core.material.vibration import Pdos
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> Dxs.from_alpha0(xs0K, Ein, M, T, Eout, theta, pdos, model="sct").data.iloc[::100]
        Eout
        1.80000     0.000299
        1.84004     0.034323
        1.88008     1.301847
        1.92012    16.328835
        1.96016    67.714845
        2.00020    92.831027
        2.04024    42.003188
        2.08028     6.283015
        2.12032     0.310753
        2.16036     0.005079
        dtype: float64

        # Using the Phonon Density of States model:
        >>> Dxs.from_alpha0(xs0K, Ein, M, T, Eout, theta, pdos, model="pdos").data.iloc[::100]
        Eout
        1.80000     0.005152
        1.84004     0.112088
        1.88008     1.741582
        1.92012    15.942947
        1.96016    66.211564
        2.00020    95.747483
        2.04024    41.058942
        2.08028     6.141493
        2.12032     0.418065
        2.16036     0.016800
        dtype: float64
        """
        # Calculate alpha0 values from the scattering function:
        alpha0 = ScatFunc.from_model(Ein, M, T, Eout, theta, *args,
                                     model=model, **kwargs).alpha0

        # Get dxs based on the alpha0:
        return cls.from_alpha(xs0K, alpha0, Ein, M, T, Eout, *args, model=model, **kwargs)

    @classmethod
    def from_theta(cls, xs0K: pd.Series, Ein: float, M: float, T: float, Eout: np.ndarray,
                   theta: float, *args, model: str = "fgm", **kwargs):
        """
        Generate the Differential xs for elastic scattering from the scattering
        function angle distribution

        Parameters
        ----------
        xs0K : pd.Series, (Z,)
            0K xs data for the given material in barns
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

        # 0K xs data for U238:
        >>> from solid_cinel.core.xs.xs import Xs
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("dxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = Xs.read_xs("u238.0.2")
        >>> os.chdir(wd)

        # Using the Free Gas Model:
        >>> Dxs.from_theta(xs0K, Ein, M, T, Eout, theta, model="fgm").data.round(6)
        Eout
        7.1000      0.000277
        7.1500      8.291518
        7.2000    212.369147
        7.2500      5.032810
        7.3157      0.000001
        Name: 15, dtype: float64
        """
        # Get the transfer function:
        transferFunc = TransferFunc.from_theta(Ein, M, T, Eout, theta, *args,
                                                model= model, **kwargs).data

        # Get the recoil energy:
        recoil = get_alpha(Ein, T, M, Eout, np.cos(np.deg2rad(theta)))
        recoil *= kb * T

        # Get the differential cross section:
        dxs = transferFunc * cls.interp_xs0K(xs0K, Eout, recoil)

        return cls(Ein, T, M, dxs)

    @classmethod
    def from_sab(cls, xs0K: pd.Series, Ein: float, M: float, T: float, Eout: np.ndarray,
                 theta: np.ndarray, *args, model: str = "fgm", recoil: bool = True,
                 **kwargs):
        """
        Generate the Differential xs for elastic scattering from the scattering
        function angle distribution

        Parameters
        ----------
        xs0K : pd.Series, (Z,)
            0K xs data for the given material in barns
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

        Examples
        --------
        # 0K xs data for U238:
        >>> from solid_cinel.core.xs.xs import Xs
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("dxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = Xs.read_xs("u238.0.2")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> theta = np.arange(1, 180, 1)
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212

        # Using the Free Gas Model(NO RECOIL):
        >>> dxs = Dxs.from_sab(xs0K, Ein, M, T, Eout, theta, model="fgm", recoil=False)
        >>> dxs.data.iloc[::100]
        Eout
        1.80000     0.768794
        1.84004     3.231998
        1.88008    10.451361
        1.92012    26.580672
        1.96016    54.522950
        2.00020    91.812241
        2.04024    34.506930
        2.08028    10.974592
        2.12032     2.920481
        2.16036     0.643604
        dtype: float64


        # Using the Free Gas Model(RECOIL):
        >>> dxs = Dxs.from_sab(xs0K, Ein, M, T, Eout, theta, model="fgm")
        >>> dxs.data.iloc[::100]
        Eout
        1.80000     0.768710
        1.84004     3.231537
        1.88008    10.449309
        1.92012    26.573581
        1.96016    54.502864
        2.00020    91.763488
        2.04024    34.493443
        2.08028    10.971388
        2.12032     2.919831
        2.16036     0.643487
        dtype: float64

        # Using the Short Collision Time model:
        >>> from solid_cinel.core.material.vibration import Pdos
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> dxs = Dxs.from_sab(xs0K, Ein, M, T, Eout, theta, pdos, model="sct")
        >>> dxs.data.iloc[::100]
        Eout
        1.80000     0.776211
        1.84004     3.249951
        1.88008    10.476815
        1.92012    26.585995
        1.96016    54.453402
        2.00020    91.615111
        2.04024    34.517809
        2.08028    11.011204
        2.12032     2.940973
        2.16036     0.650950
        dtype: float64

        # Using the Phonon Density of States model:
        >>> dxs = Dxs.from_sab(xs0K, Ein, M, T, Eout, theta, pdos, model="pdos")
        >>> dxs.data.iloc[::100]
        Eout
        1.80000     1.171084
        1.84004     3.942164
        1.88008    11.189753
        1.92012    26.706945
        1.96016    53.726842
        2.00020    80.755420
        2.04024    34.121072
        2.08028    10.825798
        2.12032     2.897511
        2.16036     0.652177
        dtype: float64
        """
        mu = np.cos(np.deg2rad(theta))
        # Calculate the scattering function:
        scatFunc = ScatFunc.from_model(Ein, M, T, Eout, theta, *args,
                                     model= model, **kwargs)

        # Get the recoil energy if needed:
        if recoil:
            alphaRecoil = get_alphaMat(Eout, Ein, T, M, mu) * kb * T
            xs0Kinterp = cls.interp_xs0K(xs0K, Eout, recoil=alphaRecoil)
            dxs = (scatFunc.data * xs0Kinterp).apply(integrate)
        else:
            transferFunc = scatFunc.to_transferFunc.data
            dxs = transferFunc * cls.interp_xs0K(xs0K, Eout)

        return cls(Ein, T, M, dxs)

    @staticmethod
    def interp_xs0K(xs0K: pd.Series, Eout: np.ndarray,
                    recoil: [None, int, float] = None) -> np.ndarray:
        """
        Interpolate the 0K cross section to the given Eout grid

        Parameters
        ----------
        xs0K: pd.Series, (Z,)
            0K xs data for the given material in barns
        Eout: np.ndarray, (N,)
            The neutron outgoing energy grid in eV
        recoil: int, float, optional
            The recoil energy of the target material in eV. The default is None.

        Returns
        -------
        np.ndarray
            The interpolated 0K cross section
        """
        if recoil is None:
            return interpolation(xs0K, Eout, values=True)
        else:
            return interpolation(xs0K, Eout + recoil, values=True)

    @property
    def integral(self) -> float:
        """
        The integral value of the Diferential xs

        Returns
        -------
        float
            The integral value of the Diferential xs

        Examples
        --------
        # 0K xs data for U238:
        >>> from solid_cinel.core.xs.xs import Xs
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("dxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = Xs.read_xs("u238.0.2")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212

        # SIGMA1 algorithm:
        >>> float(round(Dxs.from_sigma1(xs0K, Ein, M, T, Eout).integral, 2))
        9.09

        # DOPUSH algorithm:
        >>> theta = np.arange(0, 180, 1)[1::]
        >>> float(round(Dxs.from_alpha0(xs0K, Ein, M, T, Eout, theta, model="fgm").integral, 2))
        9.09
        """
        return integrate(self.data)

    @property
    def prob(self) -> dict:
        """
        Get the upscattering and downscattering probabilities for the selected Ein, T, M

        Returns
        -------
        dict
            Dictionary with the upscattering and downscattering probabilities

        Examples
        --------
        # 0K xs data for U238:
        >>> from solid_cinel.core.xs.xs import Xs
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("dxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = Xs.read_xs("u238.0.2")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> dxs = Dxs.from_sigma1(xs0K, Ein, M, T, Eout)
        >>> float(round(dxs.prob["upscattering"], 6))
        0.505184
        >>> float(round(dxs.prob["downscattering"], 6))
        0.490636
        >>> float(round(dxs.prob["Ein=Eout"], 6))
        0.004179
        """
        # Get the integral value:
        integral = self.integral

        # Get the outgoig energy grid:
        Eout = self.data.index.values

        # Get the upscattering and downscattering probabilities:
        up = integrate(self.data.loc[Eout > self.Ein]) / integral
        down = integrate(self.data.loc[Eout < self.Ein]) / integral

        return {"upscattering": up,  "downscattering": down, "Ein=Eout": 1.0 - up - down}

    @property
    def pdf(self) -> pd.Series:
        """
        Get the probability density function of the Differential XS

        Returns
        -------
        pd.Series
            The probability density function of the Differential XS

        Examples
        --------
        # 0K xs data for U238:
        >>> from solid_cinel.core.xs.xs import Xs
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("dxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = Xs.read_xs("u238.0.2")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> dxs = Dxs.from_sigma1(xs0K, Ein, M, T, Eout)

        # SIGMA1 algorithm:
        >>> dxs.pdf.iloc[::100]
        Eout
        1.80000     0.000005
        1.84004     0.001091
        1.88008     0.063339
        1.92012     1.102673
        1.96016     5.974293
        2.00020    10.438638
        2.04024     6.084029
        2.08028     1.221555
        2.12032     0.087118
        2.16036     0.002272
        dtype: float64
        """
        return self.data / self.integral

    def shift(self, dx: [float, np.ndarray, pd.DataFrame]):
        """
        Shift the Double Differential XS in the given axis and interpolate to get the values of the original axis

        Parameters
        ----------
        dx : float or np.ndarray or pd.Series or pd.DataFrame
            The shift value in the given axis. If a pd.DataFrame is given, the shift value is calculated according to
            the index or the columns of the pd.DataFrame (next argument to select).
        axis : str, optional
            The axis to shift the Double Differential XS. The default is "Eout".

        Returns
        -------
        DDxs
            The shifted Double Differential XS values in the original axis

        Examples
        --------
        # 0K xs data for U238:
        >>> from solid_cinel.core.xs.xs import Xs
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("dxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = Xs.read_xs("u238.0.2")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> dxs = Dxs.from_sigma1(xs0K, Ein, M, T, Eout)
        >>> dxs.data.iloc[::200].round(6)
        Eout
        1.80000     0.000049
        1.88008     0.575486
        1.96016    54.281606
        2.04024    55.278649
        2.12032     0.791546
        dtype: float64

        # Shift the DDXS with float:
        >>> recoil = kb * T / M
        >>> dxs.shift(recoil).data.iloc[::200].round(6)
        Eout
        1.80000     0.000000
        1.88008     0.557797
        1.96016    53.733203
        2.04024    55.817580
        2.12032     0.814398
        dtype: float64

        # Shift the DDXS in the Eout axis:
        >>> recoil = Eout * kb * T / M
        >>> dxs.shift(recoil).data.iloc[::200].round(6)
        Eout
        1.80000     0.000000
        1.88008     0.542661
        1.96016    53.207633
        2.04024    56.378145
        2.12032     0.840643
        dtype: float64
        """
        # copy data to avoid changing the original data:
        dxs = self.data.copy()

        # Check the dx:
        dx_ = check_dx(self.data, dx, 0)

        # Shift the data:
        if isinstance(dx, float) or isinstance(dx, int):
            dxs = reshift(dxs, dx_)
        else:
            dxs.loc[dx_.index] = reshift(dxs.loc[dx_.index], dx_)

        return Dxs(self.Ein, self.T, self.M, dxs)


def check_dx(data: [pd.DataFrame, pd.Series],
             dx: [float, np.ndarray, pd.DataFrame],
             axis: [str, int]) -> [float, pd.Series, pd.DataFrame]:
    """
    Check the dx value to shift the Double Differential XS and return the value
    in the correct format for the shift aplicattion.

    Parameters
    ----------
    data : pd.DataFrame, pd.Series
        Double or Single Differential XS data to shift
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