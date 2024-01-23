"""
Python file for working with scattering functions.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
import os
from scipy.constants import physical_constants as const
from solid_cinel.core.generic import integrate, reshape_differential
from solid_cinel.core.scattering_function.beta import get_beta
from solid_cinel.core.scattering_function.alpha import get_alpha_mat, get_alpha_from_Eout, get_expansion_order
from solid_cinel.core.scattering_function.sab import get_sab_sct, get_sab_sct_alpha, Sab
from solid_cinel.core.material.vibration.pdos import Pdos
from solid_cinel.core.material.vibration.tau import save_tau
from typing import Iterable
from math import sqrt, pi
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import euclidean
from scipy.spatial import distance
import warnings

# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]

# Example variables:
interv_in_energy_U238 = 6.956193E-04
rho_in_energy_U238_str = '''
0.000000E+00 1.041128E-01 3.759952E-01 8.354039E-01
1.469796E+00 2.335578E+00 3.467660E+00 4.841392E+00
6.492841E+00 8.608376E+00 1.131303E+01 1.504441E+01
2.006807E+01 2.750471E+01 4.171597E+01 1.585670E+02
1.978483E+02 1.144621E+02 7.555927E+01 4.831100E+01
4.389081E+01 4.246484E+01 4.103699E+01 3.986249E+01
3.827959E+01 3.592088E+01 3.272170E+01 3.914602E+01
8.144694E+01 9.693959E+01 5.503795E+01 2.619253E+01
1.763331E+01 1.475875E+01 1.522465E+01 1.213117E+01
6.175029E+00 2.483519E+00 1.445581E+00 1.423177E+00
1.502350E+00 1.718768E+00 2.211346E+00 3.061686E+00
3.550530E+00 3.349917E+00 2.768379E+00 2.177488E+00
1.856123E+00 1.622775E+00 1.445254E+00 1.300794E+00
1.180078E+00 1.075748E+00 9.928057E-01 9.238564E-01
8.577708E-01 8.073819E-01 7.634820E-01 7.172257E-01
6.728183E-01 6.251482E-01 5.496737E-01 4.992486E-01
3.945195E-01 2.206960E-01 1.452214E-01 1.246671E-01
9.863893E-02 7.855588E-02 6.536053E-02 6.568678E-02
7.308199E-02 8.388478E-02 1.026265E-01 1.245221E-01
1.487740E-01 1.757085E-01 2.055793E-01 2.473042E-01
3.128097E-01 3.455081E-01 3.048708E-01 1.621507E-01
2.653572E-02 0.000000E+00 0.000000E+00 0.000000E+00
0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00
0.000000E+00 7.105193E-03 5.274518E-02 1.324974E-01
2.310275E-01 4.042710E-01 6.421137E-01 8.073457E-01
9.162074E-01 1.077923E+00 1.142595E+00 1.092532E+00
1.060668E+00 1.000020E+00 8.769838E-01 7.610532E-01
6.898200E-01 6.324347E-01 5.857072E-01 5.563076E-01
5.468099E-01 5.515587E-01 4.871045E-01 3.198787E-01
1.132118E-01 2.066306E-03 0.000000E+00
'''
rho_in_energy_U238 = np.fromstring(rho_in_energy_U238_str, dtype=np.float64,
                                   sep=' ')


class ScatFuncSD:
    """
    Single Differencial (angle or Outgoing energy) scattering function base
    class.
    """

    def __init__(self, Ein: float, T: float, M: float,  *args, **kwargs):
        """
        Initialize the ScatFuncSD class.

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
        normalization = integrate(pdf_)
        if abs(normalization - 1) >= 0.1 and self.Ein >= 0.005:
            raise ValueError(f"The scattering function is not normalized ({normalization} < 0.9)")
        elif abs(normalization - 1) >= 0.01:
            warnings.warn("Normalizaton not satisfied with 1% accuracy")
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
        ScatFuncSD
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
        >>> pdf = ScatFuncSD.from_sigma1(Ein, M, T, Eout)
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
        Eout_ = np.array(Eout) if hasattr(Eout, '__len__') else np.array([Eout])
        return cls(Ein, T, M, sigma1(Eout_, Ein, T, M),
                   index=pd.Index(Eout_, name="Eout"))

    @classmethod
    def from_theta(cls, Ein: float, M: float, T: float, Eout: np.array,
                   theta: float, *args, model: str = "fgm", **kwargs):
        """
        Generate the single differential scattering function from a selected
        angle using S(alpha, -beta) table.
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
        Eout : np.array
            The neutron outgoing energy grid in eV
        theta : float
            The angle of the distribution in degrees
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
            Minimun value to take into account in the creation of tau_n
            functions. For T>200 is convenient to set into 1.0e-14 to speed up
            the calculations. The default is 0.0.
        nphonon : 'int', optional
            Phonon expansion order. The default is 1000.

        Returns
        -------
        ScatFuncSD
            Single differential scattering function for the selected angle

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
        >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = 60

        # Using the Free Gas Model:
        >>> ScatFuncSD.from_theta(Ein, M, T, Eout, theta, model="fgm").data.loc[Eout_test].round(6)
        6.7554    0.000000
        6.9050    0.005957
        7.0439    1.196663
        7.2000    5.057344
        7.3157    0.733417
        7.4480    0.003848
        dtype: float64

        # Using the Short Collision Time model:
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> ScatFuncSD.from_theta(Ein, M, T, Eout, theta, pdos, model="sct").data.loc[Eout_test].round(6)
        6.7554    0.000000
        6.9050    0.006089
        7.0439    1.200917
        7.2000    5.050131
        7.3157    0.737298
        7.4480    0.003939
        dtype: float64

        # Using the Phonon expansion model:
        >>> ScatFuncSD.from_theta(Ein, M, T, Eout, theta, pdos, threshold=1.0e-14, model="pdos").data.loc[Eout_test].round(6)
        6.7554    0.000005
        6.9050    0.010414
        7.0439    1.179140
        7.2000    5.157191
        7.3157    0.705790
        7.4480    0.004105
        dtype: float64
        """
        mu = np.cos(np.deg2rad(theta))
        if model.lower() == "pdos":
            return cls.from_pdos(Ein, M, T, Eout, mu, *args, **kwargs)
        elif model.lower() == "sct":
            return cls.from_sct(Ein, M, T, Eout, mu, *args, **kwargs)
        else:
            return cls.from_fgm(Ein, M, T, Eout, mu)

    @classmethod
    def from_pdos(cls, Ein: float, M: float, T: float, Eout: np.array,
                  mu: float, pdos, nphonon: int = None,
                  decimal: float = 1.0e-6,
                  n_order_max: int = 5000, threshold: float = 0.0,
                  tau_to_file: bool = False,
                  binary: bool = False):
        """
        Generate the single differential scattering function from a selected
        angle using S(alpha, -beta) table based on Phonon expansion model.

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
        mu : float
            The cosine of the angle of the distribution in degrees
        pdos: 'solid_cinel.core.material.Pdos'
            Pdos object.
        nphonon: 'int', optional
            Phonon expansion order. The default is None and the order is
            calculated using the get_expansion_order function.
        decimal: 'float', optional
            Decimal precision for the calculation of the expansion order.
            The default is 1.0e-6.
        n_order_max: 'int', optional
            Maximun expansion order. The default is 5000.
        threshold: 'float', optional
            Minimun value to take into account in the creation of tau_n
            functions
        tau_to_file: 'bool', optional
            Save tau_n functions to file. The default is False.
        binary: 'bool', optional
            Save tau_n functions to binary file. The default is False.

        Returns
        -------
        ScatFuncSD
            Single differential scattering function for the selected angle using
            S(alpha, -beta) table based on Phonon expansion model

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
        >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> mu = 0.5
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> ScatFuncSD.from_pdos(Ein, M, T, Eout, mu, pdos, threshold=1.0e-14).data.loc[Eout_test].round(6)
        6.7554    0.000005
        6.9050    0.010414
        7.0439    1.179140
        7.2000    5.157191
        7.3157    0.705790
        7.4480    0.004105
        dtype: float64
        """
        debye_waller_coeff = pdos.DebyeWallerCoeff(T)
        delta_beta = pdos.to_beta_grid(T).grid
        if nphonon:
            warnings.warn(
                "Is posible that the expansion order is not enough to get the correct results")
        else:
            nphonon = get_expansion_order(get_alpha_from_Eout(Eout, Ein, M, T, mu),
                                          debye_waller_coeff, decimal, n_order_max)
        tau_n = pdos.get_tau(T, nphonon, threshold, values=True)
        save_tau(tau_n, nphonon, T, tau_to_file, binary)
        mu = np.array(mu) if hasattr(mu, '__len__') else np.array([mu])
        scattfunc = get_scatfunc_pdos(Ein, M, T, Eout, mu, tau_n, delta_beta, debye_waller_coeff)[0]
        norm = np.trapz(scattfunc, x=Eout)
        return cls(Ein, T, M, scattfunc / norm, index=Eout)

    @classmethod
    def from_sct(cls, Ein: float, M: float, T: float, Eout: np.array,
                    mu: float, pdos, ws: float = 1.0):
        """
        Generate the single differential scattering function from a selected
        angle using S(alpha, -beta) table based on Short Collision Time model.

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
        mu : float
            The cosine of the angle of the distribution in degrees
        pdos: 'solid_cinel.core.material.Pdos'
            Pdos object.
        ws: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Returns
        -------
        ScatFuncSD
            Single differential scattering function for the selected angle using
            S(alpha, -beta) table based on Short Collision Time model

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
        >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> mu = 0.5
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> ScatFuncSD.from_sct(Ein, M, T, Eout, mu, pdos).data.loc[Eout_test].round(6)
        6.7554    0.000000
        6.9050    0.006089
        7.0439    1.200917
        7.2000    5.050131
        7.3157    0.737298
        7.4480    0.003939
        dtype: float64
        """
        Teff = pdos.Teff(T)
        scattfunc = get_scat_sct_angular(Eout, mu, Ein, T, M, Teff, ws)
        norm = np.trapz(scattfunc, x=Eout)
        return cls(Ein, T, M, scattfunc / norm, index=Eout)

    @classmethod
    def from_fgm(cls, Ein: float, M: float, T: float, Eout: np.array,
                 mu: float, ws: float = 1.0):
        """
        Generate the single differential scattering function from a selected
        angle using S(alpha, -beta) table based on Free Gas Model.

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
        mu : float
            The cosine of the angle of the distribution in degrees
        ws: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Returns
        -------
        ScatFuncSD
            Single differential scattering function for the selected angle using
            S(alpha, -beta) table based on Free Gas Model

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
        >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> mu = 0.5
        >>> ScatFuncSD.from_fgm(Ein, M, T, Eout, mu).data.loc[Eout_test].round(6)
        6.7554    0.000000
        6.9050    0.005957
        7.0439    1.196663
        7.2000    5.057344
        7.3157    0.733417
        7.4480    0.003848
        dtype: float64
        """
        scattfunc = get_scat_sct_angular(Eout, mu, Ein, T, M, T, ws)
        norm = np.trapz(scattfunc, x=Eout)
        return cls(Ein, T, M, scattfunc / norm, index=Eout)

    @classmethod
    def from_recoil(cls, Ein: float, M: float, T: float, Eout: np.array,
                    *args, model: str = "fgm", **kwargs):
        """
        Generate the single differential scattering function from gressier
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
            Minimun value to take into account in the creation of tau_n
            functions. For T>200 is convenient to set into 1.0e-14 to speed up
            the calculations. The default is 0.0.
        nphonon : 'int', optional
            Phonon expansion order. The default is 1000.

        Returns
        -------
        ScatFuncSD
            Single differential scattering function using S(alpha, -beta) table
            based on the gressier recoil energy

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
        >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> ScatFuncSD.from_recoil(Ein, M, T, Eout, model="fgm").data.loc[Eout_test].round(6)
        6.7554    0.000000
        6.9050    0.005968
        7.0439    1.180420
        7.2000    5.102312
        7.3157    0.709369
        7.4480    0.003058
        dtype: float64
        """
        beta = get_beta(Eout, Ein, T)
        sab = Sab.from_recoil(Ein, T, M, beta,*args, model=model,
                              **kwargs).full
        scatfunc = np.interp(Eout, Ein + sab.index.values * kb * T, sab.values)
        return cls(Ein, T, M, scatfunc / (kb * T), index=Eout)

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



class ScatFuncDD:
    """
    Double Differencial (angle, Outgoing energy) scattering function base
    class.
    """

    def __init__(self, Ein: float, T: float, M: float,  *args, **kwargs):
        """
        Initialize the ScatFuncSD class.

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
        dd_pdf_ = pd.DataFrame(dd_pdf).sort_index(axis=0).sort_index(axis=1)
        dd_pdf_.index.name = "mu"
        dd_pdf_.columns.name = "Eout"
        normalization = integrate(dd_pdf_.apply(integrate))
        if abs(normalization - 1) >= 0.1 and self.Ein <= 0.005:
            raise ValueError(f"The scattering function is not normalized ({normalization} < 0.9)")
        elif abs(normalization - 1) >= 0.01:
            warnings.warn("Normalizaton not satisfied with 1% accuracy")
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
            calculated using the get_expansion_order function.
        decimal: 'float', optional
            Decimal precision for the calculation of the expansion order.
            The default is 1.0e-6.
        n_order_max: 'int', optional
            Maximun expansion order. The default is 5000.
        threshold: 'float', optional
            Minimun value to take into account in the creation of tau_n
            functions
        tau_to_file: 'bool', optional
            Save tau_n functions to file. The default is False.
        binary: 'bool', optional
            Save tau_n functions to binary file. The default is False.

        Returns
        -------
        ScatFuncSD
            Double differential scattering scattering function

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])

        # Using the Free Gas Model:
        >>> ScatFuncDD.from_model(Ein, M, T, Eout, theta, model="fgm").data.round(6)
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
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> ScatFuncDD.from_model(Ein, M, T, Eout, theta, pdos, model="sct").data.round(6)
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

        >>> ScatFuncDD.from_model(Ein, M, T, Eout, theta, pdos, threshold=1.0e-14, model="pdos").data.loc[::, Eout_test].round(6)
        Eout         6.7554    6.9050    7.0439    7.2000    7.3157    7.4480
        mu
        -0.939693  0.109061  0.644157  1.346117  1.029210  0.373643  0.053219
        -0.500000  0.034511  0.426488  1.383082  1.262613  0.415630  0.042074
         0.173648  0.000519  0.073364  1.103240  1.912878  0.440892  0.013328
         0.766044  0.000000  0.000012  0.077506  4.022814  0.127645  0.000019
        """
        mu = np.cos(np.deg2rad(theta))
        if model.lower() == "pdos":
            return cls.from_pdos(Ein, M, T, Eout, mu, *args, **kwargs)
        elif model.lower() == "sct":
            return cls.from_sct(Ein, M, T, Eout, mu, *args, **kwargs)
        else:
            return cls.from_fgm(Ein, M, T, Eout, mu)

    @classmethod
    def from_pdos(cls, Ein: float, M: float, T: float, Eout: np.ndarray,
                  mu: np.ndarray, pdos, nphonon: int = None,
                  decimal: float = 1.0e-6,
                  n_order_max: int = 5000, threshold: float = 0.0,
                  tau_to_file: bool = False,
                  binary: bool = False):
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
            calculated using the get_expansion_order function.
        decimal: 'float', optional
            Decimal precision for the calculation of the expansion order.
            The default is 1.0e-6.
        n_order_max: 'int', optional
            Maximun expansion order. The default is 5000.
        threshold: 'float', optional
            Minimun value to take into account in the creation of tau_n
            functions
        tau_to_file: 'bool', optional
            Save tau_n functions to file. The default is False.
        binary: 'bool', optional
            Save tau_n functions to binary file. The default is False.

        Returns
        -------
        ScatFuncDD
            Double differential scattering function from a S(alpha, -beta) table
            based on Phonon expansion model.

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
        >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([40, 80, 120, 160])
        >>> mu = np.cos(np.deg2rad(theta))
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> ScatFuncDD.from_pdos(Ein, M, T, Eout, mu, pdos, threshold=1.0e-14).data.loc[::, Eout_test].round(6)
        Eout         6.7554    6.9050    7.0439    7.2000    7.3157    7.4480
        mu
        -0.939693  0.109061  0.644157  1.346117  1.029210  0.373643  0.053219
        -0.500000  0.034511  0.426488  1.383082  1.262613  0.415630  0.042074
         0.173648  0.000519  0.073364  1.103240  1.912878  0.440892  0.013328
         0.766044  0.000000  0.000012  0.077506  4.022814  0.127645  0.000019
        """
        debye_waller_coeff = pdos.DebyeWallerCoeff(T)
        delta_beta = pdos.to_beta_grid(T).grid
        if nphonon:
            warnings.warn(
                "Is posible that the expansion order is not enough to get the correct results")
        else:
            alpha_max = get_alpha_from_Eout(Eout, Ein, M, T, mu.min())
            nphonon = get_expansion_order(alpha_max, debye_waller_coeff,
                                          decimal, n_order_max)
        tau_n = pdos.get_tau(T, nphonon, threshold, values=True)
        save_tau(tau_n, nphonon, T, tau_to_file, binary)
        return cls.from_tau(Ein, M, T, Eout, mu, tau_n, delta_beta,
                            debye_waller_coeff)

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
        ScatFuncDD
            Double differential scattering function from a S(alpha, -beta) table
            based on Short Collision Time model

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
        >>> mu = np.cos(np.deg2rad(theta))
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> ScatFuncDD.from_sct(Ein, M, T, Eout, mu, pdos).data.round(6)
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
        Teff = pdos.Teff(T)
        scattfunc = get_scat_sct_angular(Eout, mu, Ein, T, M, Teff, ws)
        return cls(Ein, T, M, scattfunc, index=mu, columns=Eout)

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
        ScatFuncDD
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
        >>> ScatFuncDD.from_fgm(Ein, M, T, Eout, mu).data.round(6)
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
        scattfunc = get_scat_sct_angular(Eout, mu, Ein, T, M, T, ws)
        return cls(Ein, T, M, scattfunc, index=mu, columns=Eout)

    @classmethod
    def from_tau(cls, Ein: float, M: float, T: float, Eout: np.ndarray,
                 mu: np.ndarray, tau_n: np.ndarray, delta_beta: float,
                 debye_waller_coeff: float):
        """
        Generate the double differential scattering function from tau_n function
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
        tau_n: np.ndarray, (Z, T)
            Tau_n function. Z is the number of phonon expansion order and T is
            the number of beta grid points.
        delta_beta: float
            Beta grid spacing in tau_n function
        debye_waller_coeff: float
            Debye-Waller coefficient in LEAPR formalism

        Returns
        -------
        ScatFuncSD
            Double differential scattering function

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
        >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([40, 80, 120, 160])
        >>> mu = np.cos(np.deg2rad(theta))
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> debye_waller_coeff = pdos.DebyeWallerCoeff(T)
        >>> delta_beta = pdos.to_beta_grid(T).grid
        >>> nphonon = get_expansion_order(get_alpha_from_Eout(Eout, Ein, M, T, mu.min()), debye_waller_coeff, 1.0e-6, 5000)
        >>> tau_n = pdos.get_tau(T, nphonon, 1.0e-14, values=True)
        >>> ScatFuncDD.from_tau(Ein, M, T, Eout, mu, tau_n, delta_beta, debye_waller_coeff).data.loc[::, Eout_test].round(6)
        Eout         6.7554    6.9050    7.0439    7.2000    7.3157    7.4480
        mu
        -0.939693  0.109061  0.644157  1.346117  1.029210  0.373643  0.053219
        -0.500000  0.034511  0.426488  1.383082  1.262613  0.415630  0.042074
         0.173648  0.000519  0.073364  1.103240  1.912878  0.440892  0.013328
         0.766044  0.000000  0.000012  0.077506  4.022814  0.127645  0.000019
        """
        scattfunc = get_scatfunc_pdos(Ein, M, T, Eout, mu,
                                      tau_n, delta_beta,
                                      debye_waller_coeff)
        return cls(Ein, T, M, scattfunc, index=mu, columns=Eout)

    def to_sd(self, theta: float = None) -> ScatFuncSD:
        """
        Convert the double differential scattering function to a single
        differential scattering function finding the angular distribution
        closest to sigma1 distribution or using the given angle.

        Parameters
        ----------
        theta : float, optional
            Angle to filter the scattering function. If None, the angular
            distribution more similar to sigma1 algorith will be used.

        Returns
        -------
        ScatFuncSD
            Single differential scattering function

        Examples
        --------
        # Generate double differencial Scattering function:
        >>> Ein = 7.2
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
        >>> ddScatFunc = ScatFuncDD.from_model(Ein, M, T, Eout, theta)
        >>> ddScatFunc.to_sd().data.round(6)
        Eout
        6.7554    0.000000
        6.9050    0.006232
        7.0439    1.251925
        7.2000    5.290892
        7.3157    0.767286
        7.4480    0.004026
        Name: 0.5000000000000001, dtype: float64

        >>> ddScatFunc.to_sd(theta=60).data.round(6)
        Eout
        6.7554    0.000000
        6.9050    0.006232
        7.0439    1.251925
        7.2000    5.290892
        7.3157    0.767286
        7.4480    0.004026
        Name: 0.5000000000000001, dtype: float64
        """
        filt_angle = np.cos(np.deg2rad(theta)) if theta else self.get_angle
        scattfunc = self.data.loc[filt_angle]
        scattfunc /= integrate(self.data.loc[filt_angle])
        return ScatFunc(self.Ein, self.T, self.M, scattfunc)


    @property
    def get_angle(self) -> float:
        """
        Get the angle of the double differential scattering function closest to
        the sigma1 distribution.

        Returns
        -------
        float
            Angle of the double differential scattering function closest to the
            sigma1 distribution.

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.linspace(Ein * 0.9, Ein * 1.1, 3000)
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
        >>> ddScatFunc = ScatFuncDD.from_model(Ein, M, T, Eout, theta)
        >>> round(ddScatFunc.get_angle, 2)
        0.5
        """
        scatfunc = self.data
        if (scatfunc.iloc[::, [0, -1]] >= 1.0e-6).any().any():
            warnings.warn("The distribution tails are not longer enough. Mu fit"
                          " will only take into account the max value.")
            angular_max = scatfunc.max(axis=1) / scatfunc.apply(integrate,
                                                                axis=1)
            MD = sigma1(np.array([self.Ein]), self.Ein, self.T, self.M)[0]
            mu_fit_max = abs(angular_max - MD).idxmin()
            return mu_fit_max
        else:
            sigma1_pdf = ScatFuncSD.from_sigma1(self.Ein, self.M, self.T,
                                        scatfunc.columns.values).data
            return mu_fit_calc(scatfunc, sigma1_pdf, self.Ein).mode()[0]

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


class ScatFunc(ScatFuncSD, ScatFuncDD):
    """
    Scattering function class
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the scattering function class. Depending on the shape of the
        scattering function, the class will be initialized as a ScatFuncSD or
        ScatFuncDD class.

        Parameters
        ----------
        Ein : float
            The neutron incident energy in eV
        T : float
            Temperature of the material in K
        M : float
            Mass of the material in amu
        args : Iterable, (N,) or (N, M)
            The scattering function data for the pd.DataFrame or pd.Series
        kwargs : dict
            Optional arguments for the construction of the pd.DataFrame
            pd.Series

        Examples
        --------
        Initilization of a scattering function for a pd.DataFrame:
        >>> Ein = 7.2
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
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

        >>> Ein = 36.68723
        >>> Eout = np.linspace(Ein * 0.98 , Ein * 1.02, 1000)
        >>> M = 238.05077040419212
        >>> T = 300
        >>> ScatFunc.from_sigma1(Ein, M, T, Eout).data.iloc[::100]
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
        if len(args) == 1 and isinstance(args[0], (ScatFuncSD, ScatFuncDD)):
            self.instance = args[0]
        elif len(args[-1].shape) == 1:
            self.instance = ScatFuncSD(*args, **kwargs)
        elif len(args[-1].shape) == 2:
            self.instance = ScatFuncDD(*args, **kwargs)
        else:
            raise ValueError("Invalid input")

    @classmethod
    def from_model(cls, Ein: float, M: float, T: float, Eout: np.array,
                 theta: [np.ndarray, float], *args, model: str = "fgm", **kwargs):
        if hasattr(theta, '__len__'):
            scatfunc = ScatFuncDD.from_model(Ein, M, T, Eout, theta,
                                             *args, model=model, **kwargs)
        else:
            scatfunc = ScatFuncSD.from_model(Ein, M, T, Eout, theta,
                                             *args, model=model, **kwargs)
        return cls(scatfunc)

    # called when an attribute is not found:
    def __getattr__(self, name):
        # assume it is implemented by self.instance
        return self.instance.__getattribute__(name)

    def convolve(self, xs: [pd.Series, pd.DataFrame, np.ndarray],
                 Exs: np.array = None,
                 integral: bool = False) -> [pd.DataFrame, pd.Series, float]:
        """
        Convolve the scattering function with a cross section. If the cross
        section is a matrix, the scattering function is convolved directly
        with xs. On the other hand, if the cross section is a pd.Series, the
        cross section is linearly interpolated to the energy grid of the
        scattering function or the provided energy grid Exs.

        Parameters
        ----------
        xs : pd.Series or pd.DataFrame or np.ndarray, (N,) or (M, N) or (M, N)
            Cross section to convolve with the scattering function. If a
            pd.DataFrame is provided, the scattering function is convolved with
            the xs directly.
        Exs : np.array, optional, (N,) or (M, N)
            Displazed Energy grid of the cross section. If not provided, the
            energy grid of the scattering function is used.
        integral : bool, optional
            If True, the integral value of the doppler broadening is returned.

        Returns
        -------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("scatfunc.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate 1D Scattering function:
        >>> Ein = 36.68723
        >>> Eout = np.linspace(Ein * 0.98 , Ein * 1.02, 1000)
        >>> M = 238.05077040419212
        >>> T = 300
        >>> scattering_function = ScatFunc.from_sigma1(Ein, M, T, Eout)
        >>> scattering_function.convolve(xs_0K).iloc[::100]
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
        dtype: float64

        >>> round(scattering_function.convolve(xs_0K, integral=True), 2)
        7905.42

        # Convolve with 0K displaced cross section:
        >>> Eout_move = Eout + kb * T
        >>> scattering_function.convolve(xs_0K, Exs=Eout_move).iloc[::100]
        Eout
        35.953485    8.426475e-14
        36.100381    4.174107e-08
        36.247277    1.423077e-03
        36.394173    3.936983e+00
        36.541069    1.422147e+03
        36.687964    5.107707e+04
        36.834860    9.047164e+02
        36.981756    5.986015e+00
        37.128652    4.271668e-03
        37.275548    2.629969e-07
        dtype: float64

        >>> round(scattering_function.convolve(xs_0K, Exs=Eout_move, integral=True), 2)
        7605.61

        # Generate 2D Scattering function:
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> T = 1000
        >>> theta = np.arange(0, 180, 1)[1::]
        >>> scattering_function = ScatFunc.from_model(Ein, M, T, Eout, theta)
        >>> scattering_function.convolve(xs_0K).iloc[::18, ::200].round(6)
        Eout        1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -0.999848  1.845717  12.094245  23.732354  15.005372  3.265822
        -0.945519  1.696431  11.865713  24.032880  15.196201  3.210537
        -0.798636  1.312725  11.171664  24.885947  15.737882  3.040859
        -0.573576  0.799665   9.866638  26.318562  16.647563  2.715775
        -0.292372  0.330178   7.768991  28.345006  17.934350  2.179578
         0.017452  0.066314   4.869372  30.864798  19.534647  1.412050
         0.325568  0.002865   1.834191  33.286657  21.073911  0.566000
         0.601815  0.000002   0.178412  33.109271  20.967417  0.063077
         0.819152  0.000000   0.000129  21.693622  13.741224  0.000068
         0.956305  0.000000   0.000000   0.381820   0.241879  0.000000

        # Convolve with 0K cross section and get integral value:
        >>> round(scattering_function.convolve(xs_0K, integral=True), 2)
        9.07

        # Compare with Sigma1 algorithm:
        >>> round(ScatFunc.from_sigma1(Ein, M, T, Eout).convolve(xs_0K, integral=True), 2)
        9.09

        # Use a displaced xs for the convolution (1D desplacement):
        >>> Eout_move = Eout + kb * T
        >>> scattering_function.convolve(xs_0K, Exs=Eout_move).iloc[::18, ::200].round(6)
        Eout        1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -0.999848  1.842263  12.070931  23.685592  14.975245  3.259065
        -0.945519  1.693256  11.842839  23.985526  15.165691  3.203895
        -0.798636  1.310268  11.150128  24.836912  15.706284  3.034568
        -0.573576  0.798169   9.847618  26.266704  16.614139  2.710156
        -0.292372  0.329561   7.754014  28.289155  17.898342  2.175069
         0.017452  0.066190   4.859985  30.803983  19.495427  1.409129
         0.325568  0.002859   1.830655  33.221070  21.031600  0.564829
         0.601815  0.000002   0.178068  33.044033  20.925319  0.062947
         0.819152  0.000000   0.000129  21.650877  13.713635  0.000068
         0.956305  0.000000   0.000000   0.381067   0.241393  0.000000

        >>> round(scattering_function.convolve(xs_0K, Exs=Eout_move, integral=True), 2)
        9.05

        # Use a displaced xs for the convolution (2D desplacement):
        >>> Eout_move = Eout + np.outer(np.cos(np.deg2rad(theta)), np.sqrt(Eout)/M)
        >>> scattering_function.convolve(xs_0K, Exs=Eout_move).iloc[::18, ::200].round(6)
        Eout        1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -0.999848  1.845492  12.092687  23.729226  15.003274  3.265355
        -0.945519  1.696235  11.864267  24.029885  15.194192  3.210104
        -0.798636  1.312597  11.170515  24.883327  15.736124  3.040513
        -0.573576  0.799609   9.865909  26.316572  16.646228  2.715553
        -0.292372  0.330167   7.768698  28.343913  17.933617  2.179487
         0.017452  0.066314   4.869383  30.864869  19.534695  1.412054
         0.325568  0.002865   1.834268  33.288086  21.074870  0.566026
         0.601815  0.000002   0.178426  33.111897  20.969181  0.063083
         0.819152  0.000000   0.000129  21.695964  13.742798  0.000068
         0.956305  0.000000   0.000000   0.381868   0.241911  0.000000
        >>> round(scattering_function.convolve(xs_0K, Exs=Eout_move, integral=True), 2)
        9.07
        """
        # Create the xs matrix for the double differential scattering function:
        if len(xs.shape) == 1:
            if Exs is not None:
                E = Exs.copy()
            elif self.data.index.name == "mu":
                E = self.data.columns.values
            else:
                E = self.data.index.values
            xs_reshaped = reshape_differential(xs, E)

        elif len(xs.shape) == 2:
            xs_reshaped = xs.values if isinstance(xs, pd.DataFrame) else xs
        else:
            raise ValueError("xs must be 1D or 2D")

        # Convolve the scattering function with the cross section values matrix:
        scattfunc_conv = self.data * xs_reshaped

        # Return the data in the chosen format:
        if integral and self.data.index.name == "mu":
            return integrate(scattfunc_conv.apply(integrate))
        elif integral:
            return integrate(scattfunc_conv)
        else:
            return scattfunc_conv


@nb.jit(nopython=True, nogil=True, cache=True)
def sigma1(Eout: np.array, Ein: float, T: float, M: float) -> np.array:
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
    scattfunc : np.array
        Scattering function based on sigma1 model

    Examples
    --------
    >>> Ein = 7.2
    >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> pd.Series(sigma1(Eout, Ein, T, M), index=Eout).round(6)
    6.7554    0.000000
    6.9050    0.001153
    7.0439    0.522804
    7.2000    5.501786
    7.3157    1.568599
    7.4480    0.017808
    dtype: float64
    """
    exp_negative = np.exp(
        - M / (m * kb * T) * (sqrt(Ein) - np.sqrt(Eout)) ** 2)
    exp_positive = np.exp(
        - M / (m * kb * T) * (sqrt(Ein) + np.sqrt(Eout)) ** 2)
    scattfunc = 0.5 * (exp_negative - exp_positive) * np.sqrt(Eout) / Ein
    scattfunc *= sqrt(M / (pi * m * kb * T))
    return scattfunc


@nb.jit(nopython=True, cache=True, nogil=True)
def get_scat_sct_angular(Eout: np.ndarray, mu: np.ndarray, Ein: float, T: float,
                         M: float, Teff: float, ws: float) -> np.array:
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
    beta = (Eout - Ein) / (kb * T)
    Tratio = Teff / T
    if isinstance(mu, (int, float)):
        alpha = get_alpha_from_Eout(Eout, Ein, T, M, mu)
        sab_values = get_sab_sct_alpha(alpha, beta, Tratio, ws)
    else:
        alpha = get_alpha_mat(Eout, Ein, T, M, mu)
        sab_values = get_sab_sct(alpha, beta, Tratio, ws)
    return sab_values * normalization_factor(Eout, Ein, T, M)


@nb.jit(nopython=True, cache=True, nogil=True)
def get_sab_pdos(alpha: np.ndarray, beta: np.ndarray,
                 tau_n: np.ndarray, tau_n_beta: np.ndarray,
                 DebyeWallerCoeff: float) -> np.ndarray:
    """
    Generate the scattering function from a S(alpha, -beta) table based on
    the phonon expansion model using a single angle.

    Parameters
    ----------
    alpha : 'np.ndarray', (Z, N)
        alpha grid values.
    beta : 'np.ndarray', (N,)
        beta grid values.
    nphonon : 'int', optional
        Phonon expansion order.
    tau_n : 'np.ndarray', (M, T)
        all tau n functions in one array.
    tau_n_beta : 'np.ndarray', (M,)
        Space between beta grid points of tau n functions.
    DebyeWallerCoeff : 'float'
        Debye Waller Coefficient in LEAPR formalism.

    Returns
    -------
    S_diag : 'np.ndarray', (Z, N)
        S(alpha, -beta) values for the alpha and beta combinations.

    Examples
    --------
    >>> Ein = 7.2
    >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
    >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> mu = np.cos(np.deg2rad([120]))
    >>> beta = get_beta(Eout, Ein, T)
    >>> alpha_mat = get_alpha_mat(beta * kb * T + Ein, Ein, T, M, mu)
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> debye_waller_coeff = pdos.DebyeWallerCoeff(T)
    >>> delta_beta = pdos.to_beta_grid(T).grid
    >>> nphonon = get_expansion_order(alpha_mat, debye_waller_coeff, 1.0e-6, 5000)
    >>> tau_n = pdos.get_tau(T, nphonon, 1.0e-14, values=True)
    >>> tau_n_beta = np.arange(tau_n.shape[1]) * delta_beta
    >>> sab_values = get_sab_pdos(alpha_mat, beta, tau_n, tau_n_beta, debye_waller_coeff)
    >>> pd.DataFrame(sab_values, index=[120], columns=beta).T.iloc[::100].round(6)
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
    iter_sum = np.log(alpha * DebyeWallerCoeff)
    alpha_mul = np.exp(- alpha * DebyeWallerCoeff + iter_sum)
    S_diag = alpha_mul * np.interp(beta, tau_n_beta, tau_n[0])

    # Higher phonon expansion (nphonon >= 1):
    for n in range(1, tau_n.shape[0]):
        # Compute S(alpha, -beta) for tau_n reshape
        iter_sum += np.log(alpha * DebyeWallerCoeff / (n + 1))
        alpha_mul = np.exp(- alpha * DebyeWallerCoeff + iter_sum)
        S_diag += alpha_mul * np.interp(beta, tau_n_beta, tau_n[n])
    return S_diag


@nb.jit(nopython=True, nogil=True, cache=True)
def normalization_factor(Eout: np.ndarray, Ein: float, T: float, M: float) -> np.ndarray:
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


@nb.jit(nopython=True, nogil=True, cache=True)
def scatfunc_values_alpha_vec(Sab_mat: np.ndarray, beta: np.ndarray, Ein: float,
                        T: float, M: float) -> (np.ndarray, np.ndarray):
    """
    Generate the scattering function values from a S(alpha, -beta) table based on
    the phonon expansion model for a single angle

    Parameters
    ----------
    Sab_mat : 'np.ndarray', (N,)
        S(alpha, -beta) matrix values.
    beta: 'np.ndarray', (N,)
        Minus beta grid values.
    Ein : 'float'
        Incident energy in eV.
    T : 'float'
        Temperature in K.
    M : 'float'
        Mass of the target nucleus in amu.

    Returns
    -------
    'np.ndarray', (N, 2)
        Scattering function values for a single angle for tau_n expansion.

    Examples
    --------
    >>> Ein = 7.2
    >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
    >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> mu = np.cos(np.deg2rad([120]))
    >>> beta = get_beta(Eout, Ein, T)
    >>> alpha_mat = get_alpha_from_Eout(beta * kb * T + Ein, Ein, T, M, mu)
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> debye_waller_coeff = pdos.DebyeWallerCoeff(T)
    >>> delta_beta = pdos.to_beta_grid(T).grid
    >>> nphonon = get_expansion_order(alpha_mat, debye_waller_coeff, 1.0e-6, 5000)
    >>> tau_n = pdos.get_tau(T, nphonon, 1.0e-14, values=True)
    >>> tau_n_beta = np.arange(tau_n.shape[1]) * delta_beta
    >>> sab_values = get_sab_pdos(alpha_mat, beta, tau_n, tau_n_beta, debye_waller_coeff)
    >>> Eout_calc, scatfunc_values = scatfunc_values_alpha_vec(sab_values, beta, Ein, T, M)
    >>> pd.Series(scatfunc_values, index=Eout_calc).iloc[::200].round(6)
    6.755400    0.036933
    6.894059    0.381006
    6.991813    1.027908
    7.060847    1.470584
    7.129778    1.569324
    7.199108    1.238803
    7.268142    0.716554
    7.337073    0.307374
    7.406107    0.098213
    7.501782    0.012632
    7.640440    0.000258
    dtype: float64
    """
    # Scattering function values calculation:
    ScatFunc_values = np.concatenate((Sab_mat[::-1], Sab_mat[1::] * np.exp(-beta[1:])))

    # Eout calculation
    Eout_calc = Ein + np.concatenate((-beta[::-1], beta[1::])) * kb * T

    # Ensure the Eout values are positive:
    positive_mask = Eout_calc > 0
    Eout_calc = Eout_calc[positive_mask]

    # Normalization constant
    norm = normalization_factor(Eout_calc, Ein, T, M)

    return Eout_calc, ScatFunc_values[positive_mask] * norm

@nb.jit(nopython=True, cache=True, nogil=True)
def scatfunc_values_alpha_mat(Sab_values: np.ndarray, beta: np.ndarray, Ein: float,
                              T: float, M: float) -> (np.ndarray, np.ndarray):
    """
    Generate the scattering function from a S(alpha, -beta) table based on
    the phonon expansion model. The scattering function is calculated for all
    the angles and the outgoing energy grid is calculated based on the beta
    grid.

    Parameters
    ----------
    Sab_values: 'np.ndarray', (Z, N)
        S(alpha, -beta) for the selected alpha and beta.
    beta: 'np.ndarray', (N,)
        beta grid values.
    Ein: 'float'
        Incident energy in eV.
    T: 'float'
        Temperature in K.
    M: 'float'
        Mass of the target in amu.

    Returns
    -------
    Eout_calc: 'np.ndarray', (N,)
        Outgoing energy grid in eV.
    ScatFunc_values: 'np.ndarray', (Z, N)
        Scattering function values for all the angles and Eout calculation.

    Examples
    --------
    >>> Ein = 7.2
    >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
    >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> mu = np.cos(np.deg2rad([120]))
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> debye_waller_coeff = pdos.DebyeWallerCoeff(T)
    >>> delta_beta = pdos.to_beta_grid(T).grid
    >>> nphonon = get_expansion_order(get_alpha_from_Eout(Eout, Ein, M, T, mu), debye_waller_coeff, 1.0e-6, 5000)
    >>> tau_n = pdos.get_tau(T, nphonon, 1.0e-14, values=True)
    >>> tau_n_beta = np.arange(tau_n.shape[1]) * delta_beta
    >>> beta = get_beta(Eout, Ein, T)
    >>> alpha_mat = get_alpha_mat(beta * kb * T + Ein, Ein, T, M, mu)
    >>> sab_values = get_sab_pdos(alpha_mat, beta, tau_n, tau_n_beta, debye_waller_coeff)
    >>> Eout_calc, scatfunc_values = scatfunc_values_alpha_mat(sab_values, beta, Ein, T, M)
    >>> pd.DataFrame(scatfunc_values, index=[120], columns=Eout_calc).T.iloc[::200].round(6)
                   120
    6.755400  0.036933
    6.894059  0.381006
    6.991813  1.027907
    7.060847  1.470583
    7.129778  1.569324
    7.199108  1.238803
    7.268142  0.716554
    7.337073  0.307374
    7.406107  0.098213
    7.501782  0.012632
    7.640440  0.000258
    """
    ScatFunc_values = np.concatenate(
        (Sab_values[::, ::-1], Sab_values[::, 1:] * np.exp(-beta[1:])), axis=1)
    # Eout calculation
    Eout_calc = np.sort(Ein + np.concatenate((-beta[::-1], beta[1::])) * kb * T)

    # Ensure the Eout values are positive:
    positive_mask = Eout_calc > 0
    Eout_calc = Eout_calc[positive_mask]

    # Normalization constant
    norm = normalization_factor(Eout_calc, Ein, T, M)

    return Eout_calc, ScatFunc_values[::, positive_mask] * norm


@nb.jit(nopython=True, nogil=True, cache=True)
def get_scatfunc_pdos(Ein: float, M: float, T: float, Eout: np.ndarray,
                      mu: np.ndarray, tau_n: np.ndarray, delta_beta: float,
                      DebyeWallerCoeff: float) -> np.ndarray:
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
    tau_n : 'np.ndarray', (M, T)
        all tau n functions in one array.
    tau_n_beta : 'np.ndarray', (M,)
        Space between beta grid points of tau n functions.
    DebyeWallerCoeff : float
        Debye Waller coefficient

    Returns
    -------
    S_diag : 'np.ndarray', (N,)
        Scattering function values for a single angle.

    Examples
    --------
    >>> Ein = 7.2
    >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
    >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> mu = np.cos(np.deg2rad([120]))
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> debye_waller_coeff = pdos.DebyeWallerCoeff(T)
    >>> delta_beta = pdos.to_beta_grid(T).grid
    >>> nphonon = get_expansion_order(get_alpha_from_Eout(Eout, Ein, M, T, mu), debye_waller_coeff, 1.0e-6, 5000)
    >>> tau_n = pdos.get_tau(T, nphonon, 1.0e-14, values=True)
    >>> sd_pdf = get_scatfunc_pdos(Ein, M, T, Eout, mu, tau_n, delta_beta, debye_waller_coeff)
    >>> pd.Series(sd_pdf[0], index=Eout).loc[Eout_test].round(6)
    6.7554    0.034510
    6.9050    0.426488
    7.0439    1.383081
    7.2000    1.262613
    7.3157    0.415630
    7.4480    0.042074
    dtype: float64
    """
    tau_n_beta = np.arange(tau_n.shape[1]) * delta_beta
    beta = get_beta(Eout, Ein, T)
    alpha_mat = get_alpha_mat(beta * kb * T + Ein if len(beta) < len(Eout) else Eout,
                              Ein, T, M, mu)
    sab_values = get_sab_pdos(alpha_mat, beta, tau_n, tau_n_beta, DebyeWallerCoeff)
    Eout_calc, scatfunc_values = scatfunc_values_alpha_mat(sab_values, beta, Ein, T, M)
    # Interpolation for avoiding numerical fluctuations:
    select_scarfunc = np.zeros((len(mu), len(Eout)))
    for i in range(len(mu)):
        select_scarfunc[i] += np.interp(Eout, Eout_calc, scatfunc_values[i])
    return select_scarfunc


@nb.jit(nopython=True, nogil=True, cache=True)
def get_scatfunc_pdos_row(Ein: float, M: float, T: float, Eout: np.ndarray,
                 mu: float, tau_n: np.ndarray, delta_beta: float,
                 DebyeWallerCoeff: float) -> np.ndarray:
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
    tau_n : 'np.ndarray', (M, T)
        all tau n functions in one array.
    tau_n_beta : 'np.ndarray', (M,)
        Space between beta grid points of tau n functions.
    DebyeWallerCoeff : float
        Debye Waller coefficient

    Returns
    -------
    S_diag : 'np.ndarray', (N,)
        Scattering function values for a single angle.

    Examples
    --------
    >>> Ein = 7.2
    >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
    >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> mu = np.cos(np.deg2rad(120))
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> debye_waller_coeff = pdos.DebyeWallerCoeff(T)
    >>> delta_beta = pdos.to_beta_grid(T).grid
    >>> nphonon = get_expansion_order(get_alpha_from_Eout(Eout, Ein, M, T, mu), debye_waller_coeff, 1.0e-6, 5000)
    >>> tau_n = pdos.get_tau(T, nphonon, 1.0e-14, values=True)
    >>> sd_pdf = get_scatfunc_pdos_row(Ein, M, T, Eout, mu, tau_n, delta_beta, debye_waller_coeff)
    >>> pd.Series(sd_pdf, index=Eout).loc[Eout_test].round(6)
    6.7554    0.034510
    6.9050    0.426488
    7.0439    1.383081
    7.2000    1.262613
    7.3157    0.415630
    7.4480    0.042074
    dtype: float64
    """
    tau_n_beta = np.arange(tau_n.shape[1]) * delta_beta
    beta = get_beta(Eout, Ein, T)
    Eout_ = beta * kb * T + Ein if len(beta) < len(Eout) else Eout
    alpha = get_alpha_from_Eout(Eout_, Ein, T, M, mu)
    sab_values = get_sab_pdos(alpha, beta, tau_n, tau_n_beta, DebyeWallerCoeff)
    Eout_calc, scatfunc_values = scatfunc_values_alpha_vec(sab_values, beta, Ein, T, M)
    # Interpolation for avoiding numerical fluctuations:
    return np.interp(Eout, Eout_calc, scatfunc_values)


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Total Variation Distance between two probability distributions.

    Parameters
    ----------
    p : np.ndarray, (N,)
        Probability distribution.
    q : np.ndarray, (N,)
        Probability distribution.

    Returns
    -------
    float
        Total Variation Distance.
    """
    return 0.5 * np.sum(np.abs(p - q))


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Hellinger Distance between two probability distributions.

    Parameters
    ----------
    p : np.ndarray, (N,)
        Probability distribution.
    q : np.ndarray, (N,)
        Probability distribution.

    Returns
    -------
    float
        Hellinger Distance.
    """
    return euclidean(np.sqrt(p), np.sqrt(q)) / np.sqrt(2)


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray, Eout: np.ndarray) -> float:
    """
    Bhattacharyya Distance between two probability distributions.

    Parameters
    ----------
    p : np.ndarray, (N,)
        Probability distribution.
    q : np.ndarray, (N,)
        Probability distribution.
    Eout : np.ndarray, (N,)
        Energy grid.

    Returns
    -------
    float
        Bhattacharyya Distance for continous distribution.
    """
    BC = np.trapz(np.sqrt(p * q), x=Eout)
    return -np.log(BC)


def mu_fit_calc(scatfunc: pd.DataFrame, sigma1_pdf: pd.Series, Ein: float) -> pd.DataFrame:
    """
    Calculate the angle from the scattering function that best fits the
    sigma1_pdf. The distributions from the scattering function are shifted
    using the maximun position of the scattering function and the incident
    energy distance. For the calculation, several distances are used:
        - KL divergence
        - Jensen-Shannon divergence
        - Earth Mover's Distance
        - Total Variation Distance
        - Hellinger Distance
        - Bhattacharyya Distance
        - Maximun position distance

    Parameters
    ----------
    scatfunc : pd.DataFrame, (N, M)
        Scattering function distribution for each angle.
    sigma1_pdf : pd.Series, (M, )
        Sigma1 distribution.
    Ein : float
        Incident energy.

    Returns
    -------
    mu_fit : pd.Series, (7, )
        Best fit angle for each distance calculation.

    Examples
    --------
    >>> Ein = 7.2
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> Eout = np.linspace(Ein * 0.9, Ein * 1.1, 3000)
    >>> sigma1 = ScatFunc.from_sigma1(Ein, M, T, Eout).data
    >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
    >>> ddScatFunc = ScatFuncDD.from_model(Ein, M, T, Eout, theta).data
    >>> mu_fit_calc(ddScatFunc, sigma1, Ein).round(2)
    KL               0.5
    JS               0.5
    EMD              0.5
    TVD              0.5
    Hellinger        0.5
    Bhattacharyya    0.5
    max_pos          0.5
    dtype: float64
    """
    Eout = scatfunc.columns.values
    scatfunc_norm = scatfunc.apply(lambda x: x / integrate(x), axis=1)
    def get_distances(angular_scatfunc):
        angular_recoil = Ein - angular_scatfunc.idxmax()
        scatfunc_shift = np.interp(Eout, Eout + angular_recoil,
                                   angular_scatfunc.values)
        return pd.Series({
            "KL": entropy(scatfunc_shift, sigma1_pdf),
            "JS": distance.jensenshannon(scatfunc_shift, sigma1_pdf),
            "EMD": wasserstein_distance(scatfunc_shift, sigma1_pdf),
            "TVD": total_variation_distance(scatfunc_shift, sigma1_pdf),
            "Hellinger": hellinger_distance(scatfunc_shift, sigma1_pdf),
            "Bhattacharyya": bhattacharyya_distance(scatfunc_shift,
                                                    sigma1_pdf, Eout),
            "max_pos": abs(scatfunc_shift.max() - sigma1_pdf.max()),
        })
    return scatfunc_norm.apply(get_distances, axis=1).idxmin()
