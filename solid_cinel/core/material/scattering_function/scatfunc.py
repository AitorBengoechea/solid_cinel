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
from solid_cinel.core.material.scattering_function.beta import Beta, get_beta
from solid_cinel.core.material.scattering_function.alpha import Alpha, get_alpha
from solid_cinel.core.material.scattering_function.sab import Sab, get_S_pdos_from_alpha_beta, tau_n_CPU
from solid_cinel.core.material.vibration.pdos import Pdos
from typing import Iterable
from numba import prange
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
    def from_MD(cls, Ein: float, M: float, T: float, Eout: np.array):
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
        >>> pdf = ScatFuncSD.from_MD(Ein, M, T, Eout)
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
    def from_SabSD(cls, Ein: float, M: float, T: float, Eout: np.array,
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
        theta : np.array
            Grid of cosine of the scattering angle
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
            Double differential scattering scattering function

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
        >>> ScatFuncSD.from_SabSD(Ein, M, T, Eout, theta, model="fgm").data.loc[Eout_test].round(6)
        6.7554    0.000000
        6.9050    0.005957
        7.0439    1.196663
        7.2000    5.057344
        7.3157    0.733417
        7.4480    0.003848
        dtype: float64

        # Using the Short Collision Time model:
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> ScatFuncSD.from_SabSD(Ein, M, T, Eout, theta, pdos, model="sct").data.loc[Eout_test].round(6)
        6.7554    0.000000
        6.9050    0.006089
        7.0439    1.200917
        7.2000    5.050131
        7.3157    0.737298
        7.4480    0.003939
        dtype: float64

        # Using the Phonon expansion model:
        >>> ScatFuncSD.from_SabSD(Ein, M, T, Eout, theta, pdos, threshold=1.0e-14, model="pdos").data.loc[Eout_test].round(6)
        6.7554    0.001861
        6.9050    0.084836
        7.0439    0.126480
        7.2000    0.129529
        7.3157    1.532400
        7.4480    0.074382
        dtype: float64
        """
        mu = np.cos(theta * np.pi / 180)
        if model.lower() == "pdos":
            threshold = kwargs.pop("threshold", 0.0)
            nphonon = kwargs.pop("nphonon", 1000)
            tau1 = args[-1].get_tau_1(T)
            debye_waller_coeff = args[-1].DebyeWallerCoeff(T)
            scattfunc = get_ScatFunc_pdos_angle(Ein, M, T, Eout, mu, nphonon,
                                               tau1.values, tau1.index[1],
                                               threshold, debye_waller_coeff)
        else:
            ws = kwargs.pop("ws", 1.0)
            Teff = args[0].Teff(T) if model == "sct" else T
            scattfunc = get_scat_sct_angular(Eout, mu, Ein, T, M, Teff, ws)
        norm = integrate(pd.Series(scattfunc, index=Eout))
        return cls(Ein, T, M, scattfunc / norm, index=Eout)



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
    def from_SabDD(cls, Ein: float, M: float, T: float, Eout: np.array,
                 theta: np.array, *args, model: str = "fgm", **kwargs):
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
        Eout : np.array
            The neutron outgoing energy grid in eV
        theta : np.array
            Grid of cosine of the scattering angle
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
            Double differential scattering scattering function

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])

        # Using the Free Gas Model:
        >>> ScatFuncDD.from_SabDD(Ein, M, T, Eout, theta, model="fgm").data.round(6)
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
        >>> ScatFuncDD.from_SabDD(Ein, M, T, Eout, theta, pdos, model="sct").data.round(6)
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

        >>> ScatFuncDD.from_SabDD(Ein, M, T, Eout, theta, pdos, threshold=1.0e-14, model="pdos").data.loc[::, Eout_test].round(6)
        Eout         6.7554    6.9050    7.0439    7.2000    7.3157    7.4480
        mu
        -0.939693  0.109061  0.644157  1.346118  1.029210  0.373644  0.053219
        -0.500000  0.034511  0.426488  1.383082  1.262613  0.415630  0.042074
         0.173648  0.000519  0.073364  1.103240  1.912878  0.440892  0.013328
         0.766044  0.000000  0.000012  0.077506  4.022814  0.127645  0.000019
        """
        mu = np.cos(theta * np.pi / 180)
        if model.lower() == "pdos":
            scattfunc = scat_from_pdos(Ein, M, T, Eout, theta,
                                           *args, **kwargs)
        else:
            ws = kwargs.pop("ws", 1.0)
            Teff = args[0].Teff(T) if model == "sct" else T
            scattfunc = [get_scat_sct_angular(Eout, mu[j], Ein, T, M, Teff, ws)
                         for j in range(len(mu))]
        return cls(Ein, T, M, np.array(scattfunc),
                   index=mu, columns=Eout)

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
        >>> ddScatFunc = ScatFuncDD.from_SabDD(Ein, M, T, Eout, theta)
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
        filt_angle = np.cos(theta * np.pi / 180) if theta else self.get_angle
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
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
        >>> ddScatFunc = ScatFuncDD.from_SabDD(Ein, M, T, Eout, theta)
        >>> round(ddScatFunc.get_angle, 2)
        0.5
        """
        angular_max = self.data.max(axis=1) / self.data.apply(integrate, axis=1)
        MD = sigma1(np.array([self.Ein]), self.Ein, self.T, self.M)[0]
        return abs(angular_max - MD).idxmin()
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
        >>> ScatFunc.from_Sab(Ein, M, T, Eout, theta, model="fgm").data.round(6)
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
        >>> ScatFunc.from_MD(Ein, M, T, Eout).data.iloc[::100]
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
    def from_Sab(cls, Ein: float, M: float, T: float, Eout: np.array,
                 theta: [np.ndarray, float], *args, model: str = "fgm", **kwargs):
        if hasattr(theta, '__len__'):
            scatfunc = ScatFuncDD.from_SabDD(Ein, M, T, Eout, theta, *args, model=model, **kwargs)
        else:
            scatfunc = ScatFuncSD.from_SabSD(Ein, M, T, Eout, theta, *args, model=model, **kwargs)
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
        >>> os.chdir("../../../data/xs/U238/")
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate 1D Scattering function:
        >>> Ein = 36.68723
        >>> Eout = np.linspace(Ein * 0.98 , Ein * 1.02, 1000)
        >>> M = 238.05077040419212
        >>> T = 300
        >>> scattering_function = ScatFunc.from_MD(Ein, M, T, Eout)
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
        >>> scattering_function = ScatFunc.from_Sab(Ein, M, T, Eout, theta)
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
        >>> round(ScatFunc.from_MD(Ein, M, T, Eout).convolve(xs_0K, integral=True), 2)
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
        >>> Eout_move = Eout + np.outer(np.cos(theta * np.pi / 180), np.sqrt(Eout)/M)
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


@nb.jit(nopython=True, nogil=False, cache=True)
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
        - M / (m * kb * T) * (np.sqrt(Ein) - np.sqrt(Eout)) ** 2)
    exp_positive = np.exp(
        - M / (m * kb * T) * (np.sqrt(Ein) + np.sqrt(Eout)) ** 2)
    scattfunc = 0.5 * (exp_negative - exp_positive) * np.sqrt(Eout) / Ein
    scattfunc *= np.sqrt(M / (np.pi * m * kb * T))
    return scattfunc


@nb.jit(nopython=True, nogil=False, cache=True, parallel=False)
def get_scat_sct_angular(Eout: np.ndarray, mu: float, Ein: float, T: float,
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
    mu : float
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
    np.array, (N,)
        The scattering function values for a single angle
    """
    awr = ((M / m + 1) / (M / m)) ** 2
    beta = (Eout - Ein) / (kb * T)
    alpha = Eout + Ein - 2 * mu * np.sqrt(Eout * Ein)
    alpha /= (M * kb * T / m)
    scattfunc = np.exp(-(ws * alpha + beta) ** 2 / (4 * alpha * Teff / T * ws))
    scattfunc /= np.sqrt(4 * np.pi * ws * alpha * Teff / T)
    scattfunc *= awr * np.sqrt(Eout / Ein) / (2 * kb * T)
    return scattfunc


def scat_from_pdos(Ein: float, M: float, T: float, Eout: np.array,
                       theta: np.array, pdos: Pdos, threshold: float = 0.0,
                       nphonon: int = 1000) -> list:
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
    Eout : np.array
        The neutron outgoing energy grid in eV
    theta : np.array
        Grid of cosine of the scattering angle
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
    dd_pdf : dict
        Dictionary with the scattering function for each angle

    Examples
    --------
    >>> Ein = 7.2
    >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
    >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> theta = np.array([40, 80, 120, 160])
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> dd_pdf = scat_from_pdos(Ein, M, T, Eout, theta, pdos, threshold=1.0e-14)
    >>> pd.DataFrame(dd_pdf, index=np.cos(theta * np.pi / 180)).loc[:, Eout_test].round(6)
    Eout         6.7554    6.9050    7.0439    7.2000    7.3157    7.4480
       0.766044  0.000000  0.000012  0.077506  4.022814  0.127645  0.000019
       0.173648  0.000519  0.073364  1.103240  1.912878  0.440892  0.013328
      -0.500000  0.034511  0.426488  1.383082  1.262613  0.415630  0.042074
      -0.939693  0.109061  0.644157  1.346118  1.029210  0.373644  0.053219
        """
    beta = Beta.from_parameters(Eout, Ein, T).unique
    dd_pdf = []
    for angle in theta:
        alpha = Alpha.from_parameters(Eout, Ein, T, M, angle)
        angular_dd_pdf = Sab.from_pdos(alpha, beta, T, pdos,
                                       threshold=threshold,
                                       nphonon=nphonon)
        dd_pdf.append(angular_dd_pdf.to_ScatFunc(Ein, T, M).loc[Eout])
    return dd_pdf

@nb.jit(nopython=True, nogil=False, cache=True, parallel=False)
def get_diag_S_from_tau_n(tau: np.ndarray, beta_tau: np.ndarray,
                     debye_waller_coeff: float,  iter_sum: float,
                     alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Generate the scattering function from a S(alpha, -beta) table based on
    the phonon expansion model using a single angle for tau_n function.

    Parameters
    ----------
    tau : 'np.ndarray', (T,)
        tau function values.
    beta_tau : 'np.ndarray', (T,)
        beta grid for tau function.
    debye_waller_coeff : 'float'
        Debye Waller Coefficient.
    alpha : 1D iterable, (N,)
        Alpha grid.
    beta: 1D iterable, (N,)
        beta grid.

    Returns
    -------
    'np.ndarray', (N,)
        Scattering function values for a single angle for tau_n function.
    """
    alpha_mul = np.exp(- alpha * debye_waller_coeff + iter_sum)
    # Interpolate tau_n(-beta):
    tau_n_reshape = np.interp(beta, beta_tau, tau)
    # Bounds in nopython mode:
    if beta[-1] > beta_tau[-1]:
        tau_n_reshape[beta > beta_tau[-1]] = 0.0
    return alpha_mul * tau_n_reshape

@nb.jit(nopython=True, nogil=False, cache=True, parallel=False)
def get_diag_S_pdos(alpha: np.ndarray, beta: np.ndarray,
                    nphonon: int, tau1: np.ndarray, delta_beta: float,
                    threshold: float, DebyeWallerCoeff: float) -> np.ndarray:
    """
    Generate the scattering function from a S(alpha, -beta) table based on
    the phonon expansion model using a single angle.

    Parameters
    ----------
    alpha : 'np.ndarray', (N,)
        alpha grid values.
    beta : 'np.ndarray', (N,)
        beta grid values.
    nphonon : 'int', optional
        Phonon expansion order.
    tau1 : 'np.ndarray', (M,)
        tau1 function values.
    delta_beta : float
        Space between beta grid points.
    threshold : 'float', optional
        Minimun value to take into account in the creation of tau_n
        functions. For T>200 is convenient to set into 1.0e-14 to speed up
        the calculations.
    DebyeWallerCoeff : 'float'
        Debye Waller Coefficient in LEAPR formalism.

    Returns
    -------
    S_diag : 'np.ndarray', (N,)
        Scattering function values for a single angle.
    """
    if len(alpha) != len(beta):
        raise ValueError("alpha and beta must have the same length")

    tau_n_minus_1 = tau1.copy()
    # Zero phonon expansion:
    iter_sum = np.log(alpha * DebyeWallerCoeff)
    S_diag = get_diag_S_from_tau_n(tau1, np.arange(len(tau1)) * delta_beta,
                                DebyeWallerCoeff, iter_sum, alpha, beta)

    # Higher phonon expansion (nphonon >= 1):
    for n in range(1, nphonon + 1):
        # Tau_n(-beta)
        tau_n = tau_n_CPU(delta_beta, tau1, tau_n_minus_1, threshold)

        # Compute S(alpha, -beta) for tau_n reshape
        iter_sum += np.log(alpha * DebyeWallerCoeff / (n + 1))
        S_diag += get_diag_S_from_tau_n(tau_n, np.arange(len(tau_n)) * delta_beta,
                                        DebyeWallerCoeff, iter_sum, alpha, beta)

        # Next tau_n
        tau_n_minus_1 = tau_n
    return S_diag

@nb.jit(nopython=True, nogil=False, cache=True, parallel=False)
def get_ScatFunc_values(Sab_mat: np.ndarray, beta_grid: np.ndarray, Ein: float,
                        T: float, M: float) -> np.ndarray:
    """
    Generate the scattering function values from a S(alpha, -beta) table based on
    the phonon expansion model for a single angle

    Parameters
    ----------
    Sab_mat : 'np.ndarray', (N,)
        S(alpha, -beta) matrix values.
    beta_grid : 'np.ndarray', (N,)
        Minus beta grid values.
    Ein : 'float'
        Incident energy in eV.
    T : 'float'
        Temperature in K.
    M : 'float'
        Mass of the target nucleus in amu.

    Returns
    -------
    'np.ndarray', (N,)
        Scattering function values for a single angle for tau_n expansion.
    """
    # Get the scattering function values:
    ScatFunc_values = np.concatenate((Sab_mat[::-1], Sab_mat[1::]))
    ScatFunc_values[len(Sab_mat)::] *= np.exp(-beta_grid[1::])
    Eout = np.sort(Ein + np.concatenate((- beta_grid[::-1], beta_grid[1::])) * kb * T)
    # Avoid negative values and nan numbers:
    ScatFunc_values = ScatFunc_values[Eout > 0]
    ScatFunc_values[np.isnan(ScatFunc_values)] = 0
    Eout = Eout[Eout > 0]
    # Apply normalization constants:
    awr = ((M / m + 1) / (M / m)) ** 2
    ScatFunc_values *= awr * np.sqrt(Eout / Ein) / (2 * kb * T)
    return np.vstack((Eout, ScatFunc_values)).T

@nb.jit(nopython=True, nogil=False, cache=True, parallel=False)
def get_ScatFunc_pdos_angle(Ein: float, M: float, T: float, Eout: np.ndarray,
                 mu: float, nphonon: int, tau1: np.ndarray,
                 delta_beta: float, threshold: float,
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
    Eout : np.ndarray
        The neutron outgoing energy grid in eV
    mu : float
        Cosine of the scattering angle
    nphonon : int
        Phonon expansion order
    tau1 : np.ndarray
        Array with the tau values of the 1 phonon order
    delta_beta : float
        tau functions step size
    threshold : float
        Minimun value to take into account in the creation of tau_n
        functions. For T>200 is convenient to set into 1.0e-14 to speed up
        the calculations.
    DebyeWallerCoeff : float
        Debye Waller coefficient

    Examples
    --------
    >>> Ein = 7.2
    >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
    >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> mu = np.cos(60 / 180 * np.pi)
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> tau1 = pdos.get_tau_1(T)
    >>> debye_waller_coeff = pdos.DebyeWallerCoeff(T)
    >>> sd_pdf = get_ScatFunc_pdos_angle(Ein, M, T, Eout, mu, 1000, tau1.values, tau1.index[1], 1.0e-14, debye_waller_coeff)
    >>> pd.Series(sd_pdf, index=Eout).loc[Eout_test].round(6)
    6.7554     0.037459
    6.9050     1.707956
    7.0439     2.546358
    7.2000     2.607742
    7.3157    30.851057
    7.4480     1.497491
    dtype: float64
    """
    beta = (Eout - Ein) / (kb * T)
    alpha = Eout + Ein - 2 * mu * np.sqrt(Eout * Ein)
    alpha /= (M * kb * T / m)
    Sab_values = get_diag_S_pdos(alpha, beta, nphonon, tau1, delta_beta, threshold, DebyeWallerCoeff)
    sd_pdf = get_ScatFunc_values(Sab_values, beta, Ein, T, M)
    Eout_pos = np.searchsorted(sd_pdf[:, 0], Eout)
    return sd_pdf[Eout_pos, 1]

