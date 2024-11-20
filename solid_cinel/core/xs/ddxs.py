"""
Python for working with Double Diferential XS.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
import os
from scipy.constants import physical_constants as const
from solid_cinel.core.dynamic_structure.dynamicStruc import DynamicStruc, DoubleDiffData
from solid_cinel.core.material.pdos import Pdos
from solid_cinel.core.xs.nucInteract import NucInteract
from solid_cinel.core.generic import integrate
from solid_cinel.core.xs.xs0K import Xs0K

# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]

# Avoid numba fast math:
nb.config.FASTMATH_DEFAULT = False


class DDxs(DoubleDiffData):
    """
    Class for the Double differential cross section for elastic scattering

    Attributes
    ----------
    xs0K : Xs0K
        Xs object with the cross section xs data for the given material in barns
    Ein : float
        The incident energy of the neutron in eV
    T : float
        Temperature of the material in K

    Properties
    ----------
    scatFunc : pd.Series
        The Scattering function of the Double Differential XS for inelastic
        scattering
    angularDistr : pd.Series
        The angular probability distribution of the Double Differential XS
    angleIntegrated : float
        The integral value of the Double Differential XS
    upscattering : float
        The upscattering probability of the Double Differential XS
    downscattering : float
        The downscattering probability of the Double Differential XS

    Methods
    -------
    from_Sab -> DDxs
        Generate the Double Differential XS for elastic scattering from
        S(alpha, -beta) tables
    from_4PCF -> DDxs
        Generate the Double Differential XS for elastic scattering from Fourier
        double-Laplace transform of a 4-point correlation function
    """

    def __init__(self, xs0K: Xs0K, Ein: float, T: float, *args, **kwargs):
        """
        Class for the Double differential cross section for inelastic scattering

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
        # Atributes of the Double Differential XS:
        self.xs0K = xs0K
        self.Ein = Ein
        self.T = T
        # The ddxs data:
        super().__init__(*args, **kwargs)

    @property
    def scatFunc(self) -> pd.Series:
        """
        The Scattering function of the Double Differential XS for inelastic
        scattering

        Returns
        -------
        ScatFunc
            The Scattering function of the Double Differential XS for inelastic
            scattering

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("ddxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs = Xs0K.from_file("u238.0.2", M)
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> theta = np.arange(0, 180, 1)[1::]

        # Angular distribution:
        >>> DDxs.from_Sab(xs, Ein, T, Eout, theta, model="fgm").scatFunc.iloc[::200].round(6)
        Eout
        1.80000     0.768794
        1.88008    10.451361
        1.96016    54.522950
        2.04024    34.506930
        2.12032     2.920481
        dtype: float64
        """
        return super().columsIntegral

    @property
    def angularDistr(self) -> pd.Series:
        """
        Get angular probability distribution of the Double Differential XS

        Returns
        -------
        pd.Series
            The angular probability distribution of the Double Differential XS

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("ddxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs = Xs0K.from_file("u238.0.2", M)
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> theta = np.arange(0, 180, 15)[1::]
        >>> ddxs = DDxs.from_Sab(xs, Ein, T, Eout, theta)
        >>> ddxs.angularDistr.round(6)
        mu
        -9.659258e-01    4.455539
        -8.660254e-01    4.469560
        -7.071068e-01    4.489373
        -5.000000e-01    4.510598
        -2.588190e-01    4.529192
         6.123234e-17    4.543335
         2.588190e-01    4.554042
         5.000000e-01    4.563068
         7.071068e-01    4.570756
         8.660254e-01    4.576664
         9.659258e-01    4.580387
        dtype: float64

        """
        return super().rowIntegral

    @property
    def angleIntegrated(self) -> float:
        """
        The integral value of the Double Differential XS

        Returns
        -------
        float
            The integral value of the Double Differential XS

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("ddxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs = Xs0K.from_file("u238.0.2", M)
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> theta = np.arange(0, 180, 1)[1::]
        >>> from solid_cinel.tests.materials.UO2.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)

        # S(alpha, -beta) algorithm for FGM:
        >>> float(round(DDxs.from_Sab(xs, Ein, T, Eout, theta, model="fgm").angleIntegrated, 2))
        9.07
        """
        return super().doubleIntegral

    @property
    def upscattering(self) -> float:
        """
        Get the upscattering probability of the Double Differential XS
        Returns
        -------
        float
            The upscattering probability of the Double Differential XS

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("ddxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs = Xs0K.from_file("u238.0.2", M)
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> theta = np.arange(0, 180, 15)[1::]
        >>> ddxs = DDxs.from_Sab(xs, Ein, T, Eout, theta)
        >>> assert round(ddxs.upscattering, 6) == 0.389484
        """
        return integrate(super().columsPdf[self.Eout > self.Ein])

    @property
    def downscattering(self) -> float:
        """
        Get the downscattering probability of the Double Differential XS

        Returns
        -------
        float
            The downscattering probability of the Double Differential XS

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("ddxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs = Xs0K.from_file("u238.0.2", M)
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> theta = np.arange(0, 180, 15)[1::]
        >>> ddxs = DDxs.from_Sab(xs, Ein, T, Eout, theta)
        >>> assert round(ddxs.downscattering, 6) == 0.60678
        """
        return integrate(super().columsPdf[self.Eout < self.Ein])

    @classmethod
    def from_Sab(cls, xs0K: Xs0K, Ein: float, T: float, Eout: np.ndarray, theta: np.ndarray, *args,
                 **kwargs) -> "DDxs":
        """
        Generate the Double Differential XS for elastic scattering from
        S(alpha, -beta) tables
        ..math::
            \frac{d^2\sigma_T(E)}{dE^\prime d^\theta} = \frac{\sigma_b}{2 * k_B * T}\sqrt{\frac{E^\prime}{E}} S(\alpha(\theta, E^\prime, E, M, T), \beta( E^\prime, E, T))

        Common Parameters for fgm, sct and pdos models
        ----------------------------------------------
        xs0K: Xs0K
            Xs object with the cross section xs data for the given material in barns
        Ein : float
        The incident energy of the neutron in eV
        T : float
            Temperature of the material in K
        Eout : np.ndarray, (N,)
            The neutron outgoing energy grid in eV
        theta : np.ndarray, (M,)
            The neutron outgoing angle grid in degrees (0, 180]

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
        nphonon : 'int', optional
            Phonon expansion order. The default is 1000.

        Returns
        -------
        DDxs
            Double differential cross section for elastic scattering

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("ddxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs = Xs0K.from_file("u238.0.2", M)
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.array([1.8, 1.88008, 1.96016, 2.04024, 2.12032])
        >>> theta = np.array([40, 80, 120, 160])
        >>> from solid_cinel.tests.materials.UO2.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)

        # S(alpha, -beta) algorithm for FGM:
        >>> DDxs.from_Sab(xs, Ein, T, Eout, theta, model="fgm").data.round(6)
        Eout        1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -0.939693  1.680633  11.840323  24.065568  15.217007  3.204407
        -0.500000  0.656538   9.369964  26.822570  16.967671  2.590290
         0.173648  0.018247   3.283892  32.168542  20.363097  0.977203
         0.766044  0.000000   0.002708  26.843192  17.002521  0.001206

        # S(alpha, -beta) algorithm for SCT:
        >>> DDxs.from_Sab(xs, Ein, T, Eout, theta, pdos, model="sct").data.round(6)
        Eout        1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -0.939693  1.693186  11.848713  24.024229  15.215447  3.221508
        -0.500000  0.663895   9.388389  26.778357  16.967086  2.607195
         0.173648  0.018701   3.305738  32.128467  20.370559  0.987929
         0.766044  0.000000   0.002800  26.885282  17.056633  0.001251

        # S(alpha, -beta) algorithm for PDOS:
        >>> DDxs.from_Sab(xs, Ein, T, Eout, theta, pdos, threshold=1.0e-14, model="pdos").data.round(6)
        Eout        1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -0.939693  1.724269  11.665751  24.157346  15.274187  3.160754
        -0.500000  0.724235   9.192329  26.910247  17.022369  2.543125
         0.173648  0.041094   3.335304  32.065389  20.299803  0.982933
         0.766044  0.000016   0.040475  24.162304  15.423499  0.012977
        """
        # Get the Dynamic structure factor:
        scatfunction = DynamicStruc.from_model(Ein, xs0K.M, T, Eout, theta, *args, **kwargs)

        # Get the cross section in the correct energy grid:
        xs0Kinterp = xs0K.interpolate(Eout, values=True)

        return cls(xs0K, Ein, T, scatfunction.data * xs0Kinterp)

    @classmethod
    def from_4PCF(cls, xs0K: Xs0K, Ein: float, T: float, Eout: np.ndarray,
                  theta: np.ndarray, *args, approx: bool = True,
                  kind: str = "corrected", **kwargs) -> "DDxs":
        """
        Generate the Double Differential XS for elastic scattering from Fourier
        double-Laplace transform of a 4-point correlation function
        ..math::
            \frac{d^2\sigma_T(E)}{dE^\prime d^\theta} = \frac{1}{2 * k_B * T}\sqrt{\frac{E^\prime}{E}} S(\alpha(\theta, E^\prime, E, M, T), \beta( E^\prime, E, T)) \sigma^{T(1+\mu)/2}((E^\prime+E + \frac{\alpha k_{B} T}{1-\mu})/2 - E \mu / A)

        For the xs matrix calculation, they are the following models available:
            - "sigma1": sigma1 algorithm from NJOY2016 manual (default)
            - "fgm": Free Gas Model
            - "sct": Short Collision Time
            - "pdos": Phonon Density of States

        Common parameters
        -----------------
        xs : Xs
            Xs object with the cross section xs data for the given material in barns
        Ein : float
            The incident energy of the neutron in eV
        T : float
            Temperature of the material in K
        Eout : np.ndarray, (N,)
            The neutron outgoing energy grid in eV
        theta : np.ndarray, (M,)
            The neutron outgoing angle grid in degrees (0, 180]
        algorithm: str, optional
            The algorithm use for getting the angle-integrated xs. The options
            are:
                - "sigma1" (default)
                - "alpha0"

        Parameters for sct
        ------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object

        Parameters for pdos
        -------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object
        threshold : 'float', optional
            Minimun value to take into account in the creation of tauN functions. For T>200 is convenient to set into
            1.0e-14 to speed up the calculations. The default is 0.0.
        nphonon : 'int', optional
            Phonon expansion order. The default is 1000.

        Returns
        -------
        DDxs
            The Double Differential XS for elastic scattering

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("ddxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs = Xs0K.from_file("u238.0.2", M)
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 6.67
        >>> Eout = np.array([6.5, 6.6, 6.67, 6.8, 6.9])
        >>> theta = np.array([40, 80, 120, 160])
        >>> from solid_cinel.tests.materials.UO2.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)

        # Coercelle with sigma1 algorithm:
        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, kind="modified", model="fgm")
        >>> ddxs_test.data.round(6)
        Eout             6.50         6.60         6.67       6.80      6.90
        mu
        -0.939693  206.115432  1013.812172   165.200220  15.898128  2.399646
        -0.500000  276.409513   820.219197   464.709622  25.878328  2.277178
         0.173648  139.765613   717.641970   682.014198  54.201630  1.371722
         0.766044    2.327093   529.067649  1218.656999  10.579651  0.001333

        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, pdos, kind="modified", model="sct").data
        >>> ddxs_test.round(6)
        Eout             6.50         6.60         6.67       6.80      6.90
        mu
        -0.939693  205.830907  1012.196411   165.095610  15.954282  2.420805
        -0.500000  276.177089   818.816867   464.299330  25.974865  2.299886
         0.173648  139.990855   716.483327   681.151248  54.470850  1.391839
         0.766044    2.367526   529.455979  1216.705383  10.725959  0.001391

        # Coercelle with pdos model: (Example not very accurate, only for
        # demonstration purposes)
        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, pdos, kind="modified", threshold=1.0e-14, nphonon=100, model="pdos").data
        >>> ddxs_test.round(6)
        Eout             6.50        6.60         6.67       6.80      6.90
        mu
        -0.939693    0.467176    2.219360     0.315338   0.018165  0.001448
        -0.500000  121.397072  365.197611   200.688661   9.549414  0.674066
         0.173648  138.229730  717.583601   685.972166  53.718060  1.378087
         0.766044    3.454981  513.186644  1240.858754  11.210909  0.005570

        # alpha0 (still testing):
#        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, algorithm="alpha0", model="fgm").data
#        >>> ddxs_test.round(6)

#        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, pdos, algorithm="alpha0", model="sct").data
#        >>> ddxs_test.round(6)

#        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, pdos, algorithm="alpha0", threshold=1.0e-14, nphonon=100, model="pdos").data
#        >>> ddxs_test.round(6)
        """
        # Generate Dynamic structure of the phonon dynamics:
        dynamicStruc = DynamicStruc.from_model(Ein, xs0K.M, T, Eout, theta,
                                               *args, **kwargs).data

        # Get nuclear interaction parameters:
        nuclearInteract = NucInteract.from_sigma(xs0K, Ein, T, Eout, theta,
                                                 approx=approx, kind=kind).data

        # Convolve the dynamic structure with the nuclear Interaction:
        return cls(xs0K, Ein, T,  dynamicStruc * nuclearInteract)