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
        >>> DDxs.from_4PCF(xs, Ein, T, Eout, theta, model="fgm").scatFunc.iloc[::200].round(6)
        Eout
        1.80000     0.766980
        1.88008    10.435716
        1.96016    54.493213
        2.04024    34.519331
        2.12032     2.924035
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
        >>> ddxs = DDxs.from_4PCF(xs, Ein, T, Eout, theta)
        >>> ddxs.angularDistr.round(6)
        mu
        -9.659258e-01    4.452497
        -8.660254e-01    4.467374
        -7.071068e-01    4.487391
        -5.000000e-01    4.508718
        -2.588190e-01    4.527563
         6.123234e-17    4.542064
         2.588190e-01    4.553174
         5.000000e-01    4.562597
         7.071068e-01    4.570632
         8.660254e-01    4.576807
         9.659258e-01    4.580694
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
        >>> float(round(DDxs.from_4PCF(xs, Ein, T, Eout, theta, model="fgm").angleIntegrated, 2))
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
        >>> ddxs = DDxs.from_4PCF(xs, Ein, T, Eout, theta)
        >>> assert round(ddxs.upscattering, 2) == 0.39
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
        >>> ddxs = DDxs.from_4PCF(xs, Ein, T, Eout, theta)
        >>> assert round(ddxs.downscattering, 2) == 0.61
        """
        return integrate(super().columsPdf[self.Eout < self.Ein])

    @classmethod
    def from_4PCF(cls, xs0K: Xs0K, Ein: float, T: float, Eout: np.ndarray,
                  theta: np.ndarray, *args,
                  approx: bool = True, kind: str = "corrected", **kwargs):
        """
        Generate the Double Differential XS for elastic scattering from Fourier
        double-Laplace transform of a 4-point correlation function
        ..math::
            \frac{d^2\sigma_T(E)}{dE^\prime d^\theta} = \frac{1}{2 * k_B * T}\sqrt{\frac{E^\prime}{E}} S(\alpha(\theta, E^\prime, E, M, T), \beta( E^\prime, E, T)) \sigma^{T(1+\mu)/2}((E^\prime+E + \frac{\alpha k_{B} T}{1-\mu})/2 - E \mu / A)

        For the DSF, they are the following models available:
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
        >>> ddxs_test.data.round(5)
        Eout            6.50        6.60        6.67      6.80     6.90
        mu
        -0.939693  206.11543  1013.81217   165.20022  15.89813  2.39965
        -0.500000  276.40951   820.21920   464.70962  25.87833  2.27718
         0.173648  139.76561   717.64197   682.01420  54.20163  1.37172
         0.766044    2.32709   529.06765  1218.65700  10.57965  0.00133

        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, pdos, kind="modified", model="sct").data
        >>> ddxs_test.round(5)
        Eout            6.50        6.60        6.67      6.80     6.90
        mu
        -0.939693  205.83091  1012.19641   165.09561  15.95428  2.42081
        -0.500000  276.17709   818.81687   464.29933  25.97486  2.29989
         0.173648  139.99086   716.48333   681.15125  54.47085  1.39184
         0.766044    2.36753   529.45598  1216.70538  10.72596  0.00139

        # Coercelle with pdos model: (Example not very accurate, only for
        # demonstration purposes)
        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, pdos, kind="modified", threshold=1.0e-14, nphonon=100, model="pdos").data
        >>> ddxs_test.round(5)
        Eout            6.50       6.60        6.67      6.80     6.90
        mu
        -0.939693    0.46718    2.21936     0.31534   0.01816  0.00145
        -0.500000  121.39707  365.19761   200.68866   9.54941  0.67407
         0.173648  138.22973  717.58360   685.97217  53.71806  1.37809
         0.766044    3.45498  513.18664  1240.85875  11.21091  0.00557
        """
        # Generate Dynamic structure of the phonon dynamics:
        dynamicStruc = DynamicStruc.from_model(Ein, xs0K.M, T, Eout, theta,
                                               *args, **kwargs)

        # Get nuclear interaction parameters:
        nuclearInteract = NucInteract.from_fgm(xs0K, Ein, T, Eout, theta,
                                                 approx=approx,
                                                 kind=kind)

        # Convolve the dynamic structure with the nuclear Interaction:
        return cls(xs0K, Ein, T,  dynamicStruc.data * nuclearInteract.data)