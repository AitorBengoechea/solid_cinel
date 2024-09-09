"""
Python for working with Double Diferential XS.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
import os
from scipy.constants import physical_constants as const
from typing import Iterable
from solid_cinel.core.scattering_function.dynamicStruc import DynamicStruc
from solid_cinel.core.material.pdos import Pdos
from solid_cinel.core.xs import Xs, ScatFunc, NucInteract
from solid_cinel.core.generic import integrate, reshift
from solid_cinel.core.xs.scatfunc import check_dx

# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]

# Avoid numba fast math:
nb.config.FASTMATH_DEFAULT = False


class DDxs:
    """
    Class for the Double differential cross section for elastic scattering
    """

    def __init__(self, Ein: float, T: float, M: float, *args, **kwargs):
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
        self.Ein = Ein
        self.T = T
        self.M = M
        # The ddxs data:
        self.data = pd.DataFrame(*args, **kwargs)

    @property
    def data(self) -> pd.DataFrame:
        """
        DDXS data.

        Returns
        -------
        pd.DataFrame
            DDXS data
        """
        return self._data

    @data.setter
    def data(self, dd_pdf: Iterable):
        """
        Set the diferential data.

        Parameters
        ----------
        dd_pdf : pd.DataFrame
            Double differential scattering function data
        """
        dd_pdf_ = pd.DataFrame(dd_pdf).sort_index(axis=0).sort_index(axis=1)
        dd_pdf_.index.name = "mu"
        dd_pdf_.columns.name = "Eout"
        self._data = dd_pdf_

    @classmethod
    def from_Sab(cls, xs: Xs, Ein: float, T: float, Eout: np.ndarray, theta: np.ndarray, *args,
                 **kwargs):
        """
        Generate the Double Differential XS for elastic scattering from
        S(alpha, -beta) tables
        ..math::
            \frac{d^2\sigma_T(E)}{dE^\prime d^\theta} = \frac{\sigma_b}{2 * k_B * T}\sqrt{\frac{E^\prime}{E}} S(\alpha(\theta, E^\prime, E, M, T), \beta( E^\prime, E, T))

        Common Parameters for fgm, sct and pdos models
        ----------------------------------------------
        xs0K : Xs
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
        >>> xs = Xs.from_xs0K("u238.0.2", M)
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
        -0.939693  2.171256  11.840731  24.614351  15.555578  3.119537
        -0.500000  0.975833   9.405392  27.341180  17.288658  2.491187
         0.173648  0.064789   3.495596  32.299618  20.447022  0.940610
         0.766044  0.000025   0.043670  23.461207  14.958094  0.011931
        """
        # Get the Dynamic structure factor:
        scatfunction = DynamicStruc.from_model(Ein, xs.M, T, Eout, theta, *args, **kwargs)

        # Get the cross section in the correct energy grid:
        xs0Kinterp = xs.interp_Ein(Eout, T=0).loc[::, 0]

        # Calculate the convolution:
        ddxs = scatfunction.data * xs0Kinterp

        return cls(Ein, T, xs.M, ddxs)

    @classmethod
    def from_4PCF(cls, xs: Xs, Ein: float, T: float, Eout: np.ndarray,
                  theta: np.ndarray, *args, algorithm: str = "sigma1",
                  approx: bool = True, kind: str = "corrected", **kwargs):
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
        >>> xs = Xs.from_xs0K("u238.0.2", M)
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.array([1.8, 1.88008, 1.96016, 2.04024, 2.12032])
        >>> theta = np.array([40, 80, 120, 160])
        >>> from solid_cinel.tests.materials.UO2.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)

        # Coercelle with sigma1 algorithm:
        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, kind="modified", model="fgm").data
        >>> ddxs_test.round(6)
        Eout        1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -0.939693  1.676460  11.820750  24.047097  15.218935  3.207785
        -0.500000  0.655005   9.355903  26.806090  16.972566  2.593448
         0.173648  0.018206   3.279258  32.151616  20.370851  0.978487
         0.766044  0.000000   0.002704  26.832215  17.011020  0.001208

        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, pdos, kind="modified", model="sct").data
        >>> ddxs_test.round(6)
        Eout        1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -0.939693  1.688981  11.829126  24.005790  15.217375  3.224905
        -0.500000  0.662345   9.374300  26.761905  16.971981  2.610374
         0.173648  0.018659   3.301073  32.111562  20.378316  0.989226
         0.766044  0.000000   0.002796  26.874288  17.065159  0.001253

        # Coercelle with pdos model: (Example not very accurate, only for
        # demonstration purposes)
        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, pdos, kind="modified", threshold=1.0e-14, nphonon=100, model="pdos").data
        >>> ddxs_test.round(6)
        Eout        1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -0.939693  2.165875  11.821162  24.595460  15.557551  3.122829
        -0.500000  0.973555   9.391278  27.324381  17.293646  2.494225
         0.173648  0.064643   3.490663  32.282624  20.454809  0.941845
         0.766044  0.000025   0.043613  23.451613  14.965571  0.011948

        # alpha0 (still testing):
#        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, algorithm="alpha0", model="fgm").data
#        >>> ddxs_test.round(6)

#        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, pdos, algorithm="alpha0", model="sct").data
#        >>> ddxs_test.round(6)

#        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, pdos, algorithm="alpha0", threshold=1.0e-14, nphonon=100, model="pdos").data
#        >>> ddxs_test.round(6)
        """
        # Generate Dynamic structure of the phonon dynamics:
        dynamicStruc = DynamicStruc.from_model(Ein, xs.M, T, Eout, theta, *args, **kwargs).data

        # Use only Eout values with information for optimization:
        Eout_ = dynamicStruc.columns.values

        # Get nuclear interaction parameters:
        kwargs["algorithm"] = algorithm
        nuclearInteraction = NucInteract(xs.M, T, theta)
        if algorithm.lower() == "sigma1":
            xsMat = nuclearInteraction.from_sigma(xs, Ein, Eout_, approx=approx,
                                                  kind=kind)
        else:
            xsMat = xs.get_4PCFxs(Ein, T, Eout_, theta, *args, **kwargs)

        # Convolve the scattering function with the cross section matrix:
        ddxs = dynamicStruc * xsMat

        return cls(Ein, T, xs.M, ddxs)

    @property
    def angular(self) -> ScatFunc:
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
        >>> xs = Xs.from_xs0K("u238.0.2", M)
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> theta = np.arange(0, 180, 1)[1::]

        # Angular distribution:
        >>> DDxs.from_Sab(xs, Ein, T, Eout, theta, model="fgm").angular.data.iloc[::200].round(6)
        Eout
        1.80000     0.768794
        1.88008    10.451361
        1.96016    54.522950
        2.04024    34.506930
        2.12032     2.920481
        dtype: float64
        """
        scatfuncValues = self.data.apply(integrate, axis=0)
        return ScatFunc(self.Ein, self.T, self.M, scatfuncValues)

    @property
    def integral(self) -> float:
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
        >>> xs = Xs.from_xs0K("u238.0.2", M)
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> theta = np.arange(0, 180, 1)[1::]
        >>> from solid_cinel.tests.materials.UO2.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)

        # S(alpha, -beta) algorithm for FGM:
        >>> float(round(DDxs.from_Sab(xs, Ein, T, Eout, theta, model="fgm").integral, 2))
        9.07
        """
        return self.angular.integral

    @property
    def Eprob(self) -> dict:
        """
        Get the upscattering and downscattering probalities

        Returns
        -------
        dict
            Dictionary with the upscattering and downscattering probabilities

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("ddxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs = Xs.from_xs0K("u238.0.2", M)
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> theta = np.arange(0, 180, 15)[1::]
        >>> ddxs = DDxs.from_Sab(xs, Ein, T, Eout, theta)
        >>> probabilities = ddxs.Eprob
        >>> float(round(probabilities["upscattering"], 6))
        0.389484
        >>> float(round(probabilities["downscattering"], 6))
        0.60678
        >>> float(round(probabilities["Ein=Eout"], 6))
        0.003736
        """
        return self.angular.prob

    @property
    def AngleProb(self) -> pd.Series:
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
        >>> xs = Xs.from_xs0K("u238.0.2", M)
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> theta = np.arange(0, 180, 15)[1::]
        >>> ddxs = DDxs.from_Sab(xs, Ein, T, Eout, theta)
        >>> angular_prob = ddxs.AngleProb
        >>> angular_prob.round(6)
        mu
        -9.659258e-01    0.508586
        -8.660254e-01    0.510186
        -7.071068e-01    0.512448
        -5.000000e-01    0.514870
        -2.588190e-01    0.516993
         6.123234e-17    0.518607
         2.588190e-01    0.519829
         5.000000e-01    0.520860
         7.071068e-01    0.521737
         8.660254e-01    0.522412
         9.659258e-01    0.522836
        dtype: float64
        """
        angular_prob = self.data.apply(integrate, axis=1)
        return angular_prob / self.integral

    @property
    def pdf(self) -> pd.DataFrame:
        """
        Get the probability density function of the Double Differential XS

        Returns
        -------
        pd.DataFrame
            The probability density function of the Double Differential XS

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("ddxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs = Xs.from_xs0K("u238.0.2", M)
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> theta = np.arange(0, 180, 15)[1::]
        >>> ddxs = DDxs.from_Sab(xs, Ein, T, Eout, theta)
        >>> ddxs.pdf.iloc[::, ::200].round(6)
        Eout            1.80000   1.88008   1.96016   2.04024   2.12032
        mu
        -9.659258e-01  0.199996  1.364426  2.730286  1.726349  0.368894
        -8.660254e-01  0.169485  1.313193  2.795112  1.767512  0.356426
        -7.071068e-01  0.124606  1.218890  2.904967  1.837268  0.333172
        -5.000000e-01  0.074942  1.069553  3.061712  1.936798  0.295670
        -2.588190e-01  0.032968  0.854101  3.265153  2.065984  0.240293
         6.123234e-17  0.008520  0.575864  3.506323  2.219150  0.166594
         2.588190e-01  0.000812  0.279307  3.747518  2.372410  0.084671
         5.000000e-01  0.000009  0.066077  3.861538  2.445181  0.021837
         7.071068e-01  0.000000  0.002427  3.457385  2.189727  0.000966
         8.660254e-01  0.000000  0.000000  1.696249  1.074497  0.000000
         9.659258e-01  0.000000  0.000000  0.008420  0.005334  0.000000
        """
        return self.data / self.integral

    def shift(self, dx: [float, np.ndarray, pd.DataFrame], axis: [str, int] = "Eout"):
        """
        Shift the Double Differential XS in the given axis and interpolate to
        get the values of the original axis

        Parameters
        ----------
        dx : float or np.ndarray or pd.Series or pd.DataFrame
            The shift value in the given axis. If a pd.DataFrame is given, the
            shift value is calculated according to the index or the columns of
            the pd.DataFrame (next argument to select).
        axis : str, optional
            The axis to shift the Double Differential XS. The default is "Eout".

        Returns
        -------
        DDxs
            The shifted Double Differential XS values in the original axis

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("ddxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs = Xs.from_xs0K("u238.0.2", M)
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> theta = np.arange(0, 180, 15)[1::]
        >>> ddxs = DDxs.from_Sab(xs, Ein, T, Eout, theta)
        >>> ddxs.data.iloc[::, ::200].round(6)
        Eout            1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -9.659258e-01  1.752099  11.953256  23.919080  15.123940  3.231752
        -8.660254e-01  1.484794  11.504425  24.486993  15.484554  3.122526
        -7.071068e-01  1.091626  10.678265  25.449395  16.095659  2.918806
        -5.000000e-01  0.656538   9.369981  26.822586  16.967611  2.590263
        -2.588190e-01  0.288822   7.482483  28.604861  18.099364  2.105126
         6.123234e-17  0.074636   5.044945  30.717669  19.441192  1.459468
         2.588190e-01  0.007115   2.446909  32.830687  20.783854  0.741769
         5.000000e-01  0.000082   0.578875  33.829577  21.421371  0.191307
         7.071068e-01  0.000000   0.021259  30.288941  19.183426  0.008463
         8.660254e-01  0.000000   0.000001  14.860240   9.413288  0.000001
         9.659258e-01  0.000000   0.000000   0.073767   0.046729  0.000000

        # Shift the DDXS with float:
        >>> recoil = kb * T / M
        >>> ddxs.shift(recoil).data.iloc[::, ::200].round(6)
        Eout           1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -9.659258e-01      0.0  11.883406  23.907323  15.193081  3.262036
        -8.660254e-01      0.0  11.432343  24.471588  15.557409  3.153009
        -7.071068e-01      0.0  10.603110  25.426976  16.175325  2.949395
        -5.000000e-01      0.0   9.292316  26.788016  17.058324  2.620427
        -2.588190e-01      0.0   7.405719  28.549590  18.207481  2.133519
         6.123234e-17      0.0   4.977239  30.626550  19.576983  1.483492
         2.588190e-01      0.0   2.401004  32.675758  20.964784  0.757741
         5.000000e-01      0.0   0.562310  33.559785  21.676746  0.197247
         7.071068e-01      0.0   0.020205  29.834396  19.546943  0.008904
         8.660254e-01      0.0   0.000001  14.342692   9.783856  0.000001
         9.659258e-01      0.0   0.000000   0.063862   0.054086  0.000000

        # Shift the DDXS in the Eout axis:
        >>> recoil = Eout * kb * T / M
        >>> ddxs.shift(recoil).data.iloc[::, ::200].round(6)
        Eout           1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -9.659258e-01      0.0  11.822076  23.895518  15.264930  3.296181
        -8.660254e-01      0.0  11.369083  24.456245  15.633118  3.187391
        -7.071068e-01      0.0  10.537212  25.404835  16.258116  2.983919
        -5.000000e-01      0.0   9.224310  26.754111  17.152606  2.654508
        -2.588190e-01      0.0   7.338641  28.495673  18.319880  2.165656
         6.123234e-17      0.0   4.918270  30.538049  19.718229  1.510763
         2.588190e-01      0.0   2.361239  32.525888  21.153202  0.775960
         5.000000e-01      0.0   0.548102  33.300063  21.943376  0.204085
         7.071068e-01      0.0   0.019319  29.400359  19.928865  0.009421
         8.660254e-01      0.0   0.000001  13.858678  10.180689  0.000001
         9.659258e-01      0.0   0.000000   0.055546   0.062846  0.000000


        # Shift the DDXS in the theta axis:
        >>> recoil =  theta * kb * T / M
        >>> ddxs.shift(recoil, axis="mu").data.iloc[::, ::200].round(6)
        Eout            1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -9.659258e-01  0.000000   0.000000   0.000000   0.000000  0.000000
        -8.660254e-01  1.512354  11.550701  24.428439  15.447373  3.133787
        -7.071068e-01  1.130596  10.760152  25.354004  16.035088  2.938998
        -5.000000e-01  0.701001   9.503678  26.682255  16.878503  2.623838
        -2.588190e-01  0.329305   7.690280  28.408648  17.974768  2.158535
         6.123234e-17  0.101044   5.345472  30.457178  19.275756  1.539072
         2.588190e-01  0.016827   2.820610  32.526751  20.590726  0.845002
         5.000000e-01  0.001321   0.907921  33.653627  21.309075  0.288268
         7.071068e-01  0.000019   0.149474  31.103053  19.698006  0.050505
         8.660254e-01  0.000000   0.007025  19.957749  12.641257  0.002797
         9.659258e-01  0.000000   0.000001   8.458675   5.358187  0.000000

        # Shift the DDXS with a function that depends on theta and Eout:
        >>> recoil = np.outer(theta, Eout) * kb * T / M
        >>> ddxs.shift(recoil).data.iloc[::, ::200].round(6)
        Eout           1.80000    1.88008    1.96016    2.04024     2.12032
        mu
        -9.659258e-01      0.0  10.044976  23.349784  17.204335    4.296524
        -8.660254e-01      0.0   7.778998  22.649984  19.722305    5.510014
        -7.071068e-01      0.0   5.368781  21.298115  22.664033    6.984566
        -5.000000e-01      0.0   3.058885  18.845614  26.081612    8.843408
        -2.588190e-01      0.0   1.253561  14.854945  29.862541   11.284970
         6.123234e-17      0.0   0.282746   9.355962  33.424442   14.649544
         2.588190e-01      0.0   0.019782   3.711640  34.959481   19.575105
         5.000000e-01      0.0   0.000105   0.511907  30.150619   27.416328
         7.071068e-01      0.0   0.000000   0.004029  14.192045   41.605645
         8.660254e-01      0.0   0.000000   0.000000   0.567834   73.587937
         9.659258e-01      0.0   0.000000   0.000000   0.000000  181.402037
        """
        # Copy original data to avoid changing the original data:
        ddxs = self.data.copy()

        # Check the dx:
        dx_ = check_dx(self.data, dx, axis)
        axis_ = 1 if axis == "Eout" else 0 if axis == "mu" else axis
        if isinstance(dx_, float) or isinstance(dx_, int):
            ddxs = ddxs.apply(lambda x: reshift(x, dx_), axis=axis_)
        elif isinstance(dx_, pd.Series):
            data = ddxs.loc[::, dx_.index] if axis_ == 1 else ddxs.loc[dx_.index, ::]
            data_reshift = data.apply(lambda x: reshift(x, dx_.values), axis=axis_)
            if axis_ == 1:
                ddxs.loc[::, dx_.index] = data_reshift
            else:
                ddxs.loc[dx_.index, ::] = data_reshift
        else:
            data = ddxs.loc[dx_.index, dx_.columns]
            ddxs.loc[dx_.index, dx_.columns] = data.apply(lambda x: reshift(x, dx_.loc[x.name].values), axis=1)
        return self.__class__(self.Ein, self.T, self.M, ddxs)