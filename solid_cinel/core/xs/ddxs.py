"""
Python for working with Double Diferential XS.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from scipy.constants import physical_constants as const
from solid_cinel.core.material.scattering_function.scatfunc import ScatFunc, sigma1, get_scat_sct_angular, get_ScatFunc_pdos_angle
from solid_cinel.core.generic import integrate, reshift
import os
from math import pi
import dask.array as da

from typing import Iterable

# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]

# Avoid numba fast math:
nb.config.FASTMATH_DEFAULT = False

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


class Dxs:
    """
    Class for the differential cross section for elastic scattering
    """
    def __init__(self, Ein: float, T: float, M: float, algorithm: str, *args, **kwargs):
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
        algorithm : 'str'
            The algorithm to use for the double differencial doppler broadened
            elastic cross section. The available algorithms are:
                - "sigma1": sigma1 algorithm from NJOY2016 manual
                - "sab": S(alpha, -beta) tables for ddxs
                - "dopush": From the chosen S(alpha, -beta) model, the
                            distribution more similar to sigma1 is chosen and a
                            recoil energy
                - "courcelle": Fourier double-Laplace transform of a 4-point
        kwargs : dict
            Extra parameters for the selected algorithm
        """
        # Atributes of the scattering function (Change in these parameters will
        # change the scattering function):
        self.Ein = Ein
        self.T = T
        self.M = M
        self.algorithm = algorithm
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
    def from_sigma1(cls, xs_0K: pd.Series, Ein: float, M: float, T: float, Eout: np.ndarray):
        """
        Generate the Differential xs for elastic scattering from sigma1
        ..math::
            \frac{d\sigma_T(E)}{dE^\prime} = \frac{1}{2}\sqrt{\frac{M}{m\pi k_BT}}\frac{\sqrt{E^\prime}}{E}\sigma_0(E^\prime)\left(exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} - \sqrt{E^\prime}\right)^2 \right) - exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} + \sqrt{E^\prime}\right)^2 \right)\right)

        Parameters
        ----------
        xs_0K : pd.Series, (Z,)
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
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("ddxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212

        # SIGMA1 algorithm:
        >>> Dxs.from_sigma1(xs_0K, Ein, M, T, Eout).data.iloc[::100]
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
        scatfunction = ScatFunc.from_MD(Ein, M, T, Eout)
        return cls(Ein, T, M, "sigma1", scatfunction.convolve(xs_0K))

    @classmethod
    def from_dopush(cls, xs_0K: pd.Series, Ein: float, M: float, T: float, Eout: np.ndarray, theta: np.ndarray, *args,
                    **kwargs):
        """
        Generate the Differential xs for elastic scattering from the most similar distribution of the S(alpha, -beta)
        tables and sigma1 algorithm
        ..math::
            \frac{d\sigma_T(E)}{dE^\prime} = \frac{\sigma(E^\prime + R)}{2 * k_B * T}\sqrt{\frac{E^\prime}{E}} S(\alpha(\theta, E^\prime, E, M, T), \beta( E^\prime, E, T))

        Parameters for fgm, sct and pdos models
        ----------------------------------------
        xs_0K : pd.Series, (Z,)
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
            Minimun value to take into account in the creation of tau_n
            functions. For T>200 is convenient to set into 1.0e-14 to speed up
            the calculations. The default is 0.0.
        nphonon : 'int', optional
            Phonon expansion order. The default is 1000.

        Returns
        -------
        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("ddxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> theta = np.arange(0, 180, 1)[1::]

        # DOPUSH algorithm:
        >>> Dxs.from_dopush(xs_0K, Ein, M, T, Eout, theta, model="fgm").data.iloc[::100]
        Eout
        1.80000     0.000163
        1.84004     0.025374
        1.88008     1.152552
        1.92012    15.775754
        1.96016    67.355332
        2.00020    92.796544
        2.04024    42.650046
        2.08028     6.756453
        2.12032     0.380893
        2.16036     0.007884
        Name: 0.5000000000000001, dtype: float64
        """
        scatfunction = ScatFunc.from_Sab(Ein, M, T, Eout, theta, *args, **kwargs).to_sd()
        Exs = Eout + (Ein - scatfunction.data.idxmax())
        return cls(Ein, T, M, "dopush", scatfunction.convolve(xs_0K, Exs=Exs))

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
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("ddxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212

        # SIGMA1 algorithm:
        >>> round(Dxs.from_sigma1(xs_0K, Ein, M, T, Eout).integral, 2)
        9.09

        # DOPUSH algorithm:
        >>> theta = np.arange(0, 180, 1)[1::]
        >>> round(Dxs.from_dopush(xs_0K, Ein, M, T, Eout, theta, model="fgm").integral, 2)
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
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("ddxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> dxs = Dxs.from_sigma1(xs_0K, Ein, M, T, Eout)
        >>> round(dxs.prob["upscattering"], 6)
        0.505184
        >>> round(dxs.prob["downscattering"], 6)
        0.490636
        >>> round(dxs.prob["Ein=Eout"], 6)
        0.004179
        """
        integral = self.integral
        Eout = self.data.index.values
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
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("ddxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> dxs = Dxs.from_sigma1(xs_0K, Ein, M, T, Eout)

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
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("ddxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> dxs = Dxs.from_sigma1(xs_0K, Ein, M, T, Eout)
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
        dxs = self.data
        # Check the dx:
        dx_ = check_dx(self.data, dx, 0)
        if isinstance(dx, float) or isinstance(dx, int):
            dxs = reshift(dxs, dx_)
        else:
            dxs.loc[dx_.index] = reshift(dxs.loc[dx_.index], dx_)
        return Dxs(self.Ein, self.T, self.M, self.algorithm, dxs)


class DDxs:
    """
    Class for the Double differential cross section for elastic scattering
    """

    def __init__(self, Ein: float, T: float, M: float, algorithm: str, *args, **kwargs):
        """
        Class for the Double differential cross section for elastic scattering

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
        # Atributes of the scattering function:
        self.Ein = Ein
        self.T = T
        self.M = M
        self.algorithm = algorithm
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
    def from_Sab(cls, xs_0K: pd.Series, Ein: float, M: float, T: float, Eout: np.ndarray, theta: np.ndarray, *args,
                 **kwargs):
        """
        Generate the Double Differential XS for elastic scattering from S(alpha, -beta) tables
        ..math::
            \frac{d^2\sigma_T(E)}{dE^\prime d^\theta} = \frac{\sigma_b}{2 * k_B * T}\sqrt{\frac{E^\prime}{E}} S(\alpha(\theta, E^\prime, E, M, T), \beta( E^\prime, E, T))

        Common Parameters for fgm, sct and pdos models
        ----------------------------------------------
        xs_0K : pd.Series, (Z,)
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

        Parameters for sct
        ------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.

        Parameters for pdos
        -------------------
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
        DDxs
            Double differential cross section for elastic scattering

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("ddxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> theta = np.arange(0, 180, 1)[1::]
        >>> from solid_cinel.core.material.vibration.pdos import Pdos
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)

        # S(alpha, -beta) algorithm for FGM:
        >>> DDxs.from_Sab(xs_0K, Ein, M, T, Eout, theta, model="fgm").data.iloc[::18, ::200].round(6)
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

        # S(alpha, -beta) algorithm for SCT:
        >>> DDxs.from_Sab(xs_0K, Ein, M, T, Eout, theta, pdos, model="sct").data.iloc[::18, ::200].round(6)
        Eout        1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -0.999848  1.858801  12.101285  23.691478  15.003768  3.282861
        -0.945519  1.709037  11.873971  23.991586  15.194636  3.227633
        -0.798636  1.323836  11.183295  24.843563  15.736492  3.058043
        -0.573576  0.808007   9.883451  26.274727  16.646707  2.732824
        -0.292372  0.334761   7.791334  28.300174  17.934917  2.195682
         0.017452  0.067641   4.893611  30.821521  19.538757  1.425301
         0.325568  0.002956   1.850774  33.252955  21.086547  0.573498
         0.601815  0.000002   0.181650  33.106572  20.999520  0.064459
         0.819152  0.000000   0.000135  21.753380  13.801288  0.000071
         0.956305  0.000000   0.000000   0.389225   0.246967  0.000000

        # S(alpha, -beta) algorithm for PDOS:
        >>> theta = np.array([40, 80, 120, 160])
        >>> DDxs.from_Sab(xs_0K, Ein, M, T, Eout, theta, pdos, threshold=1.0e-14, model="pdos").data.iloc[::, ::200].round(6)
        Eout        1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -0.939693  2.283494  12.162243  23.939616  15.274148  3.160733
        -0.500000  1.042632   9.808431  26.702481  17.022308  2.543099
         0.173648  0.072043   3.820128  31.943965  20.299691  0.982917
         0.766044  0.000029   0.051432  24.534948  15.423312  0.012977
        """
        scatfunction = ScatFunc.from_Sab(Ein, M, T, Eout, theta, *args, **kwargs)
        return cls(Ein, T, M, "S(alpha, -beta)", scatfunction.convolve(xs_0K))

    @classmethod
    def from_4PCF(cls, xs_0K: pd.Series, Ein: float, M: float, T: float, Eout: np.ndarray, theta: np.ndarray, *args,
                    **kwargs):
        """
        Generate the Double Differential XS for elastic scattering from Fourier double-Laplace transform of a 4-point
        correlation function modified
        ..math::
            \frac{d^2\sigma_T(E)}{dE^\prime d^\theta} = \frac{1}{2 * k_B * T}\sqrt{\frac{E^\prime}{E}} S(\alpha(\theta, E^\prime, E, M, T), \beta( E^\prime, E, T)) \sigma^{T(1+\mu)/2}((E^\prime+E + \frac{\alpha k_{B} T}{1-\mu})/2 - E \mu / A)

        For the xs matrix calculation, they are the following models available:
            - "sigma1": sigma1 algorithm from NJOY2016 manual (default)
            - "fgm": Free Gas Model
            - "sct": Short Collision Time
            - "pdos": Phonon Density of States

        Common parameters
        -----------------
        xs_0K : pd.Series, (Z,)
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

        Parameters for sct
        ------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object

        Parameters for pdos
        -------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object
        threshold : 'float', optional
            Minimun value to take into account in the creation of tau_n functions. For T>200 is convenient to set into
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
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> theta = np.arange(0, 180, 10)[1::]
        >>> from solid_cinel.core.material.vibration.pdos import Pdos
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)

        # Coercelle with sigma1 algorithm:
        >>> DDxs.from_4PCF(xs_0K, Ein, M, T, Eout, theta).data.iloc[::, ::200].round(6)
        Eout            1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -9.848078e-01  1.799454  12.011826  23.795201  15.058832  3.254168
        -9.396926e-01  1.676368  11.820165  24.045974  15.218228  3.207630
        -8.660254e-01  1.481046  11.484979  24.467361  15.486062  3.125737
        -7.660444e-01  1.229144  10.983197  25.064005  15.865282  3.002054
        -6.427876e-01  0.943760  10.284397  25.841170  16.359232  2.827887
        -5.000000e-01  0.654925   9.354774  26.802832  16.970446  2.593109
        -3.420201e-01  0.396320   8.165850  27.947664  17.698083  2.288171
        -1.736482e-01  0.197800   6.711538  29.260922  18.532824  1.908309
         6.123234e-17  0.074460   5.037219  30.697832  19.446272  1.461210
         1.736482e-01  0.018204   3.278979  32.148848  20.368969  0.978386
         3.420201e-01  0.002218   1.689219  33.366368  21.143932  0.525271
         5.000000e-01  0.000081   0.578041  33.810815  21.428972  0.191553
         6.427876e-01  0.000000   0.090864  32.347398  20.504444  0.033458
         7.660444e-01  0.000000   0.002704  26.829842  17.009134  0.001208
         8.660254e-01  0.000000   0.000001  14.852992   9.417271  0.000001
         9.396926e-01  0.000000   0.000000   1.824940   1.157149  0.000000
         9.848078e-01  0.000000   0.000000   0.000005   0.000003  0.000000

        # Coercelle with fgm model:
        >>> DDxs.from_4PCF(xs_0K, Ein, M, T, Eout, theta, model="fgm").data.iloc[::, ::200].round(6)
        Eout            1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -9.848078e-01  1.799454  12.011827  23.795202  15.058833  3.254168
        -9.396926e-01  1.676368  11.820167  24.045979  15.218231  3.207630
        -8.660254e-01  1.481046  11.484979  24.467361  15.486062  3.125737
        -7.660444e-01  1.229143  10.983196  25.064002  15.865280  3.002054
        -6.427876e-01  0.943760  10.284396  25.841173  16.359234  2.827887
        -5.000000e-01  0.654925   9.354772  26.802836  16.970448  2.593109
        -3.420201e-01  0.396320   8.165849  27.947661  17.698086  2.288171
        -1.736482e-01  0.197800   6.711537  29.260920  18.532823  1.908310
         6.123234e-17  0.074460   5.037219  30.697833  19.446271  1.461210
         1.736482e-01  0.018204   3.278979  32.148851  20.368970  0.978386
         3.420201e-01  0.002218   1.689220  33.366374  21.143934  0.525271
         5.000000e-01  0.000081   0.578041  33.810822  21.428975  0.191553
         6.427876e-01  0.000000   0.090864  32.347404  20.504447  0.033458
         7.660444e-01  0.000000   0.002704  26.829847  17.009136  0.001208
         8.660254e-01  0.000000   0.000001  14.852993   9.417271  0.000001
         9.396926e-01  0.000000   0.000000   1.824941   1.157150  0.000000
         9.848078e-01  0.000000   0.000000   0.000005   0.000003  0.000000

        # Coercelle with sct model:
        >>> DDxs.from_4PCF(xs_0K, Ein, M, T, Eout, theta, pdos, model="sct").data.iloc[::, ::200].round(6)
        Eout            1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -9.848078e-01  1.812376  12.019176  23.754160  15.057222  3.271238
        -9.396926e-01  1.688887  11.828533  24.004617  15.216660  3.224747
        -8.660254e-01  1.492848  11.495030  24.425472  15.484572  3.142910
        -7.660444e-01  1.239858  10.995558  25.021399  15.863937  3.019258
        -6.427876e-01  0.952978  10.299610  25.797763  16.358166  2.845039
        -5.000000e-01  0.662263   9.373166  26.758645  16.969858  2.610032
        -3.420201e-01  0.401536   8.187339  27.902924  17.698312  2.304547
        -1.736482e-01  0.200934   6.735340  29.216223  18.534407  1.923624
         6.123234e-17  0.075917   5.061517  30.654357  19.450089  1.474709
         1.736482e-01  0.018657   3.300792  32.108796  20.376432  0.989124
         3.420201e-01  0.002291   1.705042  33.333670  21.157296  0.532392
         5.000000e-01  0.000085   0.585907  33.792208  21.451727  0.194923
         6.427876e-01  0.000000   0.092747  32.353400  20.541320  0.034272
         7.660444e-01  0.000000   0.002796  26.871913  17.063274  0.001253
         8.660254e-01  0.000000   0.000001  14.921198   9.475763  0.000001
         9.396926e-01  0.000000   0.000000   1.849244   1.174449  0.000000
         9.848078e-01  0.000000   0.000000   0.000006   0.000004  0.000000

        # Coercelle with pdos model: (Example not very accurate, only for
        # demonstration purposes)
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 5)
        >>> DDxs.from_4PCF(xs_0K, Ein, M, T, Eout, theta, pdos, threshold=1.0e-14, nphonon=10, model="pdos").data.round(6)
        Eout                1.8       1.9         2.0       2.1       2.2
        mu
        -9.848078e-01  0.000000  0.000000    0.000006  0.000000  0.000000
        -9.396926e-01  0.000000  0.000000    0.000012  0.000000  0.000000
        -8.660254e-01  0.000000  0.000001    0.000034  0.000000  0.000000
        -7.660444e-01  0.000000  0.000004    0.000133  0.000001  0.000000
        -6.427876e-01  0.000000  0.000023    0.000691  0.000004  0.000000
        -5.000000e-01  0.000000  0.000155    0.004365  0.000031  0.000000
        -3.420201e-01  0.000000  0.001133    0.030240  0.000244  0.000000
        -1.736482e-01  0.000001  0.008076    0.205815  0.001893  0.000000
         6.123234e-17  0.000004  0.049158    1.222034  0.012581  0.000000
         1.736482e-01  0.000021  0.221520    5.616005  0.061820  0.000002
         3.420201e-01  0.000066  0.634651   18.003875  0.192170  0.000007
         5.000000e-01  0.000104  0.988169   38.417022  0.321589  0.000011
         6.427876e-01  0.000067  0.722460   59.594677  0.248796  0.000008
         7.660444e-01  0.000015  0.229033   82.997584  0.081369  0.000002
         8.660254e-01  0.000001  0.037892  103.919838  0.013351  0.000000
         9.396926e-01  0.000000  0.005059   82.211000  0.001741  0.000000
         9.848078e-01  0.000000  0.000221   25.049987  0.000073  0.000000

        """
        scatfunction = ScatFunc.from_Sab(Ein, M, T, Eout, theta, *args, **kwargs)
        xs = xs_matrix(xs_0K, Ein, M, T, Eout, theta, scatfunction.get_angle, *args, **kwargs) if kwargs.get("model") else xs_matrix(xs_0K, Ein, M, T, Eout, theta)
        return cls(Ein, T, M, "coercelle", scatfunction.convolve(xs))

    @property
    def angular(self) -> Dxs:
        """
        The angular distribution of the Double Differential XS for elastic scattering

        Returns
        -------
        Dxs
            The angular distribution of the Double Differential XS for elastic scattering

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("ddxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> theta = np.arange(0, 180, 1)[1::]

        # Angular distribution:
        >>> DDxs.from_Sab(xs_0K, Ein, M, T, Eout, theta, model="fgm").angular.data.iloc[::200].round(6)
        Eout
        1.80000     0.768794
        1.88008    10.451361
        1.96016    54.522950
        2.04024    34.506930
        2.12032     2.920481
        dtype: float64
        """
        return Dxs(self.Ein, self.T, self.M, self.algorithm, self.data.apply(integrate, axis=0))
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
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> theta = np.arange(0, 180, 1)[1::]
        >>> from solid_cinel.core.material.vibration.pdos import Pdos
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)

        # S(alpha, -beta) algorithm for FGM:
        >>> round(DDxs.from_Sab(xs_0K, Ein, M, T, Eout, theta, model="fgm").integral, 2)
        9.07
        """
        return self.angular.integral

    @property
    def E_prob(self) -> dict:
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
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> theta = np.arange(0, 180, 15)[1::]
        >>> ddxs = DDxs.from_Sab(xs_0K, Ein, M, T, Eout, theta)
        >>> probabilities = ddxs.E_prob
        >>> round(probabilities["upscattering"], 6)
        0.389484
        >>> round(probabilities["downscattering"], 6)
        0.60678
        >>> round(probabilities["Ein=Eout"], 6)
        0.003736
        """
        return self.angular.prob

    @property
    def Angle_prob(self) -> pd.Series:
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
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> theta = np.arange(0, 180, 15)[1::]
        >>> ddxs = DDxs.from_Sab(xs_0K, Ein, M, T, Eout, theta)
        >>> angular_prob = ddxs.Angle_prob
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
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> theta = np.arange(0, 180, 15)[1::]
        >>> ddxs = DDxs.from_Sab(xs_0K, Ein, M, T, Eout, theta)
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
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("ddxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> theta = np.arange(0, 180, 15)[1::]
        >>> ddxs = DDxs.from_Sab(xs_0K, Ein, M, T, Eout, theta)
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
        return self.__class__(self.Ein, self.T, self.M, self.algorithm, ddxs)

def xs_matrix(*args, **kwargs) -> np.ndarray:
    """
    Calculate the cross section matrix for a given incident energy, target mass,
    target temperature, outgoing energy grid and outgoing angle grid using arno
    model with the most similar S(alpha, -beta) distribution with sigma1
    .. math::
        \sigma^{T(1+\mu)/2}\left( \frac{E + E^\prime}{2} - E\frac{\mu m}{M}\right)

    Parameters
    ----------
    mu_fit : float
        The cosine of the outgoing angle to fit the S(alpha, -beta) distribution
        with sigma1

    Parameters for fgm, sct and pdos models
    ---------------------------------------
    xs_0K : pd.Series
        Cross section at 0K in barns
    Ein : float
        The incident energy of the neutron in eV
    M : float
        Mass of the material in amu
    T : float
        Temperature of the material in K
    Eout : np.array, (N,)
        The neutron outgoing energy grid in eV
    theta : np.array, (M,)
        The neutron outgoing angle grid in degrees (0, 180]

    Extra parameters for sct
    ------------------------
    Teff : float
        Effective temperature of the material in K

    Extra parameters for pdos
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
    np.ndarray, (M, N)
        Cross section matrix in barns

    Examples
    --------
    # 0K xs data for U238:
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("ddxs.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
    >>> os.chdir(wd)

    >>> T = 1000
    >>> Ein = 2.0
    >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 7)
    >>> M = 238.05077040419212
    >>> theta = np.arange(10, 190, 10)
    >>> mu_fit = np.cos(np.deg2rad(60))

    # sigma1 model:
    >>> xs_values = xs_matrix(xs_0K, Ein, M, T, Eout, theta)
    >>> pd.DataFrame(xs_values, index=theta[::-1], columns=Eout).round(6)
        1.800000	1.866667	1.933333	2.000000	2.066667	2.133333	2.200000
    180	9.102355	9.095532	9.088710	9.081758	9.074679	9.067600	9.060521
    170	9.102381	9.095558	9.088736	9.081785	9.074706	9.067627	9.060548
    160	9.102454	9.095632	9.088810	9.081861	9.074782	9.067703	9.060625
    150	9.102577	9.095755	9.088932	9.081987	9.074910	9.067831	9.060753
    140	9.102746	9.095924	9.089098	9.082158	9.075085	9.068007	9.060928
    130	9.102952	9.096130	9.089299	9.082363	9.075297	9.068219	9.061139
    120	9.103190	9.096369	9.089534	9.082602	9.075545	9.068466	9.061386
    110	9.103451	9.096632	9.089797	9.082865	9.075817	9.068740	9.061657
    100	9.103729	9.096912	9.090074	9.083149	9.076110	9.069031	9.061947
    90	9.104017	9.097203	9.090360	9.083438	9.076408	9.069334	9.062245
    80	9.104301	9.097490	9.090649	9.083730	9.076705	9.069633	9.062545
    70	9.104579	9.097769	9.090927	9.084011	9.076995	9.069924	9.062834
    60	9.104837	9.098033	9.091189	9.084274	9.077265	9.070196	9.063105
    50	9.105070	9.098270	9.091426	9.084513	9.077508	9.070442	9.063350
    40	9.105269	9.098471	9.091631	9.084720	9.077716	9.070655	9.063557
    30	9.105425	9.098635	9.091795	9.084887	9.077888	9.070823	9.063725
    20	9.105525	9.098748	9.091915	9.085010	9.078011	9.070941	9.063833
    10	9.105489	9.098775	9.091979	9.085087	9.078074	9.070973	9.063803

    # fgm model:
    >>> xs_values = xs_matrix(xs_0K, Ein, M, T, Eout, theta, mu_fit, model="fgm")
    >>> pd.DataFrame(xs_values, index=theta[::-1], columns=Eout).round(6)
        1.800000	1.866667	1.933333	2.000000	2.066667	2.133333	2.200000
    180	9.102355	9.095532	9.088710	9.081758	9.074679	9.067600	9.060521
    170	9.102381	9.095559	9.088737	9.081785	9.074706	9.067627	9.060549
    160	9.102456	9.095634	9.088811	9.081863	9.074784	9.067705	9.060627
    150	9.102577	9.095755	9.088932	9.081987	9.074910	9.067831	9.060752
    140	9.102745	9.095923	9.089097	9.082157	9.075084	9.068005	9.060926
    130	9.102951	9.096129	9.089300	9.082364	9.075298	9.068219	9.061140
    120	9.103189	9.096367	9.089535	9.082603	9.075546	9.068467	9.061386
    110	9.103451	9.096631	9.089796	9.082867	9.075819	9.068741	9.061658
    100	9.103729	9.096911	9.090073	9.083148	9.076109	9.069033	9.061948
    90	9.104018	9.097203	9.090360	9.083438	9.076407	9.069333	9.062246
    80	9.104303	9.097492	9.090650	9.083728	9.076705	9.069633	9.062545
    70	9.104579	9.097771	9.090929	9.084012	9.076996	9.069924	9.062834
    60	9.104837	9.098033	9.091191	9.084276	9.077266	9.070196	9.063105
    50	9.105070	9.098269	9.091428	9.084515	9.077509	9.070443	9.063350
    40	9.105269	9.098473	9.091632	9.084722	9.077717	9.070654	9.063560
    30	9.105427	9.098637	9.091796	9.084887	9.077888	9.070825	9.063727
    20	9.105526	9.098748	9.091917	9.085012	9.078012	9.070942	9.063833
    10	9.105490	9.098776	9.091979	9.085087	9.078076	9.070974	9.063804

    # sct model:
    >>> from solid_cinel.core.material.vibration.pdos import Pdos
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> xs_values = xs_matrix(xs_0K, Ein, M, T, Eout, theta, mu_fit, pdos, model="sct")
    >>> pd.DataFrame(xs_values, index=theta[::-1], columns=Eout).round(6)
         1.800000  1.866667  1.933333  2.000000  2.066667  2.133333  2.200000
    180  9.102355  9.095532  9.088710  9.081758  9.074679  9.067600  9.060521
    170  9.102371  9.095549  9.088724  9.081772  9.074696  9.067617  9.060539
    160  9.102449  9.095627  9.088802  9.081853  9.074777  9.067698  9.060619
    150  9.102574  9.095752  9.088926  9.081981  9.074906  9.067828  9.060749
    140  9.102744  9.095919  9.089091  9.082151  9.075080  9.068001  9.060922
    130  9.102948  9.096126  9.089296  9.082360  9.075295  9.068217  9.061137
    120  9.103187  9.096365  9.089532  9.082600  9.075544  9.068465  9.061384
    110  9.103449  9.096629  9.089793  9.082865  9.075817  9.068739  9.061657
    100  9.103730  9.096910  9.090072  9.083146  9.076107  9.069031  9.061946
    90   9.104016  9.097202  9.090362  9.083436  9.076406  9.069332  9.062245
    80   9.104302  9.097491  9.090649  9.083729  9.076704  9.069632  9.062544
    70   9.104578  9.097770  9.090928  9.084011  9.076995  9.069923  9.062833
    60   9.104837  9.098032  9.091190  9.084275  9.077265  9.070196  9.063104
    50   9.105069  9.098269  9.091427  9.084514  9.077509  9.070442  9.063349
    40   9.105269  9.098472  9.091632  9.084721  9.077716  9.070654  9.063559
    30   9.105426  9.098636  9.091798  9.084886  9.077887  9.070824  9.063726
    20   9.105528  9.098748  9.091917  9.085011  9.078012  9.070941  9.063832
    10   9.105490  9.098775  9.091979  9.085086  9.078076  9.070973  9.063803

    # pdos model:
    >>> nphonon = 100
    >>> threshold = 1.0e-14
    >>> xs_values = xs_matrix(xs_0K, Ein, M, T, Eout, theta, mu_fit, pdos, nphonon=nphonon, threshold=threshold, model="pdos")
    >>> pd.DataFrame(xs_values, index=theta[::-1], columns=Eout).round(6)
         1.800000  1.866667  1.933333  2.000000  2.066667  2.133333  2.200000
    180  9.102355  9.095532  9.088710  9.081758  9.074679  9.067600  9.060521
    170  9.103715  9.096910  9.090104  9.083212  9.076165  9.069103  9.062042
    160  9.103626  9.096821  9.090015  9.083121  9.076074  9.069013  9.061952
    150  9.103167  9.096364  9.089558  9.082651  9.075602  9.068542  9.061485
    140  9.102780  9.095979  9.089172  9.082254  9.075208  9.068149  9.061094
    130  9.102666  9.095866  9.089056  9.082135  9.075097  9.068042  9.060984
    120  9.102740  9.095944  9.089135  9.082217  9.075183  9.068131  9.061076
    110  9.102926  9.096135  9.089323  9.082409  9.075386  9.068336  9.061282
    100  9.103175  9.096384  9.089570  9.082664  9.075646  9.068599  9.061543
    90   9.103449  9.096661  9.089848  9.082944  9.075937  9.068892  9.061832
    80   9.103734  9.096949  9.090132  9.083232  9.076233  9.069190  9.062131
    70   9.104011  9.097229  9.090414  9.083518  9.076522  9.069484  9.062421
    60   9.104271  9.097492  9.090677  9.083783  9.076796  9.069758  9.062696
    50   9.104647  9.097872  9.091058  9.084168  9.077187  9.070151  9.063088
    40   9.104847  9.098078  9.091263  9.084378  9.077398  9.070365  9.063300
    30   9.105006  9.098241  9.091431  9.084545  9.077570  9.070538  9.063469
    20   9.105109  9.098356  9.091553  9.084671  9.077696  9.070658  9.063578
    10   9.105072  9.098383  9.091617  9.084748  9.077762  9.070689  9.063551

    Dirac delta test for Teff calculation and pdos model:
    >>> Ein = 36.68
    >>> M = 238.05077040419212
    >>> T = 300
    >>> theta = np.arange(1, 11, 1)
    >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 7)[:6]
    >>> xs_values = xs_matrix(xs_0K, Ein, M, T, Eout, theta, mu_fit, pdos, model="sct")
    >>> pd.DataFrame(xs_values, index=theta[::-1], columns=Eout).round(6)
        33.012000	34.234667	35.457333	36.680000	37.902667	39.125333
    10	0.781011	0.212474	18.943993	7832.112249	116.900933	48.371957
    9	0.774634	0.214952	18.935478	7827.100451	116.942587	48.337172
    8	0.765202	0.218967	18.947906	7822.491592	116.933518	48.278974
    7	0.750937	0.225468	18.991055	7818.332543	116.851287	48.184526
    6	0.728496	0.236354	19.083733	7814.650622	116.653779	48.029743
    5	0.690994	0.255950	19.265986	7811.484114	116.253915	47.765243
    4	0.622453	0.296031	19.636420	7808.858742	115.447043	47.276543
    3	0.479921	0.399744	20.498496	7806.792841	113.658175	46.249378
    2	0.151264	0.848392	23.250511	7805.305048	108.749223	43.580322
    1	39.411876	15.646885	48.317676	7804.407359	88.153494	33.909257
    """
    # Common arguments:
    if len(args) == 6:
        xs_0K, Ein, M, T, Eout, theta = args
    elif len(args) == 7:
        xs_0K, Ein, M, T, Eout, theta, mu_fit = args
    else:
        xs_0K, Ein, M, T, Eout, theta, mu_fit, pdos = args

    mu = np.sort(np.cos(np.deg2rad(theta)))
    T_arno = T * (1 + mu) / 2
    model = kwargs.pop("model", "sigma1")

    # Specific arguments for S(alpha, -beta) DB:
    if model == "sigma1":
        return xs_matrix_values(xs_0K.values, xs_0K.index.values, Ein, M, T_arno,
                                Eout, mu)
    if model == "fgm":
        return xs_matrix_values(xs_0K.values, xs_0K.index.values, Ein, M, T_arno,
                                Eout, mu, mu_fit, T_arno, 1.0)
    elif model == "sct":
        Teff = np.array(
            [pdos.Teff(T_aprox) if T_aprox > 0.0 else 0 for T_aprox in T_arno]
        )
        Teff[np.isnan(Teff)] = T_arno[np.isnan(Teff)]
        return xs_matrix_values(xs_0K.values, xs_0K.index.values, Ein, M,
                                T_arno, Eout, mu, mu_fit, Teff, 1.0)
    elif model == "pdos":
        tau1 = np.zeros((len(T_arno), len(pdos.rho.values)))
        DebyeWallerCoeff = np.zeros(len(T_arno))
        delta_beta = np.zeros(len(T_arno))
        for i in range(len(T_arno)):
            if T_arno[i] > 0.0:
                tau1[i, :] = pdos.get_tau_1(T_arno[i]).values
                DebyeWallerCoeff[i] = pdos.DebyeWallerCoeff(T_arno[i])
                delta_beta[i] = pdos.to_beta_grid(T_arno[i]).grid
        return xs_matrix_values_pdos(xs_0K.values, xs_0K.index.values, Ein, M,
                                     T_arno, Eout, mu, mu_fit,
                                     kwargs.pop("nphonon", 1000), tau1, delta_beta,
                                     kwargs.pop("threshold", 0.0), DebyeWallerCoeff,
                                     chunksize= kwargs.pop("chunksize", (100, 10)))
    else:
        raise ValueError("Model not found")



@nb.jit("float64[:, :](float64, float64[:], float64[:], float64)",
    nopython=True, nogil=True, cache=True)
def get_Ein_arno(Ein: float, Eout: np.ndarray, mu: np.ndarray,
                 M: float) -> np.ndarray:
    """
    Get the incident energy matrix for the arno model.

    Parameters
    ----------
    Ein: float
        The incident energy of the neutron in eV
    Eout: np.ndarray, (Z,)
        The neutron outgoing energy grid in eV
    mu: np.ndarray, (M,)
        The neutron outgoing angle grid in degrees (0, 180]
    M: float
        Mass of the material in amu

    Returns
    -------
    Ein_arno: np.ndarray, (M, Z)
        Incident energy matrix for the arno model
    """
    Ein_arno = np.empty((len(mu), len(Eout)))
    for i in range(len(mu)):
        alpha = (Ein + Eout - 2 * mu[i] * np.sqrt(Ein * Eout)) * m / M
        Ein_arno[i, :] = (Eout + Ein) / 2 - Ein * mu[i] * m / M
        Ein_arno[i, :] += 0.5 * alpha / (1 - mu[i])
    return Ein_arno


@nb.jit(nopython=True, nogil=True, cache=True)
def default_Eout(Ein: float) -> np.ndarray:
    """
    Generate the default Eout grid for the convolution. The grid is tested with
    NJOY values to ensure a relative difference smaller than 0.4%

    Parameters
    ----------
    Ein : float
        Incident energy in eV

    Returns
    -------
    Eout : ndarray
        Outgoing energy grid in eV

    Examples
    --------
    Test the default Eout with NJOY values:
    # 0K xs data for U238:
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("ddxs.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
    >>> os.chdir(wd)

    # Generate Broadening test results:
    >>> T = 1000
    >>> Ein = 2.0
    >>> Eout = default_Eout(Ein)
    >>> M = 238.05077040419212
    >>> round(Dxs.from_sigma1(xs_0K, Ein, M, T, Eout).integral, 2)
    9.09
    """
    Eout_small = np.linspace(0,
                             0.99 * Ein,
                             2000)
    Eout_middle = np.linspace(0.99 * Ein,
                              Ein * 1.01,
                              3000)
    if Ein * 2 < 5.0:
        Eout_great = np.logspace(np.log10(Ein * 1.01),
                                 np.log10(5.0),
                                 2000)
    else:
        Eout_great = np.logspace(np.log10(Ein * 1.01),
                                 np.log10(2 * Ein),
                                 2000)
    return np.sort(np.concatenate((Eout_great, Eout_small, Eout_middle)))


@nb.jit(nopython=True, nogil=True, cache=True)
def Db(xs_values, xs_E, Ein, Eout, pdf):
    """
    Calculate the doppler broadening of a cross section for a pdf

    Parameters
    ----------
    xs_values: np.ndarray, (N,)
        Cross section values in barns
    xs_E: np.ndarray, (N,)
        Cross section energy grid in eV
    Ein: float
        The incident energy of the neutron in eV
    Eout: np.ndarray, (Z,)
        The neutron outgoing energy grid in eV
    pdf: np.ndarray, (Z,)
        Probability density function

    Returns
    -------
    Db_xs: float
        Doppler broadened cross section in barns
    """
    max_pos = np.argmax(pdf)
    if pdf[max_pos] > 1.0e308:  # Overflow found in pdf_val
        Db_xs = np.interp(Eout[max_pos], xs_E, xs_values)
    else:
        norm = np.trapz(pdf, x=Eout)
        # Recoil:
        recoil = Ein - Eout[max_pos]
        # xs:
        xs_Eout_arno = np.interp(Eout + recoil, xs_E, xs_values)
        Db_xs = np.trapz(xs_Eout_arno * pdf, x=Eout) / norm
    return Db_xs


@nb.jit(nopython=True, nogil=True, cache=True, parallel=True)
def xs_matrix_values(xs_values: np.ndarray, xs_E: np.ndarray, Ein: float,
                     M: float, T_arno: np.ndarray, Eout: np.ndarray,
                     mu: np.ndarray, *args) -> np.ndarray:
    """
    Calculate the cross section matrix for a given incident energy, target mass,
    target temperature, outgoing energy grid and outgoing angle grid using arno
    model with different pdf.

    Parameters
    ----------
    xs_values: np.ndarray, (N,)
        Cross section values in barns
    xs_E: np.ndarray, (N,)
        Cross section energy grid in eV
    Ein: float
        The incident energy of the neutron in eV
    M: float
        Mass of the material in amu
    T_arno: np.ndarray, (M,)
        Target temperature grid in K
    Eout: np.ndarray, (Z,)
        The neutron outgoing energy grid in eV
    mu: np.ndarray, (M,)
        The neutron outgoing angle grid in degrees (0, 180]

    Parameters for FGM, SCT and Pdos models:
    ----------------------------------------
    mu_fit: float
        The cosine of the outgoing angle to fit the S(alpha, -beta) distribution
        with sigma1

    Parameters for SCT models:
    --------------------------
    Teff: np.ndarray, (M,)
        Effective temperature of the material in K for all the T_arno values
    w_s: float
        Weight of the S(alpha, -beta) distribution with sigma1. For solid is 1.0

    Parameters for PDOS models:
    ---------------------------
    nphonon: int
        Phonon expansion order
    tau1: np.ndarray, (M, T)
        tau1 values for all the T_arno values
    delta_beta: np.ndarray, (M,)
        delta_beta values for all the T_arno values
    threshold: float
        Minimun value to take into account in the creation of tau_n
    DebyeWallerCoeff: np.ndarray, (M,)
        DebyeWallerCoeff values for all the T_arno values

    Returns
    -------
    np.ndarray, (M, N)
        Cross section matrix in barns

    Examples
    --------
    Test default, linear and logarithmic grids with NJOY values:
    # 0K xs data for U238:
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("ddxs.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
    >>> os.chdir(wd)

    Common parameters for all the examples:
    >>> T = 1000
    >>> Ein = 2.0
    >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 7)
    >>> M = 238.05077040419212
    >>> theta = np.arange(10, 190, 10)
    >>> mu = np.sort(np.cos(np.deg2rad(theta)))
    >>> mu_fit = np.cos(np.deg2rad(60))
    >>> T_arno = T * (1 + mu) / 2

    # sigma1 model:
    >>> xs_values = xs_matrix_values(xs_0K.values, xs_0K.index.values, Ein, M, T_arno, Eout, mu)
    >>> pd.DataFrame(xs_values, index=theta[::-1], columns=Eout).round(6)
        1.800000	1.866667	1.933333	2.000000	2.066667	2.133333	2.200000
    180	9.102355	9.095532	9.088710	9.081758	9.074679	9.067600	9.060521
    170	9.102381	9.095558	9.088736	9.081785	9.074706	9.067627	9.060548
    160	9.102454	9.095632	9.088810	9.081861	9.074782	9.067703	9.060625
    150	9.102577	9.095755	9.088932	9.081987	9.074910	9.067831	9.060753
    140	9.102746	9.095924	9.089098	9.082158	9.075085	9.068007	9.060928
    130	9.102952	9.096130	9.089299	9.082363	9.075297	9.068219	9.061139
    120	9.103190	9.096369	9.089534	9.082602	9.075545	9.068466	9.061386
    110	9.103451	9.096632	9.089797	9.082865	9.075817	9.068740	9.061657
    100	9.103729	9.096912	9.090074	9.083149	9.076110	9.069031	9.061947
    90	9.104017	9.097203	9.090360	9.083438	9.076408	9.069334	9.062245
    80	9.104301	9.097490	9.090649	9.083730	9.076705	9.069633	9.062545
    70	9.104579	9.097769	9.090927	9.084011	9.076995	9.069924	9.062834
    60	9.104837	9.098033	9.091189	9.084274	9.077265	9.070196	9.063105
    50	9.105070	9.098270	9.091426	9.084513	9.077508	9.070442	9.063350
    40	9.105269	9.098471	9.091631	9.084720	9.077716	9.070655	9.063557
    30	9.105425	9.098635	9.091795	9.084887	9.077888	9.070823	9.063725
    20	9.105525	9.098748	9.091915	9.085010	9.078011	9.070941	9.063833
    10	9.105489	9.098775	9.091979	9.085087	9.078074	9.070973	9.063803

    # fgm model:
    >>> xs_values = xs_matrix_values(xs_0K.values, xs_0K.index.values, Ein, M, T_arno, Eout, mu, mu_fit, T_arno, 1.0)
    >>> pd.DataFrame(xs_values, index=theta[::-1], columns=Eout).round(6)
        1.800000	1.866667	1.933333	2.000000	2.066667	2.133333	2.200000
    180	9.102355	9.095532	9.088710	9.081758	9.074679	9.067600	9.060521
    170	9.102381	9.095559	9.088737	9.081785	9.074706	9.067627	9.060549
    160	9.102456	9.095634	9.088811	9.081863	9.074784	9.067705	9.060627
    150	9.102577	9.095755	9.088932	9.081987	9.074910	9.067831	9.060752
    140	9.102745	9.095923	9.089097	9.082157	9.075084	9.068005	9.060926
    130	9.102951	9.096129	9.089300	9.082364	9.075298	9.068219	9.061140
    120	9.103189	9.096367	9.089535	9.082603	9.075546	9.068467	9.061386
    110	9.103451	9.096631	9.089796	9.082867	9.075819	9.068741	9.061658
    100	9.103729	9.096911	9.090073	9.083148	9.076109	9.069033	9.061948
    90	9.104018	9.097203	9.090360	9.083438	9.076407	9.069333	9.062246
    80	9.104303	9.097492	9.090650	9.083728	9.076705	9.069633	9.062545
    70	9.104579	9.097771	9.090929	9.084012	9.076996	9.069924	9.062834
    60	9.104837	9.098033	9.091191	9.084276	9.077266	9.070196	9.063105
    50	9.105070	9.098269	9.091428	9.084515	9.077509	9.070443	9.063350
    40	9.105269	9.098473	9.091632	9.084722	9.077717	9.070654	9.063560
    30	9.105427	9.098637	9.091796	9.084887	9.077888	9.070825	9.063727
    20	9.105526	9.098748	9.091917	9.085012	9.078012	9.070942	9.063833
    10	9.105490	9.098776	9.091979	9.085087	9.078076	9.070974	9.063804

    # sct model:
    >>> from solid_cinel.core.material.vibration.pdos import Pdos
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> Teff = np.nan_to_num([pdos.Teff(T_aprox) if T_aprox > 0.0 else 0 for T_aprox in T_arno])
    >>> xs_values = xs_matrix_values(xs_0K.values, xs_0K.index.values, Ein, M, T_arno, Eout, mu, mu_fit, Teff, 1.0)
    >>> pd.DataFrame(xs_values, index=theta[::-1], columns=Eout).round(6)
         1.800000  1.866667  1.933333  2.000000  2.066667  2.133333  2.200000
    180  9.102355  9.095532  9.088710  9.081758  9.074679  9.067600  9.060521
    170  9.102371  9.095549  9.088724  9.081772  9.074696  9.067617  9.060539
    160  9.102449  9.095627  9.088802  9.081853  9.074777  9.067698  9.060619
    150  9.102574  9.095752  9.088926  9.081981  9.074906  9.067828  9.060749
    140  9.102744  9.095919  9.089091  9.082151  9.075080  9.068001  9.060922
    130  9.102948  9.096126  9.089296  9.082360  9.075295  9.068217  9.061137
    120  9.103187  9.096365  9.089532  9.082600  9.075544  9.068465  9.061384
    110  9.103449  9.096629  9.089793  9.082865  9.075817  9.068739  9.061657
    100  9.103730  9.096910  9.090072  9.083146  9.076107  9.069031  9.061946
    90   9.104016  9.097202  9.090362  9.083436  9.076406  9.069332  9.062245
    80   9.104302  9.097491  9.090649  9.083729  9.076704  9.069632  9.062544
    70   9.104578  9.097770  9.090928  9.084011  9.076995  9.069923  9.062833
    60   9.104837  9.098032  9.091190  9.084275  9.077265  9.070196  9.063104
    50   9.105069  9.098269  9.091427  9.084514  9.077509  9.070442  9.063349
    40   9.105269  9.098472  9.091632  9.084721  9.077716  9.070654  9.063559
    30   9.105426  9.098636  9.091798  9.084886  9.077887  9.070824  9.063726
    20   9.105528  9.098748  9.091917  9.085011  9.078012  9.070941  9.063832
    10   9.105490  9.098775  9.091979  9.085086  9.078076  9.070973  9.063803
    """
    xs_mat = np.empty((len(mu), len(Eout)))
    Ein_arno = get_Ein_arno(Ein, Eout, mu, M)
    if mu[0] == np.cos(pi):  # mu is sorted array
        xs_mat[0, :] = np.interp(Ein_arno[0, :], xs_E, xs_values)
        start = 1
    else:
        start = 0
    for i in range(start, len(mu), 1):
        for j in prange(len(Eout)):
            Eout_db = default_Eout(Ein_arno[i, j])
            if len(args) == 0:  # sigma1
                pdf = sigma1(Eout_db, Ein_arno[i, j], T_arno[i], M)
            elif len(args) == 3:  # FGM or SCT
                pdf = get_scat_sct_angular(Eout_db, args[0], Ein_arno[i, j], T_arno[i], M, args[1][i], args[2])
            xs_mat[i, j] = Db(xs_values, xs_E, Ein_arno[i, j], Eout_db, pdf)
    return xs_mat


def xs_matrix_values_pdos(xs_values: np.ndarray, xs_E: np.ndarray, Ein: float,
                     M: float, T_arno: np.ndarray, Eout: np.ndarray,
                     mu: np.ndarray, *args,
                     chunksize: tuple = (100, 10)) -> np.ndarray:
    """
    Calculate the cross section matrix for a given incident energy, target mass,
    target temperature, outgoing energy grid and outgoing angle grid using arno
    model with different pdf.

    Parameters
    ----------
    xs_values: np.ndarray, (N,)
        Cross section values in barns
    xs_E: np.ndarray, (N,)
        Cross section energy grid in eV
    Ein: float
        The incident energy of the neutron in eV
    M: float
        Mass of the material in amu
    T_arno: np.ndarray, (M,)
        Target temperature grid in K
    Eout: np.ndarray, (Z,)
        The neutron outgoing energy grid in eV
    mu: np.ndarray, (M,)
        The neutron outgoing angle grid in degrees (0, 180]
    mu_fit: float
        The cosine of the outgoing angle to fit the S(alpha, -beta) distribution
        with sigma1
    nphonon: int
        Phonon expansion order
    tau1: np.ndarray, (M, T)
        tau1 values for all the T_arno values
    delta_beta: np.ndarray, (M,)
        delta_beta values for all the T_arno values
    threshold: float
        Minimun value to take into account in the creation of tau_n
    DebyeWallerCoeff: np.ndarray, (M,)
        DebyeWallerCoeff values for all the T_arno values

    Returns
    -------
    np.ndarray, (M, N)
        Cross section matrix in barns

    Examples
    --------
    # 0K xs data for U238:
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("ddxs.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
    >>> os.chdir(wd)

    Common parameters for all the examples:
    >>> T = 1000
    >>> Ein = 2.0
    >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 7)
    >>> M = 238.05077040419212
    >>> theta = np.arange(10, 190, 10)
    >>> mu = np.sort(np.cos(np.deg2rad(theta)))
    >>> mu_fit = np.cos(np.deg2rad(60))
    >>> T_arno = T * (1 + mu) / 2
    >>> from solid_cinel.core.material.vibration.pdos import Pdos
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> DebyeWallerCoeff = [pdos.DebyeWallerCoeff(T) if T > 0.0 else 0.0 for T in T_arno]
    >>> tau1 = [pdos.get_tau_1(T).values if T > 0.0 else np.array([0.0] * len(mu)) for T in T_arno]
    >>> delta_beta = [interv_in_energy_U238 / (kb * T) if T > 0.0 else 0.0 for T in T_arno]
    >>> nphonon = 100
    >>> threshold = 1.0e-14
    >>> xs_values = xs_matrix_values_pdos(xs_0K.values, xs_0K.index.values, Ein, M, T_arno, Eout, mu, mu_fit, nphonon, tau1, delta_beta, threshold, DebyeWallerCoeff,  chunksize=(10, 5))
    >>> pd.DataFrame(xs_values, index=theta[::-1], columns=Eout).round(6)
         1.800000  1.866667  1.933333  2.000000  2.066667  2.133333  2.200000
    180  9.102355  9.095532  9.088710  9.081758  9.074679  9.067600  9.060521
    170  9.103715  9.096910  9.090104  9.083212  9.076165  9.069103  9.062042
    160  9.103626  9.096821  9.090015  9.083121  9.076074  9.069013  9.061952
    150  9.103167  9.096364  9.089558  9.082651  9.075602  9.068542  9.061485
    140  9.102780  9.095979  9.089172  9.082254  9.075208  9.068149  9.061094
    130  9.102666  9.095866  9.089056  9.082135  9.075097  9.068042  9.060984
    120  9.102740  9.095944  9.089135  9.082217  9.075183  9.068131  9.061076
    110  9.102926  9.096135  9.089323  9.082409  9.075386  9.068336  9.061282
    100  9.103175  9.096384  9.089570  9.082664  9.075646  9.068599  9.061543
    90   9.103449  9.096661  9.089848  9.082944  9.075937  9.068892  9.061832
    80   9.103734  9.096949  9.090132  9.083232  9.076233  9.069190  9.062131
    70   9.104011  9.097229  9.090414  9.083518  9.076522  9.069484  9.062421
    60   9.104271  9.097492  9.090677  9.083783  9.076796  9.069758  9.062696
    50   9.104647  9.097872  9.091058  9.084168  9.077187  9.070151  9.063088
    40   9.104847  9.098078  9.091263  9.084378  9.077398  9.070365  9.063300
    30   9.105006  9.098241  9.091431  9.084545  9.077570  9.070538  9.063469
    20   9.105109  9.098356  9.091553  9.084671  9.077696  9.070658  9.063578
    10   9.105072  9.098383  9.091617  9.084748  9.077762  9.070689  9.063551
    """
    @nb.njit
    def compute_chunk(Ein_arno_chunk, row, *args):
        result = np.empty(Ein_arno_chunk.shape)
        for i in range(Ein_arno_chunk.shape[0]):
            i_ = i + row
            for j in range(Ein_arno_chunk.shape[1]):
                Eout_db = default_Eout(Ein_arno_chunk[i, j])
                pdf = get_ScatFunc_pdos_angle(Ein_arno_chunk[i, j], M, T_arno[i_],
                                            Eout_db, args[0], args[1],
                                            args[2][i_], args[3][i_],
                                            args[4], args[5][i_])
                result[i, j] = Db(xs_values, xs_E, Ein_arno_chunk[i, j], Eout_db, pdf)
        return result

    def chunk_wrapper(block, start, *args, block_info=None):
        # The row of the block in the array
        row = block_info[0]['array-location'][0][0]
        if start > 0:
            row += start
        return compute_chunk(block, row, *args)

    Ein_arno = get_Ein_arno(Ein, Eout, mu, M)
    start = 0
    if mu[0] == np.cos(pi):
        xs_mat180 = np.array([np.interp(Ein_arno[0, :], xs_E, xs_values)])
        start = 1

    Ein_arno_da = da.from_array(Ein_arno[start::, ::])
    xs_mat = Ein_arno_da.map_blocks(chunk_wrapper, start, *args,
                                    dtype=float, chunks=chunksize)

    # Compute the Dask array
    xs_mat = xs_mat.compute(scheduler="threads")

    if mu[0] == np.cos(pi):
        # Concatenate xs_mat180 and xs_mat
        xs_mat = np.concatenate([xs_mat180, xs_mat], axis=0)

    return xs_mat


def generate_Eout(Ein, Elim: Iterable = None, N: int = None,
                  space: str = "linear"):
    """
    Generate Eout grid for the convolution.

    Parameters
    ----------
    Ein : float
        Incident energy in eV
    Elim : Iterable, (2,)
        Outgoing energy limits in eV. The first value is the lower limit and the
        second value is the upper limit.
    N : int, optional
        Number of points in the outgoing energy grid. If None, the default
        number of points is used.
    space : str, optional
        Type of grid. Available options are "linear" and "log". Default is
        "linear".

    Returns
    -------
    Eout : ndarray
        Outgoing energy grid in eV

    Raises
    ------
    ValueError
        If the number of points is not introduced.
    ValueError
        If the space is not available.

    Examples
    --------
    Test default, linear and logarithmic grids with NJOY values:
    # 0K xs data for U238:
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("ddxs.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
    >>> os.chdir(wd)

    # Common data:
    >>> T = 1000
    >>> Ein = 2.0
    >>> Eout = default_Eout(Ein)
    >>> M = 238.05077040419212

    # Test default grid:
    >>> Eout = default_Eout(Ein)
    >>> round(Dxs.from_sigma1(xs_0K, Ein, M, T, Eout).integral, 2)
    9.09

    # Test linear grid:
    >>> Eout = generate_Eout(Ein, Elim=[Ein * 0.9, Ein * 1.1], N=5000)
    >>> round(Dxs.from_sigma1(xs_0K, Ein, M, T, Eout).integral, 2)
    9.09

    # Test logarithmic grid:
    >>> Eout = generate_Eout(Ein, Elim=[Ein * 0.9, Ein * 1.1], N=5000, space="log")
    >>> round(Dxs.from_sigma1(xs_0K, Ein, M, T, Eout).integral, 2)
    9.09
    """
    if Elim is None:
        Eout = default_Eout(Ein)
    else:
        if N is None:
            raise ValueError("The number of points is not defined")
        if space == "linear":
            Eout = np.linspace(Elim[0],
                               Elim[1],
                               num=N, endpoint=True)
        elif space == "log":
            Eout = np.logspace(np.log10(Elim[0]),
                               np.log10(Elim[1]),
                               num=N, endpoint=True)
        else:
            raise ValueError("The space {} is not available".format(space))
    return Eout


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
