"""
Python file for working xs doppler broadening functions.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from scipy.constants import physical_constants as const
from solid_cinel.core.material.scattering_function.scatfunc import ScatFunc, sigma1, get_scat_sct_angular, get_ScatFunc_pdos_angle
from solid_cinel.core.generic import integrate
import os

from typing import Iterable

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
        1.88008     0.525884
        1.96016    52.660553
        2.04024    56.917662
        2.12032     0.864760
        dtype: float64
        """
        # Check the dx:
        dx_ = check_dx(self.data, dx, axis=0)
        if isinstance(dx, float) or isinstance(dx, int):
            self.data = reshift(self.data, dx_, self.data.index)
        else:
            self.data.loc[dx_.index] = reshift(self.data.loc[dx_.index], dx_,
                                               dx_.index)
        return Dxs(self.Ein, self.T, self.M, self.algorithm, self.data)


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
        -0.939693  2.203391  11.934588  24.417997  15.575835  3.101303
        -0.500000  0.994808   9.521449  27.156911  17.307645  2.468526
         0.173648  0.066807   3.586114  32.202480  20.456875  0.922720
         0.766044  0.000026   0.045654  23.748453  14.926872  0.011525
        """
        scatfunction = ScatFunc.from_Sab(Ein, M, T, Eout, theta, *args, **kwargs)
        return cls(Ein, T, M, "S(alpha, -beta)", scatfunction.convolve(xs_0K))

    @classmethod
    def from_coercelle(cls, xs_0K: pd.Series, Ein: float, M: float, T: float, Eout: np.ndarray, theta: np.ndarray, *args,
                    **kwargs):
        """
        Generate the Double Differential XS for elastic scattering from Fourier double-Laplace transform of a 4-point
        correlation function
        ..math::
            \frac{d^2\sigma_T(E)}{dE^\prime d^\theta} = \frac{1}{2 * k_B * T}\sqrt{\frac{E^\prime}{E}} S(\alpha(\theta, E^\prime, E, M, T), \beta( E^\prime, E, T)) \sigma^{T(1+\mu)/2}((E^\prime+E)/2 - E \mu / A)

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
        >>> DDxs.from_coercelle(xs_0K, Ein, M, T, Eout, theta).data.iloc[::, ::200].round(6)
        Eout            1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -9.848078e-01  1.799780  12.014045  23.799698  15.061843  3.254832
        -9.396926e-01  1.676763  11.822952  24.051670  15.221935  3.208413
        -8.660254e-01  1.481580  11.489103  24.476148  15.491693  3.126870
        -7.660444e-01  1.229592  10.987217  25.073257  15.871246  3.003188
        -6.427876e-01  0.944073  10.287839  25.849947  16.364919  2.828878
        -5.000000e-01  0.655123   9.357644  26.811206  16.975884  2.593950
        -3.420201e-01  0.396433   8.168209  27.955887  17.703434  2.288872
        -1.736482e-01  0.197855   6.713410  29.269238  18.538228  1.908875
         6.123234e-17  0.074480   5.038601  30.706415  19.451845  1.461636
         1.736482e-01  0.018209   3.279876  32.157794  20.374777  0.978669
         3.420201e-01  0.002219   1.689684  33.375691  21.149970  0.525423
         5.000000e-01  0.000081   0.578202  33.820341  21.435136  0.191609
         6.427876e-01  0.000000   0.090890  32.356604  20.510394  0.033467
         7.660444e-01  0.000000   0.002705  26.837555  17.014119  0.001208
         8.660254e-01  0.000000   0.000001  14.857302   9.420052  0.000001
         9.396926e-01  0.000000   0.000000   1.825474   1.157494  0.000000
         9.848078e-01  0.000000   0.000000   0.000005   0.000003  0.000000

        # Coercelle with fgm model:
        >>> DDxs.from_coercelle(xs_0K, Ein, M, T, Eout, theta, model="fgm").data.iloc[::, ::200].round(6)
        Eout            1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -9.848078e-01  1.799779  12.014039  23.799686  15.061836  3.254830
        -9.396926e-01  1.676670  11.822344  24.050527  15.221266  3.208283
        -8.660254e-01  1.481313  11.487095  24.472001  15.489150  3.126373
        -7.660444e-01  1.229365  10.985218  25.068762  15.868442  3.002665
        -6.427876e-01  0.943930  10.286290  25.846084  16.362492  2.828462
        -5.000000e-01  0.655042   9.356495  26.807932  16.973823  2.593636
        -3.420201e-01  0.396391   8.167353  27.952975  17.701601  2.288637
        -1.736482e-01  0.197836   6.712774  29.266485  18.536498  1.908698
         6.123234e-17  0.074473   5.038147  30.703670  19.450123  1.461508
         1.736482e-01  0.018207   3.279584  32.154964  20.372999  0.978585
         3.420201e-01  0.002219   1.689531  33.372717  21.148112  0.525378
         5.000000e-01  0.000081   0.578148  33.817249  21.433206  0.191592
         6.427876e-01  0.000000   0.090881  32.353553  20.508492  0.033464
         7.660444e-01  0.000000   0.002704  26.834946  17.012495  0.001208
         8.660254e-01  0.000000   0.000001  14.855816   9.419127  0.000001
         9.396926e-01  0.000000   0.000000   1.825288   1.157378  0.000000
         9.848078e-01  0.000000   0.000000   0.000005   0.000003  0.000000

        # Coercelle with sct model:
        >>> DDxs.from_coercelle(xs_0K, Ein, M, T, Eout, theta, pdos, model="sct").data.iloc[::, ::200].round(6)
        Eout            1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -9.848078e-01  1.812702  12.021390  23.758675  15.060225  3.271903
        -9.396926e-01  1.689192  11.830712  24.009178  15.219694  3.225403
        -8.660254e-01  1.493117  11.497147  24.430112  15.487659  3.143549
        -7.660444e-01  1.240081  10.997583  25.026155  15.867098  3.019872
        -6.427876e-01  0.953149  10.301507  25.802668  16.361423  2.845618
        -5.000000e-01  0.662383   9.374892  26.763734  16.973233  2.610563
        -3.420201e-01  0.401608   8.188847  27.908231  17.701827  2.305016
        -1.736482e-01  0.200970   6.736581  29.221780  18.538082  1.924015
         6.123234e-17  0.075931   5.062450  30.660187  19.453941  1.475009
         1.736482e-01  0.018661   3.301400  32.114901  20.380469  0.989325
         3.420201e-01  0.002292   1.705357  33.340007  21.161476  0.532500
         5.000000e-01  0.000085   0.586015  33.798632  21.455962  0.194963
         6.427876e-01  0.000000   0.092764  32.359550  20.545372  0.034279
         7.660444e-01  0.000000   0.002796  26.877020  17.066638  0.001253
         8.660254e-01  0.000000   0.000001  14.924038   9.477631  0.000001
         9.396926e-01  0.000000   0.000000   1.849596   1.174681  0.000000
         9.848078e-01  0.000000   0.000000   0.000006   0.000004  0.000000

        # Coercelle with pdos model:
        Is not done because the test take too long
        """
        scatfunction = ScatFunc.from_Sab(Ein, M, T, Eout, theta, *args, **kwargs)
        xs = xs_matrix(scatfunction.get_angle, xs_0K, Ein, M, T, Eout, theta, *args, **kwargs) if kwargs.get("model") else xs_matrix(xs_0K, Ein, M, T, Eout, theta)
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
        -9.659258e-01      0.0  11.752439  23.882671  15.334022  3.326921
        -8.660254e-01      0.0  11.297289  24.439673  15.705924  3.218356
        -7.071068e-01      0.0  10.462484  25.381107  16.337734  3.015037
        -5.000000e-01      0.0   9.147290  26.718016  17.243286  2.685263
        -2.588190e-01      0.0   7.262820  28.438565  18.428012  2.194713
         6.123234e-17      0.0   4.851817  30.444705  19.854188  1.535498
         2.588190e-01      0.0   2.316653  32.368444  21.334787  0.792574
         5.000000e-01      0.0   0.532320  33.028519  22.201022  0.210386
         7.071068e-01      0.0   0.018355  28.950225  20.300333  0.009908
         8.660254e-01      0.0   0.000001  13.367055  10.574417  0.000001
         9.659258e-01      0.0   0.000000   0.047965   0.072548  0.000000

        # Shift the DDXS in the theta axis:
        >>> recoil =  theta * kb * T / M
        >>> ddxs.shift(recoil, axis="theta").data.iloc[::, ::200].round(6)
        Eout           1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -9.659258e-01      0.0   0.000000   0.000000   0.000000  0.000000
        -8.660254e-01      0.0  11.344216  24.382244  15.667580  3.229550
        -7.071068e-01      0.0  10.545228  25.287795  16.275111  3.035189
        -5.000000e-01      0.0   9.281694  26.581393  17.150745  2.718964
        -2.588190e-01      0.0   7.470283  28.249148  18.297584  2.248719
         6.123234e-17      0.0   5.149073  30.197366  19.678353  1.616773
         2.588190e-01      0.0   2.681311  32.091733  21.121817  0.899436
         5.000000e-01      0.0   0.846623  32.912249  22.048438  0.312936
         7.071068e-01      0.0   0.136533  29.887963  20.737366  0.056005
         8.660254e-01      0.0   0.006065  18.515600  13.787775  0.003274
         9.659258e-01      0.0   0.000000   7.600770   6.027802  0.000001


        # Shift the DDXS with a function that depends on theta and Eout:
        >>> recoil = np.outer(theta, Eout) * kb * T / M
        >>> ddxs.shift(recoil).data.iloc[::, ::200].round(6)
        Eout           1.80000   1.88008    1.96016    2.04024     2.12032
        mu
        -9.659258e-01      0.0  0.000000   0.000000   0.000000    0.000000
        -8.660254e-01      0.0  7.660641  22.480793  19.856386    5.659898
        -7.071068e-01      0.0  5.309295  21.080102  22.725649    7.167386
        -5.000000e-01      0.0  3.061451  18.618380  26.027269    9.065437
        -2.588190e-01      0.0  1.299103  14.697815  29.623749   11.554546
         6.123234e-17      0.0  0.328627   9.382356  32.912699   14.974422
         2.588190e-01      0.0  0.036612   3.963107  34.150405   19.949152
         5.000000e-01      0.0  0.001426   0.763712  29.473931   27.755249
         7.071068e-01      0.0  0.000000   0.049612  15.238504   41.390633
         8.660254e-01      0.0  0.000000   0.000237   2.564919   69.179036
         9.659258e-01      0.0  0.000000   0.000000   0.045441  131.357046
        """
        # Check the dx:
        dx_ = check_dx(self.data, dx, axis)
        axis_ = 1 if axis == "Eout" else 0 if axis == "theta" else axis
        if isinstance(dx_, float) or isinstance(dx_, int):
            x_values = self.data.columns.values if axis_ == 1 else self.data.index.values
            self.data = self.data.apply(reshift, axis=axis_, args=(dx_, x_values))
        elif isinstance(dx_, pd.Series):
            data = self.data.loc[::, dx_.index] if axis_ == 1 else self.data.loc[dx_.index, ::]
            data_reshift = data.apply(reshift, axis=axis_, args=(dx_.values, dx_.index))
            if axis_ == 1:
                self.data.loc[::, dx_.index] = data_reshift
            else:
                self.data.loc[dx_.index, ::] = data_reshift
        else:
            data = self.data.loc[dx_.index, dx_.columns]
            self.data.loc[dx_.index, dx_.columns] = data.apply(lambda x: reshift(x, dx_.loc[x.name].values, dx_.columns), axis=1)
        return DDxs(self.Ein, self.T, self.M, self.algorithm, self.data)


@nb.jit(nopython=True, nogil=False, cache=False, parallel=True)
def xs_matrix_sigma1(xs_values: np.ndarray, xs_E: np.ndarray, Ein: float,
                     M: float, T_arno: np.ndarray, Eout: np.ndarray,
                     mu: np.ndarray) -> np.ndarray:
    """
    Calculate the cross section matrix for a given incident energy, target mass,
    target temperature, outgoing energy grid and outgoing angle grid using arno
    model with sigma1 algorithm.
    .. math::
        \sigma^{T(1+\mu)/2}\left( \frac{E + E^\prime}{2} - E\frac{\mu m}{M}\right)

    Parameters
    ----------
    xs_values : ndarray, (N,)
        Cross section values at 0K in barns
    xs_E : ndarray, (N,)
        Energy grid of the cross section in eV
    Ein : float
        Incident energy in eV
    M : float
        Target mass in amu
    T_arno : ndarray, (M,)
        Target temperature according to T * (1 + mu) / 2 in K
    Eout : ndarray, (N,)
        Outgoing energy grid in eV
    theta : ndarray, (M,)
        Outgoing angle grid in degrees

    Returns
    -------
    xs_mat : ndarray, (M, N)
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
    >>> mu = np.sort(np.cos(theta * np.pi / 180))
    >>> T_arno = T * (1 + mu) / 2
    >>> xs_values = xs_matrix_sigma1(xs_0K.values, xs_0K.index.values, Ein, M, T_arno, Eout, mu)
    >>> pd.DataFrame(xs_values, index=theta[::-1], columns=Eout).round(6)
         1.800000  1.866667  1.933333  2.000000  2.066667  2.133333  2.200000
    180  9.103994  9.097201  9.090408  9.083550  9.076500  9.069451  9.062402
    170  9.104026  9.097232  9.090439  9.083582  9.076532  9.069483  9.062434
    160  9.104602  9.097777  9.090953  9.084068  9.076994  9.069920  9.062847
    150  9.105858  9.099023  9.092187  9.085287  9.078207  9.071119  9.064031
    140  9.106064  9.099251  9.092436  9.085552  9.078502  9.071432  9.064363
    130  9.105971  9.099170  9.092365  9.085491  9.078462  9.071404  9.064345
    120  9.105951  9.099155  9.092352  9.085480  9.078467  9.071414  9.064358
    110  9.106047  9.099254  9.092450  9.085579  9.078577  9.071526  9.064471
    100  9.106233  9.099444  9.092637  9.085765  9.078773  9.071725  9.064668
    90   9.106479  9.099693  9.092883  9.086010  9.079025  9.071980  9.064921
    80   9.106756  9.099973  9.093161  9.086288  9.079309  9.072267  9.065206
    70   9.107045  9.100265  9.093451  9.086577  9.079603  9.072564  9.065502
    60   9.107328  9.100550  9.093735  9.086860  9.079891  9.072854  9.065791
    50   9.107590  9.100815  9.093999  9.087124  9.080158  9.073124  9.066060
    40   9.107821  9.101047  9.094231  9.087355  9.080392  9.073361  9.066296
    30   9.108011  9.101238  9.094421  9.087546  9.080584  9.073555  9.066490
    20   9.108151  9.101380  9.094562  9.087687  9.080727  9.073699  9.066633
    10   9.108238  9.101467  9.094649  9.087774  9.080815  9.073787  9.066722
    """
    xs_mat = np.zeros((len(mu), len(Eout)))
    for i in prange(len(mu)):
        if mu[i] == np.cos(np.pi):
            Ein_arno = (Eout + Ein) / 2 + Ein * m / M
            xs_mat[i, :] = np.interp(Ein_arno, xs_E, xs_values)
        else:
            for j in prange(len(Eout)):
                Ein_arno = (Eout[j] + Ein) / 2 - Ein * mu[i] * m / M
                Eout_db = default_Eout(Ein_arno)
                pdf = sigma1(Eout_db, Ein_arno, T_arno[i], M)
                xs_Eout_arno = np.interp(Eout_db, xs_E, xs_values)
                xs_mat[i, j] = np.trapz(xs_Eout_arno * pdf, x=Eout_db)
    return xs_mat


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
    >>> mu_fit = np.cos(60 / 180 * np.pi)

    # sigma1 model:
    >>> xs_values = xs_matrix(xs_0K, Ein, M, T, Eout, theta, model="sigma1")
    >>> pd.DataFrame(xs_values, index=theta[::-1], columns=Eout).round(6)
         1.800000  1.866667  1.933333  2.000000  2.066667  2.133333  2.200000
    180  9.103994  9.097201  9.090408  9.083550  9.076500  9.069451  9.062402
    170  9.104026  9.097232  9.090439  9.083582  9.076532  9.069483  9.062434
    160  9.104602  9.097777  9.090953  9.084068  9.076994  9.069920  9.062847
    150  9.105858  9.099023  9.092187  9.085287  9.078207  9.071119  9.064031
    140  9.106064  9.099251  9.092436  9.085552  9.078502  9.071432  9.064363
    130  9.105971  9.099170  9.092365  9.085491  9.078462  9.071404  9.064345
    120  9.105951  9.099155  9.092352  9.085480  9.078467  9.071414  9.064358
    110  9.106047  9.099254  9.092450  9.085579  9.078577  9.071526  9.064471
    100  9.106233  9.099444  9.092637  9.085765  9.078773  9.071725  9.064668
    90   9.106479  9.099693  9.092883  9.086010  9.079025  9.071980  9.064921
    80   9.106756  9.099973  9.093161  9.086288  9.079309  9.072267  9.065206
    70   9.107045  9.100265  9.093451  9.086577  9.079603  9.072564  9.065502
    60   9.107328  9.100550  9.093735  9.086860  9.079891  9.072854  9.065791
    50   9.107590  9.100815  9.093999  9.087124  9.080158  9.073124  9.066060
    40   9.107821  9.101047  9.094231  9.087355  9.080392  9.073361  9.066296
    30   9.108011  9.101238  9.094421  9.087546  9.080584  9.073555  9.066490
    20   9.108151  9.101380  9.094562  9.087687  9.080727  9.073699  9.066633
    10   9.108238  9.101467  9.094649  9.087774  9.080815  9.073787  9.066722

    # fgm model:
    >>> xs_values = xs_matrix(0.0, xs_0K, Ein, M, T, Eout, theta, model="fgm")
    >>> pd.DataFrame(xs_values, index=theta[::-1], columns=Eout).round(6)
         1.800000  1.866667  1.933333  2.000000  2.066667  2.133333  2.200000
    180  9.103994  9.097201  9.090408  9.083550  9.076500  9.069451  9.062402
    170  9.104017  9.097224  9.090431  9.083573  9.076524  9.069475  9.062426
    160  9.104091  9.097298  9.090505  9.083643  9.076601  9.069552  9.062503
    150  9.104210  9.097417  9.090623  9.083754  9.076725  9.069676  9.062626
    140  9.104369  9.097578  9.090780  9.083906  9.076891  9.069842  9.062793
    130  9.104565  9.097777  9.090976  9.084098  9.077096  9.070050  9.062999
    120  9.104790  9.098007  9.091201  9.084322  9.077332  9.070290  9.063236
    110  9.105041  9.098260  9.091450  9.084570  9.077590  9.070553  9.063497
    100  9.105304  9.098531  9.091718  9.084836  9.077864  9.070833  9.063774
    90   9.105577  9.098806  9.091993  9.085113  9.078149  9.071120  9.064060
    80   9.105847  9.099083  9.092267  9.085388  9.078430  9.071409  9.064346
    70   9.106109  9.099349  9.092536  9.085658  9.078702  9.071686  9.064626
    60   9.106358  9.099598  9.092786  9.085910  9.078958  9.071946  9.064886
    50   9.106581  9.099824  9.093013  9.086138  9.079189  9.072181  9.065122
    40   9.106774  9.100019  9.093209  9.086333  9.079390  9.072384  9.065323
    30   9.106929  9.100178  9.093367  9.086494  9.079551  9.072547  9.065490
    20   9.107044  9.100294  9.093484  9.086612  9.079670  9.072668  9.065611
    10   9.107115  9.100364  9.093556  9.086683  9.079744  9.072743  9.065684


    # sct model:
    >>> from solid_cinel.core.material.vibration.pdos import Pdos
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> xs_values = xs_matrix(mu_fit, xs_0K, Ein, M, T, Eout, theta, pdos, model="sct")
    >>> pd.DataFrame(xs_values, index=theta[::-1], columns=Eout).round(6)
         1.800000  1.866667  1.933333  2.000000  2.066667  2.133333  2.200000
    180  9.103994  9.097201  9.090408  9.083550  9.076500  9.069451  9.062402
    170  9.104011  9.097218  9.090424  9.083555  9.076518  9.069469  9.062420
    160  9.104089  9.097296  9.090502  9.083635  9.076599  9.069549  9.062500
    150  9.104213  9.097421  9.090627  9.083761  9.076728  9.069679  9.062630
    140  9.104383  9.097588  9.090793  9.083928  9.076902  9.069853  9.062803
    130  9.104587  9.097795  9.091000  9.084133  9.077117  9.070068  9.063019
    120  9.104824  9.098035  9.091237  9.084370  9.077364  9.070317  9.063267
    110  9.105085  9.098299  9.091499  9.084633  9.077637  9.070591  9.063540
    100  9.105365  9.098580  9.091778  9.084912  9.077925  9.070883  9.063831
    90   9.105650  9.098871  9.092069  9.085200  9.078222  9.071184  9.064131
    80   9.105935  9.099160  9.092357  9.085492  9.078518  9.071484  9.064431
    70   9.106210  9.099440  9.092636  9.085772  9.078808  9.071775  9.064721
    60   9.106468  9.099702  9.092898  9.086035  9.079077  9.072050  9.064994
    50   9.106702  9.099939  9.093136  9.086273  9.079319  9.072294  9.065241
    40   9.106904  9.100144  9.093341  9.086479  9.079526  9.072507  9.065454
    30   9.107068  9.100310  9.093508  9.086644  9.079697  9.072680  9.065628
    20   9.107189  9.100431  9.093629  9.086768  9.079823  9.072806  9.065754
    10   9.107262  9.100507  9.093703  9.086843  9.079899  9.072885  9.065834
    """
    # Common arguments:
    if len(args) == 6:
        xs_0K, Ein, M, T, Eout, theta = args
    elif len(args) == 7:
        mu_fit, xs_0K, Ein, M, T, Eout, theta = args
    else:
        mu_fit, xs_0K, Ein, M, T, Eout, theta, pdos = args
    mu = np.sort(np.cos(theta * np.pi / 180))
    T_arno = T * (1 + mu) / 2
    model = kwargs.pop("model", "sigma1")

    # Division according to the model:
    if model == "sigma1":
        return xs_matrix_sigma1(xs_0K.values, xs_0K.index.values, Ein, M, T_arno,
                                Eout, mu)
    elif model == "fgm":
        return xs_matrix_sct(xs_0K.values, xs_0K.index.values, Ein, M, T_arno,
                             Eout, mu, T_arno, 1.0, mu_fit)
    elif model == "sct":
        Teff = pd.Series(
            [pdos.Teff(T_aprox) if T_aprox > 0.0 else 0 for T_aprox in T_arno],
            index=T_arno).backfill().values
        return xs_matrix_sct(xs_0K.values, xs_0K.index.values, Ein, M, T_arno,
                             Eout, mu, Teff, 1.0, mu_fit)
    else:
        threshold = kwargs.pop("threshold", 0.0)
        nphonon = kwargs.pop("nphonon", 1000)
        tau1 = pdos.get_tau_1(T)
        debye_waller_coeff = pdos.DebyeWallerCoeff(T)
        return xs_matrix_pdos(xs_0K.values, xs_0K.index.values, Ein, M, T, Eout,
                              mu, nphonon, tau1.values, tau1.index[1],
                              threshold, debye_waller_coeff, mu_fit)

@nb.jit(nopython=True, nogil=False, cache=True, parallel=True)
def xs_matrix_pdos(xs_values: np.ndarray, xs_E: np.ndarray, Ein: float, M: float,
                   T_arno: np.ndarray, Eout: np.ndarray, mu: np.ndarray, nphonon: int,
                   tau1: np.ndarray, delta_beta: float, threshold: float,
                   DebyeWallerCoeff: float, mu_fit: float) -> np.ndarray:
    """
    Calculate the cross section matrix for a given incident energy, target mass,
    target temperature, outgoing energy grid and outgoing angle grid using arno
    model with the most similar pdos distribution with sigma1

    Parameters
    ----------
    xs_values : np.ndarray
        Cross section values at 0K in barns
    xs_E : np.ndarray
        Energy grid of the cross section in eV
    Ein : float
        The incident energy of the neutron in eV
    M : float
        The mass of the target material in amu
    T_arno : float
        Temperature of the material according to T * (1 + mu) /2 in K
    Eout : np.ndarray, (N,)
        The neutron outgoing energy grid in eV
    mu : np.ndarray, (M,)
        The cosine of the neutron outgoing angle grid in degrees (0, 180]
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
    mu_fit : float
        The cosine of the outgoing angle to fit the S(alpha, -beta) distribution
        with sigma1

    Returns
    -------
    np.ndarray, (M, N)
        Cross section matrix in barns
    """
    xs_mat = np.zeros((len(mu), len(Eout)))
    for i in prange(len(mu)):
        if mu[i] == np.cos(np.pi):
            Ein_arno = (Eout + Ein) / 2 + Ein * m / M
            xs_mat[i, :] = np.interp(Ein_arno, xs_E, xs_values)
        else:
            for j in prange(len(Eout)):
                Ein_arno = (Eout[j] + Ein) / 2 - Ein * mu[i] * m / M
                Eout_db = default_Eout(Ein_arno)
                # Distribution + Normalization:
                pdf_val = get_ScatFunc_pdos_angle(Ein_arno, M, T_arno[i], Eout_db,
                                                 mu_fit, nphonon, tau1, delta_beta,
                                                 threshold, DebyeWallerCoeff)
                pdf_val /= np.trapz(pdf_val, x=Eout_db)
                # Recoil:
                recoil = Ein_arno - Eout_db[np.argmax(pdf_val)]
                # xs:
                xs_Eout_arno = np.interp(Eout_db, xs_E, xs_values)
                xs_mat[i, j] = np.trapz(xs_Eout_arno * pdf_val, x=Eout_db + recoil)
    return xs_mat


@nb.jit(nopython=True, nogil=False, cache=True, parallel=True)
def xs_matrix_sct(xs_values: np.ndarray, xs_E: np.ndarray, Ein: float, M: float,
                  T_arno: np.ndarray, Eout: np.ndarray, mu: np.ndarray,
                  Teff: np.ndarray, ws: float, mu_fit: float) -> np.ndarray:
    """
    Calculate the cross section matrix for a given incident energy, target mass,
    target temperature, outgoing energy grid and outgoing angle grid using arno
    model with the most similar sct distribution with sigma1

    Parameters
    ----------
    xs_values : np.ndarray
        Cross section values at 0K in barns
    xs_E : np.ndarray
        Energy grid of the cross section in eV
    Ein : float
        The incident energy of the neutron in eV
    M : float
        The mass of the target material in amu
    T_arno : np.ndarray, (M,)
        Temperature according to T * (1 + mu) / 2 of the material in K
    Eout : np.ndarray, (N,)
        The neutron outgoing energy grid in eV
    mu : np.ndarray, (M,)
        The cosine of the neutron outgoing angle grid in degrees (0, 180]
    Teff : np.ndarray, (M,)
        Effective temperature of the material for T_arno in K
    ws : float
        Normalization for continuous (vibrational) part. For solid is 1.
    mu_fit : float
        The cosine of the outgoing angle to fit the S(alpha, -beta) distribution
        with sigma1

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
    >>> mu = np.sort(np.cos(theta * np.pi / 180))
    >>> mu_fit = np.cos(60 / 180 * np.pi)
    >>> T_arno = T * (1 + mu) / 2
    >>> xs_values = xs_matrix_sct(xs_0K.values, xs_0K.index.values, Ein, M, T_arno, Eout, mu, T_arno, 1.0, 0.0)
    >>> pd.DataFrame(xs_values, index=theta[::-1], columns=Eout).round(6)
         1.800000  1.866667  1.933333  2.000000  2.066667  2.133333  2.200000
    180  9.103994  9.097201  9.090408  9.083550  9.076500  9.069451  9.062402
    170  9.104017  9.097224  9.090431  9.083573  9.076524  9.069475  9.062426
    160  9.104091  9.097298  9.090505  9.083643  9.076601  9.069552  9.062503
    150  9.104210  9.097417  9.090623  9.083754  9.076725  9.069676  9.062626
    140  9.104369  9.097578  9.090780  9.083906  9.076891  9.069842  9.062793
    130  9.104565  9.097777  9.090976  9.084098  9.077096  9.070050  9.062999
    120  9.104790  9.098007  9.091201  9.084322  9.077332  9.070290  9.063236
    110  9.105041  9.098260  9.091450  9.084570  9.077590  9.070553  9.063497
    100  9.105304  9.098531  9.091718  9.084836  9.077864  9.070833  9.063774
    90   9.105577  9.098806  9.091993  9.085113  9.078149  9.071120  9.064060
    80   9.105847  9.099083  9.092267  9.085388  9.078430  9.071409  9.064346
    70   9.106109  9.099349  9.092536  9.085658  9.078702  9.071686  9.064626
    60   9.106358  9.099598  9.092786  9.085910  9.078958  9.071946  9.064886
    50   9.106581  9.099824  9.093013  9.086138  9.079189  9.072181  9.065122
    40   9.106774  9.100019  9.093209  9.086333  9.079390  9.072384  9.065323
    30   9.106929  9.100178  9.093367  9.086494  9.079551  9.072547  9.065490
    20   9.107044  9.100294  9.093484  9.086612  9.079670  9.072668  9.065611
    10   9.107115  9.100364  9.093556  9.086683  9.079744  9.072743  9.065684
    """
    xs_mat = np.zeros((len(mu), len(Eout)))
    for i in prange(len(mu)):
        if mu[i] == np.cos(np.pi):
            Ein_arno = (Eout + Ein) / 2 + Ein * m / M
            xs_mat[i, :] = np.interp(Ein_arno, xs_E, xs_values)
        else:
            for j in prange(len(Eout)):
                Ein_arno = (Eout[j] + Ein) / 2 - Ein * mu[i] * m / M
                Eout_db = default_Eout(Ein_arno)
                # Distribution + Normalization:
                pdf_val = get_scat_sct_angular(Eout_db, mu_fit, Ein_arno,
                                               T_arno[i], M, Teff[i], ws)
                pdf_val /= np.trapz(pdf_val, x=Eout_db)
                # Recoil:
                recoil = Ein_arno - Eout_db[np.argmax(pdf_val)]
                # xs:
                xs_Eout_arno = np.interp(Eout_db + recoil, xs_E, xs_values)
                xs_mat[i, j] = np.trapz(xs_Eout_arno * pdf_val, x=Eout_db)
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


@nb.jit(nopython=True, nogil=False, cache=True)
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


def check_dx(data: pd.DataFrame, dx: [float, np.ndarray, pd.DataFrame], axis: [str, int]) -> [float, pd.Series, pd.DataFrame]:
    """
    Check the dx value to shift the Double Differential XS and return the value in the correct format for the shift
    """
    if isinstance(dx, float) or isinstance(dx, int) or isinstance(dx, pd.Series) or isinstance(dx, pd.DataFrame):
        return dx
    elif isinstance(dx, np.ndarray) and len(dx.shape) == 1:
        axis_ = 1 if axis == "Eout" else 0 if axis == "theta" else axis
        return pd.Series(dx, index=data.index if axis_ == 0 else data.columns)
    else:
        return pd.DataFrame(dx, index=data.index, columns=data.columns)

def reshift(data: pd.Series, dx: np.ndarray, index: pd.Index) -> pd.Series:
    """
    """
    x, y = data.index.values, data.values
    rehifted_data = np.interp(x, x + dx, y, left=0, right=0)
    return pd.Series(rehifted_data, index=index)
