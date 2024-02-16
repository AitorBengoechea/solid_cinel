"""
Python for working with Double Diferential XS.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
from scipy.constants import physical_constants as const
from solid_cinel.core.scattering_function import ScatFunc
from solid_cinel.core.material.vibration.pdos import Pdos
from solid_cinel.core.generic import integrate, reshift
from solid_cinel.core.xs.dxs import Dxs, check_dx
from solid_cinel.core.xs.xs_mat import XsMat
import os

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
        -0.939693  2.283483  12.162232  23.939609  15.274144  3.160730
        -0.500000  1.042632   9.808431  26.702481  17.022308  2.543099
         0.173648  0.072043   3.820128  31.943965  20.299691  0.982917
         0.766044  0.000029   0.051432  24.534948  15.423312  0.012977
        """
        scatfunction = ScatFunc.from_model(Ein, M, T, Eout, theta, *args, **kwargs)
        return cls(Ein, T, M, "S(alpha, -beta)", scatfunction.convolve(xs_0K))

    @classmethod
    def from_4PCF(cls, xs_0K: pd.Series, Ein: float, M: float, T: float,
                  Eout: np.ndarray, theta: np.ndarray, *args, **kwargs):
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
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 7)
        >>> theta = np.arange(10, 190, 10)
        >>> ddxs_test = DDxs.from_4PCF(xs_0K, Ein, M, T, Eout, theta, pdos, threshold=1.0e-14, nphonon=100, model="pdos").data
        >>> ddxs_test.set_axis(theta[::-1]).round(6)
        Eout  1.800000  1.866667   1.933333    2.000000   2.066667  2.133333  2.200000
        180   2.368698  9.777717  21.556850   22.522613  10.231718  2.251832  0.266524
        170   2.322242  9.709096  21.591080   22.629430  10.248767  2.236507  0.261395
        160   2.184305  9.495283  21.680440   22.942594  10.293308  2.188585  0.246138
        150   1.963128  9.127680  21.817322   23.476630  10.361956  2.106059  0.221630
        140   1.673261  8.592115  21.983471   24.253407  10.446299  1.985547  0.189424
        130   1.336876  7.872568  22.147225   25.305572  10.531655  1.823139  0.151908
        120   0.984164  6.957435  22.257099   26.680161  10.594023  1.615780  0.112375
        110   0.651328  5.850551  22.232566   28.445456  10.595724  1.363767  0.074832
        100   0.374294  4.587258  21.949746   30.701521  10.478464  1.074447  0.043334
        90    0.178027  3.253160  21.222714   33.598268  10.154164  0.766695  0.020801
        80    0.065776  1.993485  19.787501   37.371072   9.496498  0.473504  0.007766
        70    0.017496  0.986708  17.317657   42.432964   8.346393  0.236599  0.002087
        60    0.003118  0.359353  13.554393   49.718219   6.570866  0.087061  0.000375
        50    0.000357  0.086814   8.694447   61.832770   4.247230  0.021201  0.000043
        40    0.000026  0.013420   3.927192   83.133021   1.934737  0.003275  0.000003
        30    0.000001  0.001397   0.987060  103.939037   0.488590  0.000339  0.000000
        20    0.000000  0.000066   0.110366   82.225458   0.053962  0.000016  0.000000
        10    0.000000  0.000001   0.008850   25.054400   0.004153  0.000000  0.000000
        """
        if len(args) == 0:  # SIGMA1 or FGM
            ddxs_values = cls.gen_4PCF(xs_0K, Ein, M, T, Eout, theta,
                                       *args, **kwargs)
        elif isinstance(args[0], Pdos):  # SCT or PDOS
            ddxs_values = cls.gen_4PCF(xs_0K, Ein, M, T, Eout, theta,
                                       *args, **kwargs)
        else:  # tauN Files
            raise ValueError('Not implemented yet')
        return cls(Ein, T, M, "4PCF", ddxs_values)

    @staticmethod
    def gen_4PCF(xs_0K: pd.Series, Ein: float, M: float, T: float,
                 Eout: np.ndarray, theta: np.ndarray,
                 *args, **kwargs) -> pd.DataFrame:
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
            Minimun value to take into account in the creation of tauN functions. For T>200 is convenient to set into
            1.0e-14 to speed up the calculations. The default is 0.0.
        nphonon : 'int', optional
            Phonon expansion order. The default is 1000.

        Returns
        -------
        pd.DataFrame
            The Double Differential XS for elastic scattering
        """
        scatfunction = ScatFunc.from_model(Ein, M, T, Eout, theta,
                                           *args, **kwargs)
        if kwargs.get("model"):
            mu_fit = scatfunction.get_angle
            xs = XsMat.from_model(xs_0K, Ein, M, T, Eout, theta,
                           mu_fit, *args, **kwargs)
        else:
            xs = XsMat.from_model(xs_0K, Ein, M, T, Eout, theta)
        return scatfunction.convolve(xs.data)

    @classmethod
    def from_4PCF_recoil(cls, xs_0K: pd.Series, Ein: float, M: float, T: float,
                        Eout: np.ndarray, theta: np.ndarray, *args,
                        **kwargs) -> pd.DataFrame:
        """
        Generate the Double Differential XS for elastic scattering from Fourier
        double-Laplace transform of a 4-point correlation function modified
        ..math::
            \frac{d^2\sigma_T(E)}{dE^\prime d^\theta} = \frac{1}{2 * k_B * T}\sqrt{\frac{E^\prime}{E}} S(\alpha(\theta, E^\prime, E, M, T), \beta( E^\prime, E, T)) \sigma^{T(1+\mu)/2}((E^\prime+E + \frac{\alpha k_{B} T}{1-\mu})/2 - E \mu / A)

        For the xs matrix calculation, the gressier recoil energy is used to
        get the doppler broadening cross sections.

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
            Minimun value to take into account in the creation of tauN functions. For T>200 is convenient to set into
            1.0e-14 to speed up the calculations. The default is 0.0.
        decimal: 'float'
            Decimal precision for the calculation of the expansion order.
            The default is 1.0e-6.
        order_max: 'int'
            Maximun expansion order. The default is 5000.

        Returns
        -------
        pd.DataFrame
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

        # Coercelle with FGM algorithm:
        >>> DDxs.from_4PCF_recoil(xs_0K, Ein, M, T, Eout, theta, model="fgm").data.iloc[::, ::200].round(6)
        Eout            1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -9.848078e-01  1.799454  12.011827  23.795201  15.058833  3.254168
        -9.396926e-01  1.676368  11.820171  24.045985  15.218236  3.207631
        -8.660254e-01  1.481047  11.484990  24.467384  15.486077  3.125740
        -7.660444e-01  1.229145  10.983212  25.064040  15.865304  3.002058
        -6.427876e-01  0.943762  10.284418  25.841231  16.359271  2.827893
        -5.000000e-01  0.654927   9.354802  26.802921  16.970502  2.593117
        -3.420201e-01  0.396321   8.165883  27.947781  17.698162  2.288181
        -1.736482e-01  0.197802   6.711574  29.261084  18.532927  1.908320
         6.123234e-17  0.074460   5.037252  30.698047  19.446408  1.461220
         1.736482e-01  0.018204   3.279005  32.149110  20.369142  0.978394
         3.420201e-01  0.002218   1.689235  33.366688  21.144134  0.525276
         5.000000e-01  0.000081   0.578047  33.811182  21.429205  0.191556
         6.427876e-01  0.000000   0.090865  32.347784  20.504689  0.033458
         7.660444e-01  0.000000   0.002704  26.830185  17.009357  0.001208
         8.660254e-01  0.000000   0.000001  14.853195   9.417400  0.000001
         9.396926e-01  0.000000   0.000000   1.824966   1.157166  0.000000
         9.848078e-01  0.000000   0.000000   0.000005   0.000003  0.000000

        # Coercelle with SCT algorithm:
        >>> DDxs.from_4PCF_recoil(xs_0K, Ein, M, T, Eout, theta, pdos, model="sct").data.iloc[::, ::200].round(6)
        Eout            1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -9.848078e-01  1.812377  12.019189  23.754185  15.057238  3.271241
        -9.396926e-01  1.688889  11.828546  24.004643  15.216676  3.224750
        -8.660254e-01  1.492850  11.495045  24.425504  15.484593  3.142914
        -7.660444e-01  1.239860  10.995579  25.021448  15.863968  3.019264
        -6.427876e-01  0.952980  10.299636  25.797829  16.358208  2.845046
        -5.000000e-01  0.662266   9.373197  26.758737  16.969917  2.610041
        -3.420201e-01  0.401538   8.187375  27.903049  17.698392  2.304558
        -1.736482e-01  0.200935   6.735377  29.216391  18.534514  1.923635
         6.123234e-17  0.075917   5.061551  30.654575  19.450228  1.474719
         1.736482e-01  0.018657   3.300818  32.109059  20.376606  0.989133
         3.420201e-01  0.002291   1.705058  33.333987  21.157498  0.532397
         5.000000e-01  0.000085   0.585913  33.792571  21.451959  0.194925
         6.427876e-01  0.000000   0.092748  32.353783  20.541564  0.034273
         7.660444e-01  0.000000   0.002796  26.872254  17.063492  0.001253
         8.660254e-01  0.000000   0.000001  14.921401   9.475894  0.000001
         9.396926e-01  0.000000   0.000000   1.849270   1.174466  0.000000
         9.848078e-01  0.000000   0.000000   0.000006   0.000004  0.000000

        # Coercelle with pdos model: (Example not very accurate, only for
        # demonstration purposes)
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 7)
        >>> theta = np.arange(10, 190, 10)
        >>> ddxs_test = DDxs.from_4PCF_recoil(xs_0K, Ein, M, T, Eout, theta, pdos, threshold=1.0e-14, model="pdos").data
        >>> ddxs_test.set_axis(theta[::-1]).round(6)
        Eout  1.800000  1.866667   1.933333    2.000000   2.066667  2.133333  2.200000
        180   2.368692  9.777714  21.556849   22.522613  10.231717  2.251831  0.266523
        170   1.065432  4.510560  10.153941   10.770182   4.935073  1.089313  0.128745
        160   1.049539  4.618627  10.672548   11.426635   5.185495  1.114927  0.126765
        150   1.134515  5.332886  12.883144   14.007421   6.245290  1.281919  0.136205
        140   1.201536  6.223266  16.056316   17.858496   7.752670  1.484856  0.142710
        130   1.119359  6.632177  18.768109   21.566705   9.024823  1.570539  0.131526
        120   0.900970  6.394603  20.534025   24.703531   9.842909  1.506148  0.105078
        110   0.624735  5.624970  21.423056   27.467395  10.251675  1.321948  0.072665
        100   0.367234  4.506707  21.590946   30.234341  10.330029  1.060278  0.042802
        90    0.176504  3.227634  21.069999   33.376565  10.092712  0.762439  0.020695
        80    0.065519  1.986464  19.724574   37.263767   9.471872  0.472395  0.007749
        70    0.017464  0.985112  17.292756   42.378768   8.336923  0.236360  0.002085
        60    0.003115  0.359076  13.545275   49.689060   6.567504  0.087022  0.000375
        50    0.000357  0.086780   8.691473   61.814614   4.246158  0.021197  0.000043
        40    0.000026  0.013417   3.926490   83.120568   1.934495  0.003275  0.000003
        30    0.000001  0.001396   0.986962  103.930777   0.488559  0.000339  0.000000
        20    0.000000  0.000066   0.110359   82.221789   0.053960  0.000016  0.000000
        10    0.000000  0.000001   0.008850   25.053692   0.004152  0.000000  0.000000
        """
        scatfunction = ScatFunc.from_model(Ein, M, T, Eout, theta,
                                           *args, **kwargs)
        xs = XsMat.from_recoil(xs_0K, Ein, M, T, Eout, theta, *args,
                               **kwargs)
        ddxs = scatfunction.convolve(xs.data)
        return cls(Ein, T, M, "4PCF", ddxs)

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