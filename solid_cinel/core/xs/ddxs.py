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
from solid_cinel.core.scattering_function.scatfunc import ScatFunc
from solid_cinel.core.material.vibration.pdos import Pdos
from solid_cinel.core.xs import Xs, Dxs
from solid_cinel.core.generic import integrate, reshift
from solid_cinel.core.xs.dxs import check_dx

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
        Generate the Double Differential XS for elastic scattering from S(alpha, -beta) tables
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
        >>> xs0K = pd.read_csv("u238.0.2", delim_whitespace=True, header = None, index_col = 0, usecols = [0, 1], engine = "python").iloc[::, 0]
        >>> xs0K.index.name = "E"
        >>> xs0K = xs0K.reset_index().drop_duplicates(subset='E', keep='first').set_index('E').iloc[:, 0]
        >>> os.chdir(wd)

        # Get the Xs object:
        >>> M = 238.05077040419212
        >>> xs = Xs(M, 0, xs0K)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> theta = np.arange(0, 180, 1)[1::]
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)

        # S(alpha, -beta) algorithm for FGM:
        >>> DDxs.from_Sab(xs, Ein, T, Eout, theta, model="fgm").data.iloc[::18, ::200].round(6)
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
        >>> DDxs.from_Sab(xs, Ein, T, Eout, theta, pdos, model="sct").data.iloc[::18, ::200].round(6)
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
        >>> DDxs.from_Sab(xs, Ein, T, Eout, theta, pdos, threshold=1.0e-14, model="pdos").data.iloc[::, ::200].round(6)
        Eout        1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -0.939693  2.283483  12.162232  23.939609  15.274144  3.160730
        -0.500000  1.042632   9.808431  26.702481  17.022308  2.543099
         0.173648  0.072043   3.820128  31.943965  20.299691  0.982917
         0.766044  0.000029   0.051432  24.534948  15.423312  0.012977
        """
        # Get the scattering function:
        scatfunction = ScatFunc.from_model(Ein, xs.M, T, Eout, theta, *args, **kwargs)

        # Get the cross section in the correct energy grid:
        xs0Kinterp = xs.interp_Ein(Eout, T=0).loc[::, 0]

        # Calculate the convolution:
        ddxs = scatfunction.data * xs0Kinterp

        return cls(Ein, T, xs.M, ddxs)

    @classmethod
    def from_4PCF(cls, xs: Xs, Ein: float, T: float, Eout: np.ndarray,
                  theta: np.ndarray, *args, algorithm: str = "sigma1", **kwargs):
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
        >>> xs0K = pd.read_csv("u238.0.2", delim_whitespace=True, header = None, index_col = 0, usecols = [0, 1], engine = "python").iloc[::, 0]
        >>> xs0K.index.name = "E"
        >>> xs0K = xs0K.reset_index().drop_duplicates(subset='E', keep='first').set_index('E').iloc[:, 0]
        >>> os.chdir(wd)

        # Get the Xs object:
        >>> M = 238.05077040419212
        >>> xs = Xs(M, 0, xs0K)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> theta = np.arange(0, 180, 10)[1::]
        >>> index = pd.Index(theta[::-1], name="theta")
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)

        # Coercelle with sigma1 algorithm:
        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, model="fgm").data.iloc[::, ::200]
        >>> ddxs_test.set_axis(index).round(6)
        Eout   1.80000    1.88008    1.96016    2.04024   2.12032
        theta
        170   1.799455  12.011833  23.795214  15.058840  3.254169
        160   1.676460  11.820765  24.047103  15.218889  3.207758
        150   1.481311  11.486974  24.471480  15.488588  3.126231
        140   1.229370  10.985188  25.068483  15.868074  3.002575
        130   0.943903  10.285942  25.845029  16.361656  2.828302
        120   0.655005   9.355920  26.806106  16.972506  2.593422
        110   0.396361   8.166704  27.950568  17.699917  2.288406
        100   0.197819   6.712172  29.263668  18.534550  1.908486
        90    0.074467   5.037671  30.700572  19.447990  1.461338
        80    0.018206   3.279271  32.151673  20.370743  0.978470
        70    0.002219   1.689372  33.369338  21.145787  0.525316
        60    0.000081   0.578095  33.813903  21.430898  0.191570
        50    0.000000   0.090873  32.350444  20.506342  0.033461
        40    0.000000   0.002704  26.832445  17.010759  0.001208
        30    0.000000   0.000001  14.854473   9.418193  0.000001
        20    0.000000   0.000000   1.825126   1.157265  0.000000
        10    0.000000   0.000000   0.000005   0.000003  0.000000

        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, pdos, model="sct").data.iloc[::, ::200]
        >>> ddxs_test.set_axis(index).round(6)
        Eout    1.80000    1.88008    1.96016    2.04024   2.12032
        theta
        170    1.812378  12.019195  23.754255  15.057246  3.271243
        160    1.688981  11.829141  24.005795  15.217329  3.224878
        150    1.493116  11.497031  24.429613  15.487104  3.143408
        140    1.240086  10.997557  25.025894  15.866739  3.019783
        130    0.953122  10.301162  25.801627  16.360593  2.845458
        120    0.662345   9.374318  26.761920  16.971921  2.610348
        110    0.401578   8.188197  27.905835  17.700146  2.304785
        100    0.200953   6.735978  29.218973  18.536138  1.923802
        90     0.075924   5.061972  30.657097  19.451811  1.474838
        80     0.018659   3.301086  32.111619  20.378208  0.989209
        70     0.002291   1.705196  33.336635  21.159152  0.532438
        60     0.000085   0.585961  33.795291  21.453654  0.194940
        50     0.000000   0.092756  32.356444  20.543220  0.034276
        40     0.000000   0.002796  26.874517  17.064899  0.001253
        30     0.000000   0.000001  14.922686   9.476692  0.000001
        20     0.000000   0.000000   1.849432   1.174567  0.000000
        10     0.000000   0.000000   0.000006   0.000004  0.000000

        # Coercelle with pdos model: (Example not very accurate, only for
        # demonstration purposes)
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 7)
        >>> theta = np.arange(10, 190, 10)
        >>> index = pd.Index(theta[::-1], name="theta")
        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, pdos, threshold=1.0e-14, nphonon=100, model="pdos").data
        >>> ddxs_test.set_axis(index).round(6)
        Eout   1.800000  1.866667   1.933333    2.000000   2.066667  2.133333  2.200000
        theta
        180    2.368698  9.777717  21.556850   22.522613  10.231718  2.251832  0.266524
        170    2.321903  9.707659  21.587844   22.625885  10.247125  2.236144  0.261352
        160    2.184144  9.494530  21.678610   22.940447  10.292278  2.188355  0.246111
        150    1.963352  9.128663  21.819529   23.478804  10.362838  2.106225  0.221646
        140    1.673562  8.593624  21.987238   24.257452  10.447988  1.985860  0.189452
        130    1.337120  7.873980  22.151133   25.309972  10.533444  1.823442  0.151932
        120    0.984333  6.958613  22.260799   26.684541  10.595729  1.616035  0.112392
        110    0.651434  5.851483  22.236041   28.449841  10.597323  1.363968  0.074842
        100    0.374352  4.587958  21.953029   30.706029  10.479973  1.074598  0.043340
        90     0.178054  3.253646  21.225810   33.603077  10.155584  0.766800  0.020804
        80     0.065786  1.993781  19.790373   37.376383   9.497815  0.473568  0.007767
        70     0.017498  0.986856  17.320184   42.439017   8.347557  0.236631  0.002087
        60     0.003118  0.359408  13.556401   49.725412   6.571793  0.087073  0.000375
        50     0.000357  0.086827   8.695622   61.840892   4.247772  0.021204  0.000043
        40     0.000026  0.013422   3.927734   83.144145   1.934989  0.003275  0.000003
        30     0.000001  0.001397   0.987198  103.953211   0.488655  0.000339  0.000000
        20     0.000000  0.000066   0.110381   82.236817   0.053969  0.000016  0.000000
        10     0.000000  0.000001   0.008851   25.057890   0.004153  0.000000  0.000000

        # alpha0:
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> theta = np.arange(0, 180, 10)[1::]
        >>> index = pd.Index(theta[::-1], name="theta")
        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, algorithm="alpha0", model="fgm").data.iloc[::, ::200]
        >>> ddxs_test.set_axis(index).round(6)
        Eout    1.80000    1.88008    1.96016    2.04024   2.12032
        theta
        170    1.799454  12.011827  23.795201  15.058833  3.254168
        160    1.676368  11.820171  24.045985  15.218236  3.207631
        150    1.481047  11.484990  24.467384  15.486077  3.125740
        140    1.229145  10.983212  25.064040  15.865304  3.002058
        130    0.943762  10.284418  25.841231  16.359271  2.827893
        120    0.654927   9.354802  26.802921  16.970502  2.593117
        110    0.396321   8.165883  27.947781  17.698162  2.288181
        100    0.197802   6.711574  29.261084  18.532927  1.908320
        90     0.074460   5.037252  30.698047  19.446408  1.461220
        80     0.018204   3.279005  32.149110  20.369142  0.978394
        70     0.002218   1.689235  33.366688  21.144134  0.525276
        60     0.000081   0.578047  33.811182  21.429205  0.191556
        50     0.000000   0.090865  32.347784  20.504689  0.033458
        40     0.000000   0.002704  26.830185  17.009357  0.001208
        30     0.000000   0.000001  14.853195   9.417400  0.000001
        20     0.000000   0.000000   1.824966   1.157166  0.000000
        10     0.000000   0.000000   0.000005   0.000003  0.000000

        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, pdos, algorithm="alpha0", model="sct").data.iloc[::, ::200]
        >>> ddxs_test.set_axis(index).round(6)
        Eout    1.80000    1.88008    1.96016    2.04024   2.12032
        theta
        170    1.418728   9.432845  18.689928  11.876672  2.586581
        160    1.422079   9.973461  20.267428  12.864941  2.730009
        150    1.366027  10.522000  22.365509  14.183555  2.879851
        140    1.190620  10.558988  24.028284  15.234573  2.899543
        130    0.934896  10.103778  25.306300  16.045967  2.790649
        120    0.655777   9.281028  26.494695  16.801908  2.584118
        110    0.399326   8.142063  27.747922  17.599566  2.291635
        100    0.200261   6.712676  29.117395  18.471388  1.917050
        90     0.075751   5.050406  30.586677  19.406900  1.471416
        80     0.018629   3.295766  32.059596  20.345022  0.987590
        70     0.002289   1.703141  33.296246  21.133386  0.531786
        60     0.000085   0.585404  33.763011  21.433065  0.194752
        50     0.000000   0.092683  32.330895  20.526930  0.034248
        40     0.000000   0.002794  26.856213  17.053232  0.001252
        30     0.000000   0.000001  14.913579   9.470889  0.000001
        20     0.000000   0.000000   1.848385   1.173900  0.000000
        10     0.000000   0.000000   0.000006   0.000004  0.000000

        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 7)
        >>> theta = np.arange(10, 190, 10)
        >>> index = pd.Index(theta[::-1], name="theta")
        >>> ddxs_test = DDxs.from_4PCF(xs, Ein, T, Eout, theta, pdos, algorithm="alpha0", threshold=1.0e-14, nphonon=100, model="pdos").data
        >>> ddxs_test.set_axis(index).round(6)
        Eout   1.800000  1.866667   1.933333    2.000000   2.066667  2.133333  2.200000
        theta
        180    2.368698  9.777717  21.556850   22.522613  10.231718  2.251832  0.266524
        170    2.322358  9.709553  21.592034   22.630398  10.249174  2.236589  0.261404
        160    2.184449  9.495879  21.681735   22.943925  10.293871  2.188697  0.246150
        150    1.963212  9.128038  21.818094   23.477408  10.362263  2.106113  0.221635
        140    1.673276  8.592165  21.983522   24.253411  10.446267  1.985534  0.189422
        130    1.336884  7.872591  22.147231   25.305535  10.531607  1.823125  0.151906
        120    0.984180  6.957531  22.257347   26.680416  10.594098  1.615787  0.112375
        110    0.651347  5.850704  22.233098   28.446094  10.595934  1.363790  0.074833
        100    0.374309  4.587426  21.950504   30.702523  10.478783  1.074477  0.043335
        90     0.178035  3.253304  21.223608   33.599626  10.154549  0.766722  0.020802
        80     0.065779  1.993584  19.788448   37.372793   9.496911  0.473523  0.007766
        70     0.017497  0.986762  17.318553   42.435080   8.346791  0.236609  0.002087
        60     0.003118  0.359373  13.555137   49.720852   6.571197  0.087065  0.000375
        50     0.000357  0.086818   8.694808   61.835208   4.247386  0.021202  0.000043
        40     0.000026  0.013420   3.927362   83.136428   1.934812  0.003275  0.000003
        30     0.000001  0.001397   0.987104  103.943448   0.488610  0.000339  0.000000
        20     0.000000  0.000066   0.110371   82.229012   0.053964  0.000016  0.000000
        10     0.000000  0.000001   0.008851   25.055494   0.004153  0.000000  0.000000
        """
        # Generate Scatering function:
        scatfunc = ScatFunc.from_model(Ein, xs.M, T, Eout, theta,*args, **kwargs).data

        # Use only Eout values with information for optimization:
        Eout_ = scatfunc.columns.values

        # Get Xs matrix for convolution with scattering function
        kwargs["algorithm"] = algorithm
        xsMat = xs.get_4PCFxs(Ein, T, Eout_, theta, *args, **kwargs)

        # Convolve the scattering function with the cross section matrix:
        ddxs = scatfunc * xsMat

        return cls(Ein, T, xs.M, ddxs)

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
        >>> xs0K = pd.read_csv("u238.0.2", delim_whitespace=True, header = None, index_col = 0, usecols = [0, 1], engine = "python").iloc[::, 0]
        >>> xs0K.index.name = "E"
        >>> xs0K = xs0K.reset_index().drop_duplicates(subset='E', keep='first').set_index('E').iloc[:, 0]
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> theta = np.arange(0, 180, 1)[1::]

        # Angular distribution:
        >>> DDxs.from_Sab(xs0K, Ein, M, T, Eout, theta, model="fgm").angular.data.iloc[::200].round(6)
        Eout
        1.80000     0.768794
        1.88008    10.451361
        1.96016    54.522950
        2.04024    34.506930
        2.12032     2.920481
        dtype: float64
        """
        dxs = self.data.apply(integrate, axis=0)
        return Dxs(self.Ein, self.T, self.M, self.algorithm, dxs)
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
        >>> xs0K = pd.read_csv("u238.0.2", delim_whitespace=True, header = None, index_col = 0, usecols = [0, 1], engine = "python").iloc[::, 0]
        >>> xs0K.index.name = "E"
        >>> xs0K = xs0K.reset_index().drop_duplicates(subset='E', keep='first').set_index('E').iloc[:, 0]
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> theta = np.arange(0, 180, 1)[1::]
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)

        # S(alpha, -beta) algorithm for FGM:
        >>> round(DDxs.from_Sab(xs0K, Ein, M, T, Eout, theta, model="fgm").integral, 2)
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
        >>> xs0K = pd.read_csv("u238.0.2", delim_whitespace=True, header = None, index_col = 0, usecols = [0, 1], engine = "python").iloc[::, 0]
        >>> xs0K.index.name = "E"
        >>> xs0K = xs0K.reset_index().drop_duplicates(subset='E', keep='first').set_index('E').iloc[:, 0]
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> theta = np.arange(0, 180, 15)[1::]
        >>> ddxs = DDxs.from_Sab(xs0K, Ein, M, T, Eout, theta)
        >>> probabilities = ddxs.Eprob
        >>> round(probabilities["upscattering"], 6)
        0.389484
        >>> round(probabilities["downscattering"], 6)
        0.60678
        >>> round(probabilities["Ein=Eout"], 6)
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
        >>> xs0K = pd.read_csv("u238.0.2", delim_whitespace=True, header = None, index_col = 0, usecols = [0, 1], engine = "python").iloc[::, 0]
        >>> xs0K.index.name = "E"
        >>> xs0K = xs0K.reset_index().drop_duplicates(subset='E', keep='first').set_index('E').iloc[:, 0]
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> theta = np.arange(0, 180, 15)[1::]
        >>> ddxs = DDxs.from_Sab(xs0K, Ein, M, T, Eout, theta)
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
        >>> xs0K = pd.read_csv("u238.0.2", delim_whitespace=True, header = None, index_col = 0, usecols = [0, 1], engine = "python").iloc[::, 0]
        >>> xs0K.index.name = "E"
        >>> xs0K = xs0K.reset_index().drop_duplicates(subset='E', keep='first').set_index('E').iloc[:, 0]
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> theta = np.arange(0, 180, 15)[1::]
        >>> ddxs = DDxs.from_Sab(xs0K, Ein, M, T, Eout, theta)
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
        >>> xs0K = pd.read_csv("u238.0.2", delim_whitespace=True, header = None, index_col = 0, usecols = [0, 1], engine = "python").iloc[::, 0]
        >>> xs0K.index.name = "E"
        >>> xs0K = xs0K.reset_index().drop_duplicates(subset='E', keep='first').set_index('E').iloc[:, 0]
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> theta = np.arange(0, 180, 15)[1::]
        >>> ddxs = DDxs.from_Sab(xs0K, Ein, M, T, Eout, theta)
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