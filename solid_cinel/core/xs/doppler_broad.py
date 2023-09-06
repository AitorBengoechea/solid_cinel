"""
Python file for working xs doppler broadening functions.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from scipy.constants import physical_constants as const
from solid_cinel.core.material.scattering_function.scatfunc import ScatFunc, sigma1
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


def get_DB(*args, **kwargs) -> [float, pd.Series, pd.DataFrame]:
    """
    Calculate the Double differential or singe differential dopper broadened
    cross sections for elastic scattering at a given temperature and incident
    energy using one of the following formalism:
        - sigma1: sigma1 algorithm from NJOY2016 manual
                  ..math::
                  \frac{d\sigma_T(E)}{dE^\prime} = \frac{1}{2}\sqrt{\frac{M}{m\pi k_BT}}\frac{\sqrt{E^\prime}}{E}\sigma_0(E^\prime)\left(exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} - \sqrt{E^\prime}\right)^2 \right) - exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} + \sqrt{E^\prime}\right)^2 \right)\right)

        - S(alpha, -beta): S(alpha, -beta) tables for ddxs
                           ..math::
                           \frac{d^2\sigma_T(E)}{dE^\prime d^\theta} = \frac{\sigma_b}{2 * k_B * T}\sqrt{\frac{E^\prime}{E}} S(\alpha(\theta, E^\prime, E, M, T), \beta( E^\prime, E, T))

        - Dopush: From the chosen S(alpha, -beta) model, the distribution more
                  similar to sigma1 is chosen. Then, the new grid for the xs is
                  calculated adding the recoil energy to the outgoing energy
                  ..math::
                  \frac{d\sigma_T(E)}{dE^\prime} = \frac{\sigma(E^\prime + R)}{2 * k_B * T}\sqrt{\frac{E^\prime}{E}} S(\alpha(\theta, E^\prime, E, M, T), \beta( E^\prime, E, T))

        - Courcelle: Fourier double-Laplace transform of a 4-point correlation
                     function
                     ..math::
                     \frac{d^2\sigma_T(E)}{dE^\prime d^\theta} = \frac{1}{2 * k_B * T}\sqrt{\frac{E^\prime}{E}} S(\alpha(\theta, E^\prime, E, M, T), \beta( E^\prime, E, T)) \sigma^{T(1+\mu)/2}((E^\prime+E)/2 - E \mu / A)

    Parameters for get_DB
    ---------------------
    algorithm : 'str'
        The algorithm to use for the calculation of the dopper broadened elastic
        cross section. The available algorithms are:
            - "sigma1": sigma1 algorithm from NJOY2016 manual
            - "sab": S(alpha, -beta) tables for ddxs
            - "dopush": From the chosen S(alpha, -beta) model, the distribution
                        more similar to sigma1 is chosen and a recoil energy
            - "courcelle": Fourier double-Laplace transform of a 4-point

    Parameters for convolution
    --------------------------
    xs : pd.Series or pd.DataFrame, (N,) or (M, N)
        0K xs data for the given material in barns. If the cross
        section is a matrix, the scattering function is convolved directly
        with xs. If the cross section is a vector, the scattering function is
        convolved with the cross section for each outgoing energy or with the
        Exs introduced by the user.
    Exs : np.ndarray, optional, (N,) or (M, N)
        Displazed Energy grid of the cross section. If not provided, the
        energy grid of the scattering function is used.
    integral : bool, optional
        If True, the integral of the cross section is returned instead of the
        differential cross section, by default False.

    Parameters for sigma1
    ---------------------
    Ein : float
        The incident energy of the neutron in eV
    Eout : np.array, (N,)
        The neutron outgoing energy grid in eV
    M : float
        Mass of the material in amu
    T : float
        Temperature of the material in K

    Parameters for sab
    ------------------
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
    model : str
        The model used to calculate the S(alpha, beta) distribution. The
        available models are:
            - "fgm": Free Gas Model
            - "sct": Short Collision Time
            - "pdos": Phonon Density of States

    Extra parameters for sab algoritm using pdos model
    --------------------------------------------------
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
    float or pd.Series or pd.DataFrame
        Doppler broadened differential cross section (single or double) or the
        integral value for the given temperature, incident energy and mass

    Examples
    --------
    # 0K xs data for U238:
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("doppler_broad.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> xs_0K = pd.read_csv("u238.0.2", sep="    ", header=None, engine="python").set_index(0).drop([2], axis=1).iloc[::, 0]
    >>> os.chdir(wd)
    >>> xs_0K = xs_0K[~xs_0K.index.duplicated(keep='first')]

    # Generate Broadening test results:
    >>> T = 1000
    >>> Ein = 2.0
    >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
    >>> M = 238.05077040419212

    # SIGMA1 algorithm:
    >>> algorithm = "sigma1"
    >>> get_DB(xs_0K, Ein, M, T, Eout, algorithm=algorithm).iloc[::100]
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

    >>> round(get_DB(xs_0K, Ein, M, T, Eout, algorithm=algorithm, integral=True), 2)
    9.09


    # SAB algorithm:
    >>> algorithm = "sab"
    >>> theta = np.arange(0, 180, 1)[1::]
    >>> get_DB(xs_0K, Ein, M, T, Eout, theta, algorithm=algorithm).iloc[::18, ::200].round(6)
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
    >>> round(get_DB(xs_0K, Ein, M, T, Eout, theta, algorithm=algorithm, integral=True), 2)
        9.07

    # Use a displaced xs for the convolution (1D desplacement):
    >>> Eout_move = Eout + kb * T
    >>> get_DB(xs_0K, Ein, M, T, Eout, theta, algorithm=algorithm, Exs=Eout_move).iloc[::18, ::200].round(6)
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

    >>> round(get_DB(xs_0K, Ein, M, T, Eout, theta, algorithm=algorithm, integral=True, Exs=Eout_move), 2)
        9.05

    # Use a displaced xs for the convolution (2D desplacement):
    >>> Eout_move = Eout + np.outer(np.cos(theta * np.pi / 180), np.sqrt(Eout)/M)
    >>> get_DB(xs_0K, Ein, M, T, Eout, theta, algorithm=algorithm, Exs=Eout_move).iloc[::18, ::200].round(6)
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
    >>> round(get_DB(xs_0K, Ein, M, T, Eout, theta, algorithm=algorithm, integral=True, Exs=Eout_move), 2)
    9.07

    # SAB algorithm(pdos):
    >>> from solid_cinel.core.material.vibration.pdos import Pdos
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> theta = np.array([40, 80, 120, 160])
    >>> get_DB(xs_0K, Ein, M, T, Eout, theta, pdos, threshold=1.0e-14, model="pdos", algorithm=algorithm).iloc[::, ::200].round(6)
    Eout        1.80000    1.88008    1.96016    2.04024   2.12032
    mu
    -0.939693  2.203391  11.934588  24.417997  15.575835  3.101303
    -0.500000  0.994808   9.521449  27.156911  17.307645  2.468526
     0.173648  0.066807   3.586114  32.202480  20.456875  0.922720
     0.766044  0.000026   0.045654  23.748453  14.926872  0.011525

    # Dopush algorithm:
    >>> algorithm = "dopush"
    >>> theta = np.arange(0, 180, 1)[1::]
    >>> get_DB(xs_0K, Ein, M, T, Eout, theta, algorithm=algorithm).iloc[::100]
    Eout
    1.808208     0.000163
    1.848248     0.025374
    1.888288     1.152552
    1.928328    15.775754
    1.968368    67.355332
    2.008408    92.796544
    2.048448    42.650046
    2.088488     6.756453
    2.128529     0.380893
    2.168569     0.007884
    Name: 0.5000000000000001, dtype: float64

    # Courcelle algorithm:
    >>> algorithm = "courcelle"
    >>> get_DB(xs_0K, Ein, M, T, Eout, theta, algorithm=algorithm).iloc[::18, ::200].round(6)
    Eout       1.808208   1.888288   1.968368   2.048448  2.128529
    mu
    -0.999848  2.366691  13.667442  23.824066  13.476444  2.641000
    -0.945519  2.193366  13.472951  24.159927  13.624406  2.584196
    -0.798636  1.740061  12.867109  25.124228  14.037419  2.413506
    -0.573576  1.111180  11.675304  26.785878  14.704432  2.098952
    -0.292372  0.498004   9.635481  29.256439  15.574374  1.608490
     0.017452  0.115575   6.561346  32.657570  16.463891  0.960546
     0.325568  0.006573   2.894419  36.926567  16.777201  0.329704
     0.601815  0.000010   0.399131  40.784055  14.715889  0.026084
     0.819152  0.000000   0.000803  36.327474   6.664267  0.000010
     0.956305  0.000000   0.000000   3.726110   0.014073  0.000000
    """
    algorithm = kwargs.pop("algorithm").lower()
    # Parameters for convolution:
    xs = args[0]
    integral = kwargs.pop("integral", False)
    Exs = kwargs.pop("Exs", None)

    # Create Scattering function object:
    scattfunc = algorithm_scattfunc(algorithm, *args[1::], **kwargs)

    # Update convolution parameters according to the model:
    if algorithm == "dopush":
        Exs = scattfunc.data.index.values
        # Add recoil energy to outgoing energy:
        Exs += scattfunc.Ein - scattfunc.data.idxmax()
    elif algorithm == "courcelle":
        # Create Courcelle cross section matrix:
        xs = xs_matrix_sigma1(xs.values, xs.index.values, *args[1::])

    # Convolve scattering function with xs:
    return scattfunc.convolve(xs, Exs=Exs, integral=integral)


def algorithm_scattfunc(algorithm: str, *args, **kwargs) -> ScatFunc:
    """
    Create a scattering function object for the chosen algorithm

    Parameters
    ----------
    algorithm : str
        The algorithm to use for the calculation of the dopper broadened elastic
        cross section. The available algorithms are:
            - "sigma1": sigma1 algorithm from NJOY2016 manual
            - "sab": S(alpha, -beta) tables for ddxs
            - "dopush": From the chosen S(alpha, -beta) model, the distribution
                        more similar to sigma1 is chosen.
            - "courcelle": Fourier double-Laplace transform of a 4-point
                           correlation function
    Parameters for sigma1
    ---------------------
    Ein : float
        The incident energy of the neutron in eV
    Eout : np.array, (N,)
        The neutron outgoing energy grid in eV
    M : float
        Mass of the material in amu
    T : float
        Temperature of the material in K

    Parameters for sab
    ------------------
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
    model : str
        The model used to calculate the S(alpha, beta) distribution. The
        available models are:
            - "fgm": Free Gas Model
            - "sct": Short Collision Time
            - "pdos": Phonon Density of States

    Extra parameters for sab algoritm using pdos model
    --------------------------------------------------
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
    ScatFunc
        Scattering function object.
    """
    if algorithm == "sigma1":
        scattfunc = ScatFunc.from_MD(*args, **kwargs)
    elif algorithm in ["sab", "dopush", "courcelle"]:
        scattfunc = ScatFunc.from_Sab(*args, **kwargs)
        if algorithm == "dopush":
            scattfunc = scattfunc.to_sd()
    else:
        raise ValueError("The algorithm {} is not available".format(algorithm))
    return scattfunc


@nb.jit(nopython=True, nogil=False, cache=False, parallel=True)
def xs_matrix_sigma1(xs_values: np.ndarray, xs_E: np.ndarray, Ein: float,
                     M: float, T: float, Eout: np.ndarray,
                     theta: np.ndarray) -> np.ndarray:
    """
    Calculate the cross section matrix for a given incident energy, target mass,
    target temperature, outgoing energy grid and outgoing angle grid using arno
    model.
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
    T : float
        Target temperature in K
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
    >>> os.chdir(__file__.replace("doppler_broad.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> xs_0K = pd.read_csv("u238.0.2", sep="    ", header=None, engine="python").set_index(0).drop([2], axis=1).iloc[::, 0]
    >>> os.chdir(wd)
    >>> xs_0K = xs_0K[~xs_0K.index.duplicated(keep='first')]

    >>> T = 1000
    >>> Ein = 2.0
    >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 7)
    >>> M = 238.05077040419212
    >>> theta = np.arange(10, 180, 10)
    >>> xs_values = xs_matrix_sigma1(xs_0K.values, xs_0K.index.values, Ein, M, T, Eout, theta)
    >>> pd.DataFrame(xs_values, index=theta, columns=Eout).round(6)
         1.800000  1.866667  1.933333  2.000000  2.066667  2.133333  2.200000
    10   9.108238  9.101467  9.094649  9.087774  9.080815  9.073787  9.066722
    20   9.108151  9.101380  9.094562  9.087687  9.080727  9.073699  9.066633
    30   9.108011  9.101238  9.094421  9.087546  9.080584  9.073555  9.066490
    40   9.107821  9.101047  9.094231  9.087355  9.080392  9.073361  9.066296
    50   9.107590  9.100815  9.093999  9.087124  9.080158  9.073124  9.066060
    60   9.107328  9.100550  9.093735  9.086860  9.079891  9.072854  9.065791
    70   9.107045  9.100265  9.093451  9.086577  9.079603  9.072564  9.065502
    80   9.106756  9.099973  9.093161  9.086288  9.079309  9.072267  9.065206
    90   9.106479  9.099693  9.092883  9.086010  9.079025  9.071980  9.064921
    100  9.106233  9.099444  9.092637  9.085765  9.078773  9.071725  9.064668
    110  9.106047  9.099254  9.092450  9.085579  9.078577  9.071526  9.064471
    120  9.105951  9.099155  9.092352  9.085480  9.078467  9.071414  9.064358
    130  9.105971  9.099170  9.092365  9.085491  9.078462  9.071404  9.064345
    140  9.106064  9.099251  9.092436  9.085552  9.078502  9.071432  9.064363
    150  9.105858  9.099023  9.092187  9.085287  9.078207  9.071119  9.064031
    160  9.104602  9.097777  9.090953  9.084068  9.076994  9.069920  9.062847
    170  9.104026  9.097232  9.090439  9.083582  9.076532  9.069483  9.062434

    """
    mu = np.cos(theta * np.pi / 180)
    xs_mat = np.zeros((len(mu), len(Eout)))
    T_arno = T * (1 + mu) / 2
    for i in prange(len(mu)):
        for j in prange(len(Eout)):
            Ein_arno = (Eout[j] + Ein) / 2 - Ein * mu[i] * m / M
            Eout_db = default_Eout(Ein_arno)
            pdf = sigma1(Eout_db, Ein_arno, T_arno[i], M)
            xs_Eout_arno = np.interp(Eout_db, xs_E, xs_values)
            xs_mat[i, j] = np.trapz(xs_Eout_arno * pdf, x=Eout_db)
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
    >>> os.chdir(__file__.replace("doppler_broad.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> xs_0K = pd.read_csv("u238.0.2", sep="    ", header=None, engine="python").set_index(0).drop([2], axis=1).iloc[::, 0]
    >>> os.chdir(wd)
    >>> xs_0K = xs_0K[~xs_0K.index.duplicated(keep='first')]

    # Common data:
    >>> T = 1000
    >>> Ein = 2.0
    >>> Eout = default_Eout(Ein)
    >>> M = 238.05077040419212

    # Test default grid:
    >>> Eout = default_Eout(Ein)
    >>> round(get_DB(xs_0K, Ein, M, T, Eout, algorithm="sigma1", integral=True), 2)
    9.09

    # Test linear grid:
    >>> Eout = generate_Eout(Ein, Elim=[Ein * 0.9, Ein * 1.1], N=5000)
    >>> round(get_DB(xs_0K, Ein, M, T, Eout, algorithm="sigma1", integral=True), 2)
    9.09

    # Test logarithmic grid:
    >>> Eout = generate_Eout(Ein, Elim=[Ein * 0.9, Ein * 1.1], N=5000, space="log")
    >>> round(get_DB(xs_0K, Ein, M, T, Eout, algorithm="sigma1", integral=True), 2)
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
    >>> os.chdir(__file__.replace("doppler_broad.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> xs_0K = pd.read_csv("u238.0.2", sep="    ", header=None, engine="python").set_index(0).drop([2], axis=1).iloc[::, 0]
    >>> os.chdir(wd)
    >>> xs_0K = xs_0K[~xs_0K.index.duplicated(keep='first')]

    # Generate Broadening test results:
    >>> T = 1000
    >>> Ein = 2.0
    >>> Eout = default_Eout(Ein)
    >>> M = 238.05077040419212
    >>> round(get_DB(xs_0K, Ein, M, T, Eout, algorithm="sigma1", integral=True), 2)
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
