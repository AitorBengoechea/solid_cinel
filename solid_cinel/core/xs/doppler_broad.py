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
            \frac{d^2\sigma_T(E)}{dE^\prime d^\theta} = \frac{\sigma_b}{2 * k_B * T}\sqrt{\frac{^\prime}{E}} S(\alpha(\theta, E^\prime, E, M, T), \beta( E^\prime, E, T))

        - Dopush: From the chosen S(alpha, -beta) model, the distribution more
                  similar to sigma1 is chosen. Then, the new grid for the xs is
                  calculated adding the recoil energy to the outgoing energy
        ..math::
         \frac{d^2\sigma_T(E)}{dE^\prime d^\theta} = \frac{\sigma(E^\prime + R)}{2 * k_B * T}\sqrt{\frac{^\prime}{E}} S(\alpha(\theta, E^\prime, E, M, T), \beta( E^\prime, E, T))


    Parameters for get_DB
    ---------------------
    algorithm : 'str'
        The algorithm to use for the calculation of the dopper broadened elastic
        cross section. The available algorithms are:
            - "sigma1": sigma1 algorithm from NJOY2016 manual
            - "sab": S(alpha, -beta) tables for ddxs

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

    Parameters for convolution
    --------------------------
    xs_0K : pd.Series or pd.DataFrame, (N,) or (M, N)
        0K xs data for the given material in barns. If the cross
        section is a matrix, the scattering function is convolved directly
        with xs. If the cross section is a vector, the scattering function is
        convolved with the cross section for each outgoing energy or with the
        Exs introduced by the user.
    Exs : np.array, optional, (N,) or (M, N)
        Displazed Energy grid of the cross section. If not provided, the
        energy grid of the scattering function is used.
    integral : bool, optional
        If True, the integral of the cross section is returned instead of the
        differential cross section, by default False.

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


    # Use dopush algorithm:
    >>> algorithm = "dopush"
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
    """
    algorithm = kwargs.pop("algorithm").lower()
    # Parameters for convolution:
    xs = args[0]
    integral = kwargs.pop("integral", False)
    Exs = kwargs.pop("Exs", None)

    # Create Scattering function:
    if algorithm == "sigma1":
        scattfunc = ScatFunc.from_MD(*args[1::], **kwargs)
    elif algorithm == "sab" or algorithm == "dopush":
        scattfunc = ScatFunc.from_Sab(*args[1::], **kwargs)
        if algorithm == "dopush":
            scattfunc = scattfunc.to_sd()
            Exs = scattfunc.data.index.values
            Exs += scattfunc.Ein - scattfunc.data.idxmax()
    else:
        raise ValueError("The algorithm {} is not available".format(algorithm))

    # Convolve scattering function with xs:
    return scattfunc.convolve(xs, Exs=Exs, integral=integral)

@nb.jit(nopython=True, nogil=False, cache=False, parallel=True)
def arno_xs_matrix(xs_values: np.ndarray, xs_E: np.ndarray, Ein: float,
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
    xs_0K : pd.Series
        Cross section at 0K in barns
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
    >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 10)
    >>> M = 238.05077040419212
    >>> theta = np.arange(10, 180, 10)
    >>> pd.DataFrame(arno_xs_matrix(xs_0K.values, xs_0K.index.values, Ein, M, T, Eout, theta), index=theta, columns=Eout).round(6)
         1.800000  1.844444  1.888889  1.933333  1.977778  2.022222  2.066667  2.111111  2.155556  2.200000
    10   9.108238  9.103730  9.099199  9.094649  9.090074  9.085464  9.080815  9.076135  9.071435  9.066722
    20   9.108151  9.103643  9.099112  9.094562  9.089987  9.085377  9.080727  9.076047  9.071347  9.066633
    30   9.108011  9.103501  9.098970  9.094421  9.089846  9.085235  9.080584  9.075903  9.071203  9.066490
    40   9.107821  9.103311  9.098779  9.094231  9.089656  9.085044  9.080392  9.075709  9.071009  9.066296
    50   9.107590  9.103079  9.098547  9.093999  9.089425  9.084812  9.080158  9.075473  9.070772  9.066060
    60   9.107328  9.102814  9.098282  9.093735  9.089161  9.084548  9.079891  9.075204  9.070502  9.065791
    70   9.107045  9.102529  9.097997  9.093451  9.088878  9.084263  9.079603  9.074914  9.070212  9.065502
    80   9.106756  9.102238  9.097706  9.093161  9.088590  9.083973  9.079309  9.074617  9.069915  9.065206
    90   9.106479  9.101958  9.097425  9.092883  9.088313  9.083693  9.079025  9.074331  9.069628  9.064921
    100  9.106233  9.101710  9.097177  9.092637  9.088069  9.083446  9.078773  9.074075  9.069373  9.064668
    110  9.106047  9.101520  9.096988  9.092450  9.087884  9.083257  9.078577  9.073877  9.069175  9.064471
    120  9.105951  9.101421  9.096889  9.092352  9.087787  9.083154  9.078467  9.073765  9.069062  9.064358
    130  9.105971  9.101437  9.096902  9.092365  9.087801  9.083159  9.078462  9.073757  9.069051  9.064345
    140  9.106064  9.101522  9.096979  9.092436  9.087869  9.083211  9.078502  9.073789  9.069076  9.064363
    150  9.105858  9.101301  9.096744  9.092187  9.087613  9.082932  9.078207  9.073481  9.068756  9.064031
    160  9.104602  9.100052  9.095502  9.090953  9.086395  9.081711  9.076994  9.072278  9.067562  9.062847
    170  9.104026  9.099497  9.094968  9.090439  9.085908  9.081232  9.076532  9.071833  9.067133  9.062434
    """
    mu = np.cos(theta * np.pi / 180)
    xs_mat =  np.zeros((len(mu), len(Eout)))
    T_arno = T * (1 + mu) / 2
    for i in prange(len(mu)):
        for j in prange(len(Eout)):
            Ein_arno = (Eout[j] + Ein) / 2 - Ein * mu[i] * m / M
            Eout_db = default_Eout(Ein_arno)
            pdf = sigma1(Eout_db, Ein_arno, T_arno[i], M)
            xs_Eout_arno = np.interp(Eout_db, xs_E, xs_values)
            xs_mat[i, j] = np.trapz(xs_Eout_arno * pdf, x=Eout_db)
    return xs_mat

def generate_Eout(Ein, Elim: Iterable= None, N: int=None,
                 space: str= "linear"):
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