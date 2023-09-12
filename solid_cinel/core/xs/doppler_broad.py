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


    # SAB algorithm (FGM):
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

    # SAB algorithm (SCT):
    >>> Teff = 1003.48
    >>> get_DB(xs_0K, Ein, M, T, Eout, theta, Teff, algorithm=algorithm, model="sct").iloc[::18, ::200].round(6)
        Eout        1.80000    1.88008    1.96016    2.04024   2.12032
    mu
    -0.999848  1.858811  12.101290  23.691448  15.003767  3.282874
    -0.945519  1.709047  11.873977  23.991555  15.194635  3.227646
    -0.798636  1.323844  11.183304  24.843531  15.736491  3.058056
    -0.573576  0.808013   9.883464  26.274694  16.646706  2.732837
    -0.292372  0.334764   7.791351  28.300141  17.934918  2.195694
     0.017452  0.067642   4.893629  30.821489  19.538760  1.425311
     0.325568  0.002956   1.850787  33.252930  21.086557  0.573503
     0.601815  0.000002   0.181653  33.106570  20.999544  0.064460
     0.819152  0.000000   0.000135  21.753425  13.801333  0.000071
     0.956305  0.000000   0.000000   0.389231   0.246971  0.000000

    # SAB algorithm (PDOS):
    >>> algorithm = "sab"
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

    # Use a displaced xs for the convolution (1D desplacement):
    >>> Eout_move = Eout + kb * T
    >>> theta = np.arange(0, 180, 1)[1::]
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
    >>> theta = np.arange(0, 180, 10)[1::]

    # Courcelle/sigma1:
    >>> algorithm = "courcelle"
    >>> get_DB(xs_0K, Ein, M, T, Eout, theta, algorithm=algorithm).iloc[::, ::200].round(6)
    Eout           1.808208   1.888288   1.968368   2.048448  2.128529
    mu
    -9.848078e-01  2.317394  13.608765  23.904997  13.510793  2.624357
    -9.396926e-01  2.174082  13.445857  24.187016  13.634857  2.576835
    -8.660254e-01  1.944376  13.157055  24.665105  13.842015  2.494349
    -7.660444e-01  1.643019  12.712963  25.345237  14.128659  2.371558
    -6.427876e-01  1.293729  12.077151  26.243932  14.492648  2.202467
    -5.000000e-01  0.929179  11.203973  27.381268  14.927536  1.980888
    -3.420201e-01  0.589072  10.044530  28.779737  15.418523  1.702736
    -1.736482e-01  0.313239   8.561272  30.461975  15.935687  1.370291
     6.123234e-17  0.128719   6.757019  32.443222  16.420854  0.998705
     1.736482e-01  0.035638   4.724137  34.711859  16.762967  0.623438
     3.420201e-01  0.005215   2.703690  37.180687  16.753071  0.301875
     5.000000e-01  0.000254   1.088985  39.563688  16.008980  0.093809
     6.427876e-01  0.000002   0.224478  41.060114  13.889304  0.012557
     7.660444e-01  0.000000   0.011002  39.570839   9.617649  0.000278
     8.660254e-01  0.000000   0.000014  30.301870   3.603613  0.000000
     9.396926e-01  0.000000   0.000000   9.402088   0.145280  0.000000
     9.848078e-01  0.000000   0.000000   0.003899   0.000000  0.000000

    # Courcelle/fgm:
    >>> get_DB(xs_0K, Ein, M, T, Eout, theta, algorithm=algorithm, model="fgm").iloc[::, ::200].round(6)
    Eout           1.808208   1.888288   1.968368   2.048448  2.128529
    mu
    -9.848078e-01  2.317392  13.608758  23.904985  13.510787  2.624355
    -9.396926e-01  2.173962  13.445171  24.185875  13.634263  2.576731
    -8.660254e-01  1.944027  13.154763  24.660940  13.839750  2.493953
    -7.660444e-01  1.642716  12.710654  25.340701  14.126167  2.371146
    -6.427876e-01  1.293533  12.075334  26.240014  14.490501  2.202144
    -5.000000e-01  0.929064  11.202598  27.377927  14.925724  1.980648
    -3.420201e-01  0.589010  10.043478  28.776741  15.416927  1.702561
    -1.736482e-01  0.313209   8.560461  30.459112  15.934201  1.370164
     6.123234e-17  0.128708   6.756411  32.440324  16.419402  0.998617
     1.736482e-01  0.035635   4.723717  34.708808  16.761507  0.623384
     3.420201e-01  0.005215   2.703446  37.177379  16.751602  0.301848
     5.000000e-01  0.000254   1.088884  39.560077  16.007541  0.093801
     6.427876e-01  0.000002   0.224456  41.056249  13.888018  0.012556
     7.660444e-01  0.000000   0.011001  39.567001   9.616729  0.000278
     8.660254e-01  0.000000   0.000014  30.298844   3.603260  0.000000
     9.396926e-01  0.000000   0.000000   9.401132   0.145266  0.000000
     9.848078e-01  0.000000   0.000000   0.003899   0.000000  0.000000

    # Courcelle/sct:
    >>> get_DB(xs_0K, Ein, M, T, Eout, theta, Teff, algorithm=algorithm, model="sct").iloc[::, ::200].round(6)
    Eout           1.808208   1.888288   1.968368   2.048448  2.128529
    mu
    -9.848078e-01  2.331977  13.611078  23.862984  13.514270  2.640078
    -9.396926e-01  2.188217  13.448568  24.143384  13.637896  2.592437
    -8.660254e-01  1.957665  13.159977  24.617642  13.843657  2.509609
    -7.660444e-01  1.655366  12.718428  25.296279  14.130508  2.386678
    -6.427876e-01  1.304740  12.086395  26.194180  14.495503  2.217417
    -5.000000e-01  0.938338  11.217557  27.330442  14.931715  1.995435
    -3.420201e-01  0.595946  10.062652  28.727467  15.424380  1.716504
    -1.736482e-01  0.317667   8.583563  30.408096  15.943809  1.382749
     6.123234e-17  0.130979   6.782097  32.387970  16.432182  1.009177
     1.736482e-01  0.036437   4.749140  34.656187  16.778930  0.631216
     3.420201e-01  0.005370   2.724341  37.126903  16.775697  0.306532
     5.000000e-01  0.000265   1.101291  39.516729  16.040752  0.095689
     6.427876e-01  0.000002   0.228394  41.029957  13.931827  0.012906
     7.660444e-01  0.000000   0.011320  39.575802   9.666471  0.000290
     8.660254e-01  0.000000   0.000014  30.363013   3.637766  0.000000
     9.396926e-01  0.000000   0.000000   9.472456   0.148504  0.000000
     9.848078e-01  0.000000   0.000000   0.004046   0.000000  0.000000


    # Courcelle/pdos:
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
        if "model" in kwargs:
            mu_fit = scattfunc.get_angle
            xs = xs_matrix_sab(mu_fit, *args, **kwargs)
        else:
            # use sigma1 model:
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
    >>> mu = np.sort(np.cos(theta * np.pi / 180))
    >>> xs_values = xs_matrix_sigma1(xs_0K.values, xs_0K.index.values, Ein, M, T, Eout, theta)
    >>> pd.DataFrame(xs_values, index=mu, columns=Eout).round(6)
                   1.800000  1.866667  1.933333  2.000000  2.066667  2.133333  2.200000
    -9.848078e-01  9.104026  9.097232  9.090439  9.083582  9.076532  9.069483  9.062434
    -9.396926e-01  9.104602  9.097777  9.090953  9.084068  9.076994  9.069920  9.062847
    -8.660254e-01  9.105858  9.099023  9.092187  9.085287  9.078207  9.071119  9.064031
    -7.660444e-01  9.106064  9.099251  9.092436  9.085552  9.078502  9.071432  9.064363
    -6.427876e-01  9.105971  9.099170  9.092365  9.085491  9.078462  9.071404  9.064345
    -5.000000e-01  9.105951  9.099155  9.092352  9.085480  9.078467  9.071414  9.064358
    -3.420201e-01  9.106047  9.099254  9.092450  9.085579  9.078577  9.071526  9.064471
    -1.736482e-01  9.106233  9.099444  9.092637  9.085765  9.078773  9.071725  9.064668
     6.123234e-17  9.106479  9.099693  9.092883  9.086010  9.079025  9.071980  9.064921
     1.736482e-01  9.106756  9.099973  9.093161  9.086288  9.079309  9.072267  9.065206
     3.420201e-01  9.107045  9.100265  9.093451  9.086577  9.079603  9.072564  9.065502
     5.000000e-01  9.107328  9.100550  9.093735  9.086860  9.079891  9.072854  9.065791
     6.427876e-01  9.107590  9.100815  9.093999  9.087124  9.080158  9.073124  9.066060
     7.660444e-01  9.107821  9.101047  9.094231  9.087355  9.080392  9.073361  9.066296
     8.660254e-01  9.108011  9.101238  9.094421  9.087546  9.080584  9.073555  9.066490
     9.396926e-01  9.108151  9.101380  9.094562  9.087687  9.080727  9.073699  9.066633
     9.848078e-01  9.108238  9.101467  9.094649  9.087774  9.080815  9.073787  9.066722



    """
    mu = np.sort(np.cos(theta * np.pi / 180))
    xs_mat = np.zeros((len(mu), len(Eout)))
    T_arno = T * (1 + mu) / 2
    for i in prange(len(mu)):
        if theta[i] == 180:
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


def xs_matrix_sab(mu_fit: float, *args, **kwargs) -> np.ndarray:
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
    >>> mu = np.sort(np.cos(theta * np.pi / 180))
    >>> mu_fit = np.cos(60 / 180 * np.pi)

    # fgm model:
    >>> xs_values = xs_matrix_sab(0.0, xs_0K, Ein, M, T, Eout, theta, model="fgm")
    >>> pd.DataFrame(xs_values, index=mu, columns=Eout).round(6)
                   1.800000  1.866667  1.933333  2.000000  2.066667  2.133333  2.200000
    -9.848078e-01  9.104017  9.097224  9.090431  9.083573  9.076524  9.069475  9.062426
    -9.396926e-01  9.104091  9.097298  9.090505  9.083643  9.076601  9.069552  9.062503
    -8.660254e-01  9.104210  9.097417  9.090623  9.083754  9.076725  9.069676  9.062626
    -7.660444e-01  9.104369  9.097578  9.090780  9.083906  9.076891  9.069842  9.062793
    -6.427876e-01  9.104565  9.097777  9.090976  9.084098  9.077096  9.070050  9.062999
    -5.000000e-01  9.104790  9.098007  9.091201  9.084322  9.077332  9.070290  9.063236
    -3.420201e-01  9.105041  9.098260  9.091450  9.084570  9.077590  9.070553  9.063497
    -1.736482e-01  9.105304  9.098531  9.091718  9.084836  9.077864  9.070833  9.063774
     6.123234e-17  9.105577  9.098806  9.091993  9.085113  9.078149  9.071120  9.064060
     1.736482e-01  9.105847  9.099083  9.092267  9.085388  9.078430  9.071409  9.064346
     3.420201e-01  9.106109  9.099349  9.092536  9.085658  9.078702  9.071686  9.064626
     5.000000e-01  9.106358  9.099598  9.092786  9.085910  9.078958  9.071946  9.064886
     6.427876e-01  9.106581  9.099824  9.093013  9.086138  9.079189  9.072181  9.065122
     7.660444e-01  9.106774  9.100019  9.093209  9.086333  9.079390  9.072384  9.065323
     8.660254e-01  9.106929  9.100178  9.093367  9.086494  9.079551  9.072547  9.065490
     9.396926e-01  9.107044  9.100294  9.093484  9.086612  9.079670  9.072668  9.065611
     9.848078e-01  9.107115  9.100364  9.093556  9.086683  9.079744  9.072743  9.065684

    # sct model:
    >>> Teff = 1003.48
    >>> xs_values = xs_matrix_sab(mu_fit, xs_0K, Ein, M, T, Eout, theta, Teff, model="sct")
    >>> pd.DataFrame(xs_values, index=mu, columns=Eout).round(6)
                   1.800000  1.866667  1.933333  2.000000  2.066667  2.133333  2.200000
    -9.848078e-01  9.103885  9.097106  9.090276  9.083375  9.076392  9.069357  9.062287
    -9.396926e-01  9.103963  9.097184  9.090354  9.083455  9.076472  9.069438  9.062368
    -8.660254e-01  9.104089  9.097311  9.090483  9.083585  9.076604  9.069570  9.062501
    -7.660444e-01  9.104261  9.097484  9.090657  9.083761  9.076782  9.069749  9.062681
    -6.427876e-01  9.104472  9.097697  9.090872  9.083979  9.077002  9.069970  9.062903
    -5.000000e-01  9.104718  9.097944  9.091121  9.084230  9.077256  9.070226  9.063161
    -3.420201e-01  9.104989  9.098217  9.091396  9.084509  9.077538  9.070509  9.063445
    -1.736482e-01  9.105277  9.098507  9.091689  9.084805  9.077837  9.070810  9.063749
     6.123234e-17  9.105575  9.098807  9.091991  9.085111  9.078147  9.071121  9.064061
     1.736482e-01  9.105872  9.099107  9.092293  9.085416  9.078456  9.071432  9.064374
     3.420201e-01  9.106161  9.099397  9.092585  9.085713  9.078755  9.071734  9.064677
     5.000000e-01  9.106431  9.099670  9.092862  9.085990  9.079036  9.072017  9.064961
     6.427876e-01  9.106676  9.099916  9.093110  9.086241  9.079290  9.072272  9.065218
     7.660444e-01  9.106886  9.100129  9.093324  9.086457  9.079509  9.072493  9.065439
     8.660254e-01  9.107057  9.100301  9.093498  9.086633  9.079687  9.072671  9.065619
     9.396926e-01  9.107183  9.100428  9.093626  9.086762  9.079817  9.072803  9.065751
     9.848078e-01  9.107261  9.100506  9.093704  9.086841  9.079897  9.072884  9.065832
    """
    if kwargs["model"] == "pdos":
        xs_0K, Ein, M, T, Eout, theta, pdos = args
        threshold = kwargs.pop("threshold", 0.0)
        nphonon = kwargs.pop("nphonon", 1000)
        tau1 = pdos.get_tau_1(T)
        debye_waller_coeff = pdos.DebyeWallerCoeff(T)
        return xs_matrix_pdos(xs_0K.values, xs_0K.index.values, Ein, M, T, Eout,
                              theta, nphonon, tau1.values, tau1.index[1],
                              threshold, debye_waller_coeff, mu_fit)
    else:
        if kwargs["model"] == "fgm":
            xs_0K, Ein, M, T, Eout, theta = args
            Teff = T
        else:
            xs_0K, Ein, M, T, Eout, theta, Teff = args
        return xs_matrix_sct(xs_0K.values, xs_0K.index.values, Ein, M, T, Eout,
                             theta, Teff, 1.0, mu_fit)


@nb.jit(nopython=True, nogil=False, cache=True, parallel=True)
def xs_matrix_pdos(xs_values: np.ndarray, xs_E: np.ndarray, Ein: float, M: float,
                   T: float, Eout: np.ndarray, theta: np.ndarray, nphonon: int,
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
    T : float
        Temperature of the material in K
    Eout : np.ndarray, (N,)
        The neutron outgoing energy grid in eV
    theta : np.ndarray, (M,)
        The neutron outgoing angle grid in degrees (0, 180]
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
    mu = np.sort(np.cos(theta * np.pi / 180))
    xs_mat = np.zeros((len(mu), len(Eout)))
    T_arno = T * (1 + mu) / 2
    for i in prange(len(mu)):
        if theta[i] == 180:
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
                  T: float, Eout: np.ndarray, theta: np.ndarray,
                  Teff: float, ws: float, mu_fit: float) -> np.ndarray:
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
    T : float
        Temperature of the material in K
    Eout : np.ndarray, (N,)
        The neutron outgoing energy grid in eV
    theta : np.ndarray, (M,)
        The neutron outgoing angle grid in degrees (0, 180]
    Teff : float
        Effective temperature of the material in K
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
    >>> mu = np.sort(np.cos(theta * np.pi / 180))
    >>> mu_fit = np.cos(60 / 180 * np.pi)
    >>> xs_values = xs_matrix_sct(xs_0K.values, xs_0K.index.values, Ein, M, T, Eout, theta, T, 1.0, 0.0)
    >>> pd.DataFrame(xs_values, index=mu, columns=Eout).round(6)
                   1.800000  1.866667  1.933333  2.000000  2.066667  2.133333  2.200000
    -9.848078e-01  9.104017  9.097224  9.090431  9.083573  9.076524  9.069475  9.062426
    -9.396926e-01  9.104091  9.097298  9.090505  9.083643  9.076601  9.069552  9.062503
    -8.660254e-01  9.104210  9.097417  9.090623  9.083754  9.076725  9.069676  9.062626
    -7.660444e-01  9.104369  9.097578  9.090780  9.083906  9.076891  9.069842  9.062793
    -6.427876e-01  9.104565  9.097777  9.090976  9.084098  9.077096  9.070050  9.062999
    -5.000000e-01  9.104790  9.098007  9.091201  9.084322  9.077332  9.070290  9.063236
    -3.420201e-01  9.105041  9.098260  9.091450  9.084570  9.077590  9.070553  9.063497
    -1.736482e-01  9.105304  9.098531  9.091718  9.084836  9.077864  9.070833  9.063774
     6.123234e-17  9.105577  9.098806  9.091993  9.085113  9.078149  9.071120  9.064060
     1.736482e-01  9.105847  9.099083  9.092267  9.085388  9.078430  9.071409  9.064346
     3.420201e-01  9.106109  9.099349  9.092536  9.085658  9.078702  9.071686  9.064626
     5.000000e-01  9.106358  9.099598  9.092786  9.085910  9.078958  9.071946  9.064886
     6.427876e-01  9.106581  9.099824  9.093013  9.086138  9.079189  9.072181  9.065122
     7.660444e-01  9.106774  9.100019  9.093209  9.086333  9.079390  9.072384  9.065323
     8.660254e-01  9.106929  9.100178  9.093367  9.086494  9.079551  9.072547  9.065490
     9.396926e-01  9.107044  9.100294  9.093484  9.086612  9.079670  9.072668  9.065611
     9.848078e-01  9.107115  9.100364  9.093556  9.086683  9.079744  9.072743  9.065684
    """
    mu = np.sort(np.cos(theta * np.pi / 180))
    xs_mat = np.zeros((len(mu), len(Eout)))
    T_arno = T * (1 + mu) / 2
    for i in prange(len(mu)):
        if theta[i] == 180:
            Ein_arno = (Eout + Ein) / 2 + Ein * m / M
            xs_mat[i, :] = np.interp(Ein_arno, xs_E, xs_values)
        else:
            Teff_ = Teff if T != Teff else T_arno[i]
            for j in prange(len(Eout)):
                Ein_arno = (Eout[j] + Ein) / 2 - Ein * mu[i] * m / M
                Eout_db = default_Eout(Ein_arno)
                # Distribution + Normalization:
                pdf_val = get_scat_sct_angular(Eout_db, mu_fit, Ein_arno, T_arno[i],
                                               M, Teff_, ws)
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
