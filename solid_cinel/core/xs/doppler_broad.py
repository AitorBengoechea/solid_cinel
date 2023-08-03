"""
Python file for working xs doppler broadening functions.

@author: AB272525
"""
import numpy as np
import pandas as pd
from scipy.constants import physical_constants as const
from solid_cinel.core.material.scattering_function.scatfunc import ScatFunc
import os

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
    >>> Ein = 2.0
    >>> T = 1000
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