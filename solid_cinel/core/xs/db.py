import numpy as np
import pandas as pd
from solid_cinel.core.xs import XsMat, ScatFunc, DDxs, Pdos
from solid_cinel.core.scattering_function.alpha import get_alpha_from_Eout, get_expansion_order
from solid_cinel.core.material.vibration.tau import save_tau
import warnings

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


def get_results(results: dict, prob: bool) -> pd.DataFrame:
    """
    Get the results in a DataFrame

    Parameters
    ----------
    results: dict
        Results of the calculation
    prob: bool
        If True, apart from the cross section, the probability of each energy
        for downscattering, upscattering and Ein=Eout is calculated.

    Returns
    -------
    pd.DataFrame
        Cross section at each energy in Ein_grid for the selected temperature
        and the downscattering, upscattering and Ein=Eout probabilities if prob
        is True.
    """
    xs_db = pd.DataFrame(results).T.sort_index()
    xs_db.index.name = "Ein"
    columns_order = ["xs", "downscattering", "upscattering", "Ein=Eout"]
    return xs_db[columns_order] if prob else xs_db

def from_fgm(xs_0K: pd.Series, Ein_grid: np.ndarray, M: float, T: float,
             theta_diff: float, Eout_num: int,
             prob: bool) -> pd.DataFrame:
    """
    Generate doppler broadening cross section using 4PCF with SIGMA1 doppler
    broadened XsMat and scattering function based on FGM.

    Parameters
    ----------
    xs_0K: pd.Series
        Cross section at 0K
    Ein_grid: np.ndarray
        Energies to calculate the cross section in eV
    M: float
        Mass of the target in amu
    T: float
        Temperature in K
    theta_diff: float
        Angle step in degrees
    Eout_num: int
        Number of energies to calculate the cross section
    prob: bool
        If True, apart from the cross section, the probability of each energy
        for downscattering, upscattering and Ein=Eout is calculated.

    Returns
    -------
    pd.DataFrame
        Cross section at each energy in Ein_grid for the selected temperature
        and the downscattering, upscattering and Ein=Eout probabilities if prob
        is True.
    """
    results = {}
    theta = np.arange(1, 180 + theta_diff, theta_diff)
    for Ein in Ein_grid:
        Eout = np.linspace(Ein * 0.95, Ein * 1.05, Eout_num)
        # XsMat
        xs_mat = XsMat.from_model(xs_0K, Ein, M, T, Eout, theta)
        # Scattering function:
        scatfunc = ScatFunc.from_model(Ein, M, T, Eout, theta, model="fgm")
        # DDxs
        ddxs = DDxs(Ein, T, M, "4PCF(FGM)", scatfunc.convolve(xs_mat.data))
        result = {"xs": ddxs.integral}
        if prob:
            result.update(ddxs.E_prob)
        results[Ein] = result
    return get_results(results, prob)


def from_sct(xs_0K: pd.Series, Ein_grid: np.ndarray, M: float, T: float,
             theta_diff: float, Eout_num: int, prob: bool,
             pdos: Pdos) -> pd.DataFrame:
    """
    Generate doppler broadening cross section using 4PCF with SIGMA1 doppler
    broadened XsMat and scattering function based on SCT.

    Parameters
    ----------
    xs_0K: pd.Series
        Cross section at 0K
    Ein_grid: np.ndarray
        Energies to calculate the cross section in eV
    M: float
        Mass of the target in amu
    T: float
        Temperature in K
    theta_diff: float
        Angle step in degrees
    Eout_num: int
        Number of energies to calculate the cross section
    pdos: Pdos
        Phonon density of states of the target
    prob: bool
        If True, apart from the cross section, the probability of each energy
        for downscattering, upscattering and Ein=Eout is calculated.

    Returns
    -------
    pd.DataFrame
        Cross section at each energy in Ein_grid for the selected temperature
        and the downscattering, upscattering and Ein=Eout probabilities if prob
        is True.
    """
    results = {}
    theta = np.arange(1, 180 + theta_diff, theta_diff)
    for Ein in Ein_grid:
        Eout = np.linspace(Ein * 0.95, Ein * 1.05, Eout_num)
        # XsMat
        xs_mat = XsMat.from_model(xs_0K, Ein, M, T, Eout, theta)
        # Scattering function:
        scatfunc = ScatFunc.from_model(Ein, M, T, Eout, theta, pdos, model="sct")
        # DDxs
        ddxs = DDxs(Ein, T, M, "4PCF(SCT)", scatfunc.convolve(xs_mat.data))
        result = {"xs": ddxs.integral}
        if prob:
            result.update(ddxs.E_prob)
        results[Ein] = result
    return get_results(results, prob)


def from_pdos(xs_0K: pd.Series, Ein_grid: np.ndarray, M: float, T: float,
              theta_diff: float, Eout_num: int, prob: bool, pdos: Pdos,
              nphonon: int = None, decimal: float = 1.0e-6,
              n_order_max: int = 5000, threshold: float = 0.0,
              tau_to_file: bool = False, binary: bool = False) -> pd.DataFrame:
    """
    Generate doppler broadening cross section using 4PCF with SIGMA1 doppler
    broadened XsMat and scattering function based on Phonon Expansion.

    Parameters
    ----------
    xs_0K: pd.Series
        Cross section at 0K
    Ein_grid: np.ndarray
        Energies to calculate the cross section in eV
    M: float
        Mass of the target in amu
    T: float
        Temperature in K
    theta_diff: float
        Angle step in degrees
    Eout_num: int
        Number of energies to calculate the cross section
    pdos: Pdos
        Phonon density of states of the target
    prob: bool
        If True, apart from the cross section, the probability of each energy
        for downscattering, upscattering and Ein=Eout is calculated.
    nphonon: int
        Expansion order. If None, the order is calculated automatically.
    decimal: float
        Precision of the tau_n functions
    n_order_max: int
        Maximum expansion order
    threshold: float
        Threshold to calculate the tau_n functions
    tau_to_file: bool
        If True, the tau_n functions are saved to a file
    binary: bool
        If True, the tau_n functions are saved in binary format

    Returns
    -------
    pd.DataFrame
        Cross section at each energy in Ein_grid for the selected temperature
        and the downscattering, upscattering and Ein=Eout probabilities if prob
        is True.
    """
    results = {}
    theta = np.arange(1, 180 + theta_diff, theta_diff)
    mu = np.cos(np.deg2rad(theta))

    # Calculate the tau_n functions:
    debye_waller_coeff = pdos.DebyeWallerCoeff(T)
    delta_beta = pdos.to_beta_grid(T).grid
    if nphonon:
        warnings.warn(
            "Is posible that the expansion order is not enough to get the correct results")
    else:
        nphonon = get_expansion_order(
            get_alpha_from_Eout(1.05 * Ein_grid[-1], Ein_grid[-1], M, T, mu.min()),
            debye_waller_coeff, decimal, n_order_max)
    tau_n = pdos.get_tau(T, nphonon, threshold, values=True)
    save_tau(tau_n, nphonon, T, tau_to_file, binary)

    # start the loop
    for Ein in Ein_grid:
        Eout = np.linspace(Ein * 0.95, Ein * 1.05, Eout_num)
        # XsMat
        xs_mat = XsMat.from_model(xs_0K, Ein, M, T, Eout, theta)
        # Minimize the expansion order for each energy:
        min_nphonon = get_expansion_order(
            get_alpha_from_Eout(1.05 * Ein, Ein, M, T, mu.min()),
            debye_waller_coeff, decimal, n_order_max)
        # Scattering function:
        scatfunc = ScatFunc.from_tau(Ein, M, T, Eout, mu, tau_n[:min_nphonon],
                                     delta_beta, debye_waller_coeff)
        # DDxs
        ddxs = DDxs(Ein, T, M, "4PCF(CLM)", scatfunc.convolve(xs_mat.data))
        result = {"xs": ddxs.integral}
        if prob:
            result.update(ddxs.E_prob)
        results[Ein] = result
    return get_results(results, prob)


def from_model(xs_0K: pd.Series, Ein_grid: np.ndarray, M: float, T: float,
               *args, theta_diff: float = 1.0, Eout_num: int = 3000,
               model: str = "fgm", prob: bool = True, **kwargs) -> pd.DataFrame:
    """
    Generate doppler broadening cross section using 4PCF with SIGMA1 doppler
    broadened XsMat and scattering function based on the selected model.

    Parameters
    ----------
    xs_0K: pd.Series
        Cross section at 0K
    Ein_grid: np.ndarray
        Energies to calculate the cross section in eV
    M: float
        Mass of the target in amu
    T: float
        Temperature in K
    theta_diff: float
        Angle step in degrees
    Eout_num: int
        Number of energies to calculate the cross section
    model: str
        Model to calculate the scattering function. Options: "fgm", "sct" and
        "pdos". Default: "fgm"
    prob: bool, optional
        If True, apart from the cross section, the probability of each energy
        for downscattering, upscattering and Ein=Eout is calculated.

    Parameters for SCT
    ------------------
    pdos: Pdos
        Phonon density of states of the target

    Parameters for PDOS
    -------------------
    nphonon: int
        Expansion order. If None, the order is calculated automatically.
    decimal: float
        Precision of the tau_n functions. Default: 1.0e-6
    n_order_max: int
        Maximum expansion order. Default: 5000
    threshold: float
        Threshold to calculate the tau_n functions. Default: 0.0
    tau_to_file: bool
        If True, the tau_n functions are saved to a file. Default: False
    binary: bool
        If True, the tau_n functions are saved in binary format. Default: False

    Returns
    -------
    pd.DataFrame
        Cross section at each energy in Ein_grid for the selected temperature
        and the downscattering, upscattering and Ein=Eout probabilities if prob
        is True.

    Examples
    --------
    # 0K xs data for U238:
    >>> import os
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("db.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
    >>> os.chdir(wd)

    # Generate DDXS test variables:
    >>> T = 300
    >>> M = 238.05077040419212
    >>> Ein = np.array([6.67])
    >>> from_model(xs_0K, Ein, M, T, model="fgm", Eout_num=1000)
                  xs  downscattering  upscattering  Ein=Eout
    Ein
    6.67  456.375935        0.846605      0.148305   0.00509

    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> from_model(xs_0K, Ein, M, T, pdos, model="sct", Eout_num=1000)
                  xs  downscattering  upscattering  Ein=Eout
    Ein
    6.67  453.228141        0.843325      0.151599  0.005077

    >>> from_model(xs_0K, Ein, M, T, pdos, model="pdos", Eout_num=1000)
                  xs  downscattering  upscattering  Ein=Eout
    Ein
    6.67  444.117578        0.852297      0.143335  0.004368
    """
    model = model.lower()
    if model == "pdos":
        return from_pdos(xs_0K, Ein_grid, M, T, theta_diff, Eout_num, prob,
                         *args, **kwargs)
    elif model == "sct":
        return from_sct(xs_0K, Ein_grid, M, T, theta_diff, Eout_num, prob,
                        *args)
    else:
        return from_fgm(xs_0K, Ein_grid, M, T, theta_diff, Eout_num, prob)






