import numpy as np
import pandas as pd
from scipy.constants import physical_constants as const
from solid_cinel.core.xs import XsMat, ScatFunc, DDxs, Pdos, Sab
from solid_cinel.core.scattering_function.alpha import get_alpha_from_Eout, get_expansion_order, get_gressier_recoil
from solid_cinel.core.scattering_function.beta import get_beta
from solid_cinel.core.scattering_function import get_scatfunc_pdos_row
from solid_cinel.core.material.vibration.tau import save_tau, tau_n_functions
from solid_cinel.core.xs.xs import ddxs_clm_0K, Ein_arno_row
from solid_cinel.core.xs.xs_mat import update_xs_mat_pdos_recoil_row, default_abs_beta
from solid_cinel.core.generic import reshape_differential, integrate
import warnings

# constants
kb = const["Boltzmann constant in eV/K"][0]

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
              order_max: int = 5000, threshold: float = 0.0,
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
    order_max: int
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
            debye_waller_coeff, decimal, order_max)
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
            debye_waller_coeff, decimal, order_max)
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
    6.67  444.117576        0.852297      0.143335  0.004368
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


def from_recoil_fgm(xs_0K: pd.Series, Ein_grid: np.ndarray, M: float, T: float,
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
        xs_mat = XsMat.from_recoil(xs_0K, Ein, M, T, Eout, theta)
        # Scattering function:
        scatfunc = ScatFunc.from_model(Ein, M, T, Eout, theta, model="fgm")
        # DDxs
        ddxs = DDxs(Ein, T, M, "4PCF(FGM)", scatfunc.convolve(xs_mat.data))
        result = {"xs": ddxs.integral}
        if prob:
            result.update(ddxs.E_prob)
        results[Ein] = result
    return get_results(results, prob)


def from_recoil_sct(xs_0K: pd.Series, Ein_grid: np.ndarray, M: float, T: float,
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
        xs_mat = XsMat.from_recoil(xs_0K, Ein, M, T, Eout, theta, pdos,
                                   model="sct")
        # Scattering function:
        scatfunc = ScatFunc.from_model(Ein, M, T, Eout, theta, pdos,
                                       model="sct")
        # DDxs
        ddxs = DDxs(Ein, T, M, "4PCF(SCT)", scatfunc.convolve(xs_mat.data))
        result = {"xs": ddxs.integral}
        if prob:
            result.update(ddxs.E_prob)
        results[Ein] = result
    return get_results(results, prob)


def from_recoil_pdos(xs_0K: pd.Series, Ein_grid: np.ndarray, M: float, T: float,
                     theta_diff: float, Eout_num: int, prob: bool, pdos: Pdos,
                     nphonon: int = None, decimal: float = 1.0e-6,
                     order_max: int = 5000, threshold: float = 0.0) -> pd.DataFrame:
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
    # Get common variables:
    xs_0K_values, xs_0K_E = xs_0K.values, xs_0K.index.values
    theta = np.arange(1, 180 + theta_diff, theta_diff)
    mu = np.sort(np.cos(np.deg2rad(theta)))
    T_arno = T * (1 + mu) / 2

    # Calculate the tau_n functions for scattering function:
    debye_waller_coeff_scatt = pdos.DebyeWallerCoeff(T)
    delta_beta_scatt = pdos.to_beta_grid(T).grid
    if nphonon:
        warnings.warn(
            "Is posible that the expansion order is not enough to get the correct results")
    else:
        alpha_max = get_alpha_from_Eout(Ein_grid[-1] * 1.1, Ein_grid[-1],
                                        M, T, mu.min())
        nphonon = get_expansion_order(alpha_max, debye_waller_coeff_scatt,
                                      decimal, order_max)
    tau_n_scatt = pdos.get_tau(T, nphonon, threshold, values=True)

    # Create xs_mat creation data:
    tau1, DebyeWallerCoeff, delta_beta = XsMat.get_pdos_variables(pdos, T_arno)

    # Create a list to hold the results
    if mu[0] == np.cos(np.pi):
        result = ddxs_clm_0K(Ein_grid, Eout_num, M, T,
                             tau_n_scatt, delta_beta_scatt,
                             debye_waller_coeff_scatt,
                             xs_0K_values, xs_0K_E, prob)
        start = 1
    else:
        result = []
        start = 0

    for i in range(start, len(theta)):
        # Create angle tau_n function:
        alpha_max = get_alpha_from_Eout(Ein_grid[-1] * 1.1, Ein_grid[-1],
                                        M, T_arno[i], mu.min())
        nphonon_row = get_expansion_order(alpha_max, DebyeWallerCoeff[i],
                                          decimal, order_max)
        tau_n_angle = tau_n_functions(tau1[i], delta_beta[i], nphonon_row,
                                      threshold)
        beta = default_abs_beta(T_arno[i])
        # Select the especific data for the next function:
        for Ein in Ein_grid:
            # Gen Eout grid:
            Eout = np.linspace(Ein * 0.9, Ein * 1.1, Eout_num)
            # Scattering function for selected angle and Ein:
            scattfunc_row = get_scatfunc_pdos_row(Ein, M, T, Eout, mu[i],
                                                  tau_n_scatt,
                                                  delta_beta_scatt,
                                                  debye_waller_coeff_scatt)

            # xs_mat row for selected angle and Ein:
            Ein_row = Ein_arno_row(Ein, Eout, mu[i], M)
            recoil_row = get_gressier_recoil(Ein_row, T_arno[i], M)
            alpha_recoil = recoil_row / (kb * T_arno[i])
            sab = Sab.from_tau(alpha_recoil, beta, tau_n_angle, delta_beta[i],
                               DebyeWallerCoeff[i]).full
            sab /= (kb * T_arno[i])
            xs_mat_row = np.zeros(Eout_num)
            update_xs_mat_pdos_recoil_row(xs_mat_row, sab.values,
                                          sab.columns.values, recoil_row,
                                          Ein_row, T_arno[i], xs_0K_values,
                                          xs_0K_E)
            ddxs_angle = xs_mat_row * scattfunc_row
            Ein_results = [mu[i], Ein, np.trapz(ddxs_angle, x=Eout)]

            # Get probability of upscattering and downscattering:
            if prob:
                mask_up, mask_down = Eout > Ein, Eout < Ein
                Ein_results.append(
                    np.trapz(ddxs_angle[mask_up], x=Eout[mask_up]))
                Ein_results.append(
                    np.trapz(ddxs_angle[mask_down], x=Eout[mask_down]))

            # Update results:
            result.append(Ein_results)
    if prob:
        df = pd.DataFrame(result,
                          columns=["mu", "Ein", "xs", "xs_up", "xs_down"])
        df_grouped = df.groupby("Ein")
        xs_db = df_grouped.apply(lambda group: pd.Series({
            'xs': np.trapz(group['xs'], x=group['mu']),
            'upscattering': np.trapz(group['xs_up'], x=group['mu']),
            'downscattering': np.trapz(group['xs_down'], x=group['mu'])
        }))
        xs_db['upscattering'] /= xs_db['xs']
        xs_db['downscattering'] /= xs_db['xs']
        xs_db['Ein=Eout'] = 1.0 - xs_db['upscattering'] - xs_db[
            'downscattering']
        xs_db = xs_db[["xs", "downscattering", "upscattering", "Ein=Eout"]]
    else:
        df = pd.DataFrame(result, columns=["mu", "Ein", "xs"])
        df_grouped = df.groupby("Ein")
        xs_db = df_grouped.apply(lambda group: pd.Series({
            'xs': np.trapz(group['xs'], x=group['mu'])}))
    return xs_db


def from_recoil(xs_0K: pd.Series, Ein_grid: np.ndarray, M: float, T: float,
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
    >>> from_recoil(xs_0K, Ein, M, T, model="fgm", Eout_num=1000)
                  xs  downscattering  upscattering  Ein=Eout
    Ein
    6.67  457.003682        0.846442      0.148459  0.005098

    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> from_recoil(xs_0K, Ein, M, T, pdos, model="sct", Eout_num=1000)
                  xs  downscattering  upscattering  Ein=Eout
    Ein
    6.67  438.888555        0.837399      0.157379  0.005222

    #>>> from_recoil(xs_0K, Ein, M, T, pdos, model="pdos", Eout_num=1000)
    #              xs  downscattering  upscattering  Ein=Eout
    #Ein
    #6.67  425.113084        0.845742      0.145287  0.008971
    """
    model = model.lower()
    if model == "pdos":
        return from_recoil_pdos(xs_0K, Ein_grid, M, T, theta_diff, Eout_num,
                                prob, *args, **kwargs)
    elif model == "sct":
        return from_recoil_sct(xs_0K, Ein_grid, M, T, theta_diff, Eout_num,
                               prob, *args)
    else:
        return from_recoil_fgm(xs_0K, Ein_grid, M, T, theta_diff, Eout_num,
                               prob)


def from_alpha0_pdos(xs_0K: pd.Series, Ein_grid: np.ndarray, M: float, T: float,
                     Eout_num: int, pdos: Pdos) -> pd.Series:
    xs_db = {}
    recoil = get_gressier_recoil(Ein_grid, T, M)
    alpha = recoil / (kb * T)
    debye_waller_coeff = pdos.DebyeWallerCoeff(T)
    delta_beta = pdos.to_beta_grid(T).grid
    nphonon = get_expansion_order(alpha, debye_waller_coeff, 1.0e-6, 5000)
    tau_n = pdos.get_tau(T, nphonon, 0.0, values=True)
    for i in range(len(Ein_grid)):
        Eout = np.linspace(Ein_grid[i] * 0.95, Ein_grid[i] * 1.05, Eout_num)
        beta = get_beta(Eout, Ein_grid[i], T)
        scatfunc = Sab.from_tau(alpha[i], beta,
                                tau_n, delta_beta, debye_waller_coeff).full
        Eout_calc = Ein_grid[i] + scatfunc.index.values * kb * T
        # xs_0K interpolation
        xs_0K_interp = reshape_differential(xs_0K, Eout_calc + recoil[i])
        # XsMat
        dxs = scatfunc * xs_0K_interp
        dxs.index = pd.Index(Eout_calc, name="Eout")
        xs_db[Ein_grid[i]] = integrate(dxs) / (kb * T)
    return pd.Series(xs_db, name="xs")
