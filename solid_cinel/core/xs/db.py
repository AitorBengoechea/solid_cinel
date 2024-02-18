import numpy as np
import pandas as pd
from scipy.constants import physical_constants as const
from solid_cinel.core.xs import XsMat, ScatFunc, DDxs, Pdos, Sab
from solid_cinel.core.scattering_function.alpha import get_alphaFromEout, get_expansionOrder, get_gressierRecoil
from solid_cinel.core.scattering_function.beta import get_beta
from solid_cinel.core.scattering_function import get_ScatFuncClmRow
from solid_cinel.core.material.vibration.tau import save_tau, get_tauNfunc, get_tauNbeta
from solid_cinel.core.xs.xs_mat import update_XsMatClmRecoilRow, default_absBeta, EinArnoRow
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
        Cross section at each energy in EinGrid for the selected temperature
        and the downscattering, upscattering and Ein=Eout probabilities if prob
        is True.
    """
    xsDb = pd.DataFrame(results).T.sort_index()
    xsDb.index.name = "Ein"
    columnsOrder = ["xs", "downscattering", "upscattering", "Ein=Eout"]
    return xsDb[columnsOrder] if prob else xsDb

def from_fgm(xs0K: pd.Series, EinGrid: np.ndarray, M: float, T: float,
             thetaDiff: float, EoutNum: int, prob: bool) -> pd.DataFrame:
    """
    Generate doppler broadening cross section using 4PCF with SIGMA1 doppler
    broadened XsMat and scattering function based on FGM.

    Parameters
    ----------
    xs0K: pd.Series
        Cross section at 0K
    EinGrid: np.ndarray
        Energies to calculate the cross section in eV
    M: float
        Mass of the target in amu
    T: float
        Temperature in K
    thetaDiff: float
        Angle step in degrees
    EoutNum: int
        Number of energies to calculate the cross section
    prob: bool
        If True, apart from the cross section, the probability of each energy
        for downscattering, upscattering and Ein=Eout is calculated.

    Returns
    -------
    pd.DataFrame
        Cross section at each energy in EinGrid for the selected temperature
        and the downscattering, upscattering and Ein=Eout probabilities if prob
        is True.
    """
    results = {}
    theta = np.arange(1, 180 + thetaDiff, thetaDiff)
    for Ein in EinGrid:
        Eout = np.linspace(Ein * 0.95, Ein * 1.05, EoutNum)
        # XsMat
        xs_mat = XsMat.from_model(xs0K, Ein, M, T, Eout, theta)
        # Scattering function:
        scatfunc = ScatFunc.from_model(Ein, M, T, Eout, theta, model="fgm")
        # DDxs
        ddxs = DDxs(Ein, T, M, "4PCF(FGM)", scatfunc.convolve(xs_mat.data))
        result = {"xs": ddxs.integral}
        if prob:
            result.update(ddxs.Eprob)
        results[Ein] = result
    return get_results(results, prob)


def from_sct(xs0K: pd.Series, EinGrid: np.ndarray, M: float, T: float,
             thetaDiff: float, EoutNum: int, prob: bool, pdos: Pdos) -> pd.DataFrame:
    """
    Generate doppler broadening cross section using 4PCF with SIGMA1 doppler
    broadened XsMat and scattering function based on SCT.

    Parameters
    ----------
    xs0K: pd.Series
        Cross section at 0K
    EinGrid: np.ndarray
        Energies to calculate the cross section in eV
    M: float
        Mass of the target in amu
    T: float
        Temperature in K
    thetaDiff: float
        Angle step in degrees
    EoutNum: int
        Number of energies to calculate the cross section
    pdos: Pdos
        Phonon density of states of the target
    prob: bool
        If True, apart from the cross section, the probability of each energy
        for downscattering, upscattering and Ein=Eout is calculated.

    Returns
    -------
    pd.DataFrame
        Cross section at each energy in EinGrid for the selected temperature
        and the downscattering, upscattering and Ein=Eout probabilities if prob
        is True.
    """
    results = {}
    theta = np.arange(1, 180 + thetaDiff, thetaDiff)
    for Ein in EinGrid:
        Eout = np.linspace(Ein * 0.95, Ein * 1.05, EoutNum)
        # XsMat
        xs_mat = XsMat.from_model(xs0K, Ein, M, T, Eout, theta)
        # Scattering function:
        scatfunc = ScatFunc.from_model(Ein, M, T, Eout, theta, pdos, model="sct")
        # DDxs
        ddxs = DDxs(Ein, T, M, "4PCF(SCT)", scatfunc.convolve(xs_mat.data))
        result = {"xs": ddxs.integral}
        if prob:
            result.update(ddxs.Eprob)
        results[Ein] = result
    return get_results(results, prob)


def from_pdos(xs0K: pd.Series, EinGrid: np.ndarray, M: float, T: float,
              thetaDiff: float, EoutNum: int, prob: bool, pdos: Pdos,
              nphonon: int = None, decimal: float = 1.0e-6,
              orderMax: int = 5000, threshold: float = 0.0,
              tauToFile: bool = False, binary: bool = False) -> pd.DataFrame:
    """
    Generate doppler broadening cross section using 4PCF with SIGMA1 doppler
    broadened XsMat and scattering function based on Phonon Expansion.

    Parameters
    ----------
    xs0K: pd.Series
        Cross section at 0K
    EinGrid: np.ndarray
        Energies to calculate the cross section in eV
    M: float
        Mass of the target in amu
    T: float
        Temperature in K
    thetaDiff: float
        Angle step in degrees
    EoutNum: int
        Number of energies to calculate the cross section
    pdos: Pdos
        Phonon density of states of the target
    prob: bool
        If True, apart from the cross section, the probability of each energy
        for downscattering, upscattering and Ein=Eout is calculated.
    nphonon: int
        Expansion order. If None, the order is calculated automatically.
    decimal: float
        Precision of the tauN functions
    orderMax: int
        Maximum expansion order
    threshold: float
        Threshold to calculate the tauN functions
    tauToFile: bool
        If True, the tauN functions are saved to a file
    binary: bool
        If True, the tauN functions are saved in binary format

    Returns
    -------
    pd.DataFrame
        Cross section at each energy in EinGrid for the selected temperature
        and the downscattering, upscattering and Ein=Eout probabilities if prob
        is True.
    """
    results = {}
    theta = np.arange(1, 180 + thetaDiff, thetaDiff)
    mu = np.cos(np.deg2rad(theta))
    muMin = mu.min()

    # Calculate the tauN functions:
    DebyeWallerCoeff = pdos.DebyeWallerCoeff(T)
    beta = pdos.beta_grid(T).data.index.values
    if nphonon:
        warnings.warn(
            "Is posible that the expansion order is not enough to get the correct results")
    else:
        alphaMax = get_alphaFromEout(1.05 * EinGrid[-1], EinGrid[-1], M, T, muMin)
        nphonon = get_expansionOrder(alphaMax, DebyeWallerCoeff, decimal, orderMax)
    tauN = pdos.tauN(T, nphonon, threshold, values=True)
    save_tau(tauN, nphonon, T, tauToFile, binary)
    tauNbeta = get_tauNbeta(beta, tauN.shape[1])
    # start the loop
    for Ein in EinGrid:
        Eout = np.linspace(Ein * 0.95, Ein * 1.05, EoutNum)
        # XsMat
        xs_mat = XsMat.from_model(xs0K, Ein, M, T, Eout, theta)
        # Minimize the expansion order for each energy:
        alphaMax = get_alphaFromEout(1.05 * Ein, Ein, M, T, muMin)
        minNphonon = get_expansionOrder(alphaMax, DebyeWallerCoeff, decimal, orderMax)
        # Scattering function:
        scatfunc = ScatFunc.from_tau(Ein, M, T, Eout, mu, tauN[:minNphonon],
                                     tauNbeta, DebyeWallerCoeff)
        # DDxs
        ddxs = DDxs(Ein, T, M, "4PCF(CLM)", scatfunc.convolve(xs_mat.data))
        result = {"xs": ddxs.integral}
        if prob:
            result.update(ddxs.Eprob)
        results[Ein] = result
    return get_results(results, prob)


def from_model(xs0K: pd.Series, EinGrid: np.ndarray, M: float, T: float,
               *args, thetaDiff: float = 1.0, EoutNum: int = 3000,
               model: str = "fgm", prob: bool = True, **kwargs) -> pd.DataFrame:
    """
    Generate doppler broadening cross section using 4PCF with SIGMA1 doppler
    broadened XsMat and scattering function based on the selected model.

    Parameters
    ----------
    xs0K: pd.Series
        Cross section at 0K
    EinGrid: np.ndarray
        Energies to calculate the cross section in eV
    M: float
        Mass of the target in amu
    T: float
        Temperature in K
    thetaDiff: float
        Angle step in degrees
    EoutNum: int
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
        Precision of the tauN functions. Default: 1.0e-6
    n_orderMax: int
        Maximum expansion order. Default: 5000
    threshold: float
        Threshold to calculate the tauN functions. Default: 0.0
    tauToFile: bool
        If True, the tauN functions are saved to a file. Default: False
    binary: bool
        If True, the tauN functions are saved in binary format. Default: False

    Returns
    -------
    pd.DataFrame
        Cross section at each energy in EinGrid for the selected temperature
        and the downscattering, upscattering and Ein=Eout probabilities if prob
        is True.

    Examples
    --------
    # 0K xs data for U238:
    >>> import os
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("db.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
    >>> os.chdir(wd)

    # Generate DDXS test variables:
    >>> T = 300
    >>> M = 238.05077040419212
    >>> Ein = np.array([6.67])
    >>> from_model(xs0K, Ein, M, T, model="fgm", EoutNum=1000)
                  xs  downscattering  upscattering  Ein=Eout
    Ein
    6.67  456.375935        0.846605      0.148305   0.00509

    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> from_model(xs0K, Ein, M, T, pdos, model="sct", EoutNum=1000)
                  xs  downscattering  upscattering  Ein=Eout
    Ein
    6.67  453.228141        0.843325      0.151599  0.005077

    >>> from_model(xs0K, Ein, M, T, pdos, model="pdos", EoutNum=1000)
                  xs  downscattering  upscattering  Ein=Eout
    Ein
    6.67  444.117576        0.852297      0.143335  0.004368
    """
    model = model.lower()
    if model == "pdos":
        return from_pdos(xs0K, EinGrid, M, T, thetaDiff, EoutNum, prob,
                         *args, **kwargs)
    elif model == "sct":
        return from_sct(xs0K, EinGrid, M, T, thetaDiff, EoutNum, prob,
                        *args)
    else:
        return from_fgm(xs0K, EinGrid, M, T, thetaDiff, EoutNum, prob)


def from_recoilFgm(xs0K: pd.Series, EinGrid: np.ndarray, M: float, T: float,
                    thetaDiff: float, EoutNum: int,
                    prob: bool) -> pd.DataFrame:
    """
    Generate doppler broadening cross section using 4PCF with SIGMA1 doppler
    broadened XsMat and scattering function based on FGM.

    Parameters
    ----------
    xs0K: pd.Series
        Cross section at 0K
    EinGrid: np.ndarray
        Energies to calculate the cross section in eV
    M: float
        Mass of the target in amu
    T: float
        Temperature in K
    thetaDiff: float
        Angle step in degrees
    EoutNum: int
        Number of energies to calculate the cross section
    prob: bool
        If True, apart from the cross section, the probability of each energy
        for downscattering, upscattering and Ein=Eout is calculated.

    Returns
    -------
    pd.DataFrame
        Cross section at each energy in EinGrid for the selected temperature
        and the downscattering, upscattering and Ein=Eout probabilities if prob
        is True.
    """
    results = {}
    theta = np.arange(1, 180 + thetaDiff, thetaDiff)
    for Ein in EinGrid:
        Eout = np.linspace(Ein * 0.95, Ein * 1.05, EoutNum)
        # XsMat
        xs_mat = XsMat.from_recoil(xs0K, Ein, M, T, Eout, theta)
        # Scattering function:
        scatfunc = ScatFunc.from_model(Ein, M, T, Eout, theta, model="fgm")
        # DDxs
        ddxs = DDxs(Ein, T, M, "4PCF(FGM)", scatfunc.convolve(xs_mat.data))
        result = {"xs": ddxs.integral}
        if prob:
            result.update(ddxs.Eprob)
        results[Ein] = result
    return get_results(results, prob)


def from_recoilSct(xs0K: pd.Series, EinGrid: np.ndarray, M: float, T: float,
                    thetaDiff: float, EoutNum: int, prob: bool,
                    pdos: Pdos) -> pd.DataFrame:
    """
    Generate doppler broadening cross section using 4PCF with SIGMA1 doppler
    broadened XsMat and scattering function based on SCT.

    Parameters
    ----------
    xs0K: pd.Series
        Cross section at 0K
    EinGrid: np.ndarray
        Energies to calculate the cross section in eV
    M: float
        Mass of the target in amu
    T: float
        Temperature in K
    thetaDiff: float
        Angle step in degrees
    EoutNum: int
        Number of energies to calculate the cross section
    pdos: Pdos
        Phonon density of states of the target
    prob: bool
        If True, apart from the cross section, the probability of each energy
        for downscattering, upscattering and Ein=Eout is calculated.

    Returns
    -------
    pd.DataFrame
        Cross section at each energy in EinGrid for the selected temperature
        and the downscattering, upscattering and Ein=Eout probabilities if prob
        is True.
    """
    results = {}
    theta = np.arange(1, 180 + thetaDiff, thetaDiff)
    for Ein in EinGrid:
        Eout = np.linspace(Ein * 0.95, Ein * 1.05, EoutNum)
        # XsMat
        xs_mat = XsMat.from_recoil(xs0K, Ein, M, T, Eout, theta, pdos,
                                   model="sct")
        # Scattering function:
        scatfunc = ScatFunc.from_model(Ein, M, T, Eout, theta, pdos,
                                       model="sct")
        # DDxs
        ddxs = DDxs(Ein, T, M, "4PCF(SCT)", scatfunc.convolve(xs_mat.data))
        result = {"xs": ddxs.integral}
        if prob:
            result.update(ddxs.Eprob)
        results[Ein] = result
    return get_results(results, prob)


def ddxsClm0K(EinGrid: np.ndarray, num_Eout: int, M: float, T: float,
              tauNscatt: np.ndarray, tauNscattBeta: float, DebyeWallerCoeffScatt: float,
              xs0KValues: np.ndarray, xs0KE: np.ndarray, prob: bool) -> list:
    """
    Compute the ddxs for 180 degree in clm model using

    Parameters
    ----------
    EinGrid : np.ndarray
        Incoming energy grid.
    num_Eout : int
        Number of energy grid for outgoing energy grid.
    M : float
        Mass of the target in amu.
    T : float
        Temperature in kelvin.
    tauNscatt : np.ndarray
        Tau(-beta) function for n expansion for calculation of the scattering function.
    delta_betaScatt : float
        Interval of beta for the scattering function.
    DebyeWallerCoeffScatt : float
        Debye-Waller coefficient for the scattering function.
    xs0KValues : np.ndarray
        Cross section values at 0K.
    xs0KE : np.ndarray
        Cross section energy grid.
    prob : bool
        If True, return probability of upscattering and downscattering.

    Returns
    -------
    list
        Cross section or cross section and probability of upscattering and
        downscattering.

    # 0K xs data for U238:
    >>> import os
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("db.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
    >>> os.chdir(wd)

    # Generate DDXS test variables:
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> EinGrid = np.array([2.0, 6.67, 36.6])
    >>> num_Eout = 1000
    >>> xs0KValues, xs0KE = xs0K.values, xs0K.index.values
    >>> from solid_cinel.core.material.vibration.pdos import Pdos
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> DebyeWallerCoeffScatt = pdos.DebyeWallerCoeff(T)
    >>> betaScatt = pdos.beta_grid(T).data.index.values
    >>> tauNscatt = pdos.tauN(T, 10, 0.0, values=True)
    >>> tauNbeta = get_tauNbeta(betaScatt, tauNscatt.shape[1])
    >>> tauNscattBeta = get_tauNbeta(betaScatt, tauNscatt.shape[1])
    >>> result = ddxsClm0K(EinGrid, num_Eout, M, T, tauNscatt, tauNscattBeta, DebyeWallerCoeffScatt, xs0KValues, xs0KE, True)
    >>> pd.DataFrame(result, columns=["mu", "Ein", "xs", "xs_up", "xs_down"]).round(6)
        mu    Ein   xs  xs_up  xs_down
    0 -1.0   2.00  0.0    0.0      0.0
    1 -1.0   6.67  0.0    0.0      0.0
    2 -1.0  36.60  0.0    0.0      0.0
    """
    result = []
    for Ein in EinGrid:
        # Gen Eout grid:
        Eout = np.linspace(Ein * 0.9, Ein * 1.1, num_Eout)
        EinRow = EinArnoRow(Ein, Eout, -1.0, M)
        scattFuncRow = get_ScatFuncClmRow(Ein, M, T, Eout, -1.0, tauNscatt,
                                          tauNscattBeta, DebyeWallerCoeffScatt)
        rowResults = scattFuncRow * np.interp(EinRow, xs0KE, xs0KValues)
        EinResults = [-1.0, Ein, np.trapz(rowResults, x=Eout)]

        # Get probability of upscattering and downscattering:
        if prob:
            mask_up, mask_down = Eout > Ein, Eout < Ein
            EinResults.append(np.trapz(rowResults[mask_up], x=Eout[mask_up]))
            EinResults.append(np.trapz(rowResults[mask_down], x=Eout[mask_down]))

        # Update results:
        result.append(EinResults)
    return result


def from_recoilClm(xs0K: pd.Series, EinGrid: np.ndarray, M: float, T: float,
                     thetaDiff: float, EoutNum: int, prob: bool, pdos: Pdos,
                     nphonon: int = None, decimal: float = 1.0e-6,
                     orderMax: int = 5000, threshold: float = 0.0) -> pd.DataFrame:
    """
    Generate doppler broadening cross section using 4PCF with SIGMA1 doppler
    broadened XsMat and scattering function based on Phonon Expansion.

    Parameters
    ----------
    xs0K: pd.Series
        Cross section at 0K
    EinGrid: np.ndarray
        Energies to calculate the cross section in eV
    M: float
        Mass of the target in amu
    T: float
        Temperature in K
    thetaDiff: float
        Angle step in degrees
    EoutNum: int
        Number of energies to calculate the cross section
    pdos: Pdos
        Phonon density of states of the target
    prob: bool
        If True, apart from the cross section, the probability of each energy
        for downscattering, upscattering and Ein=Eout is calculated.
    nphonon: int
        Expansion order. If None, the order is calculated automatically.
    decimal: float
        Precision of the tauN functions
    n_orderMax: int
        Maximum expansion order
    threshold: float
        Threshold to calculate the tauN functions
    tauToFile: bool
        If True, the tauN functions are saved to a file
    binary: bool
        If True, the tauN functions are saved in binary format

    Returns
    -------
    pd.DataFrame
        Cross section at each energy in EinGrid for the selected temperature
        and the downscattering, upscattering and Ein=Eout probabilities if prob
        is True.
    """
    # Get common variables:
    xs0KValues, xs0KE = xs0K.values, xs0K.index.values
    theta = np.arange(1, 180 + thetaDiff, thetaDiff)
    mu = np.sort(np.cos(np.deg2rad(theta)))
    muMin = mu.min()
    Tarno = T * (1 + mu) / 2

    # Calculate the tauN functions for scattering function:
    DebyeWallerCoeffScatt = pdos.DebyeWallerCoeff(T)
    betaScatt = pdos.beta_grid(T).data.index.values
    if nphonon:
        warnings.warn(
            "Is posible that the expansion order is not enough to get the correct results")
    else:
        alphaMax = get_alphaFromEout(EinGrid[-1] * 1.1, EinGrid[-1], M, T, muMin)
        nphonon = get_expansionOrder(alphaMax, DebyeWallerCoeffScatt, decimal, orderMax)
    tauNscatt = pdos.tauN(T, nphonon, threshold, values=True)
    tauNscattBeta = get_tauNbeta(betaScatt, tauNscatt.shape[1])

    # Create xs_mat creation data:
    tau1, DebyeWallerCoeff, beta_tau1 = XsMat.get_pdos_variables(pdos, Tarno)

    # Create a list to hold the results
    if mu[0] == np.cos(np.pi):
        result = ddxsClm0K(EinGrid, EoutNum, M, T, tauNscatt, tauNscattBeta,
                           DebyeWallerCoeffScatt, xs0KValues, xs0KE, prob)
        start = 1
    else:
        result = []
        start = 0

    for i in range(start, len(theta)):
        # Create angle tauN function:
        alphaMax = get_alphaFromEout(EinGrid[-1] * 1.1, EinGrid[-1], M,
                                     Tarno[i], muMin)
        minNphonon = get_expansionOrder(alphaMax, DebyeWallerCoeff[i], decimal,
                                         orderMax)
        tauNangle = get_tauNfunc(tau1[i], beta_tau1[i], minNphonon, threshold)
        beta_tauNangle = get_tauNbeta(beta_tau1[i], tauNangle.shape[1])
        beta = default_absBeta(Tarno[i])
        # Select the especific data for the next function:
        for Ein in EinGrid:
            # Gen Eout grid:
            Eout = np.linspace(Ein * 0.9, Ein * 1.1, EoutNum)
            # Scattering function for selected angle and Ein:
            scattFuncRow = get_ScatFuncClmRow(Ein, M, T, Eout, mu[i], tauNscatt,
                                              tauNscattBeta, DebyeWallerCoeffScatt)

            # xs_mat row for selected angle and Ein:
            EinRow = EinArnoRow(Ein, Eout, mu[i], M)
            recoilRow = get_gressierRecoil(EinRow, Tarno[i], M)
            alphaRecoil = recoilRow / (kb * Tarno[i])
            sab = Sab.from_tau(alphaRecoil, beta, tauNangle, beta_tauNangle,
                               DebyeWallerCoeff[i]).full
            sab /= (kb * Tarno[i])
            xsMatRow = np.zeros(EoutNum)
            update_XsMatClmRecoilRow(xsMatRow, sab.values, sab.columns.values,
                                     recoilRow, EinRow, Tarno[i], xs0KValues, xs0KE)
            ddxsAngle = xsMatRow * scattFuncRow
            EinResults = [mu[i], Ein, np.trapz(ddxsAngle, x=Eout)]

            # Get probability of upscattering and downscattering:
            if prob:
                mask_up, mask_down = Eout > Ein, Eout < Ein
                EinResults.append(
                    np.trapz(ddxsAngle[mask_up], x=Eout[mask_up]))
                EinResults.append(
                    np.trapz(ddxsAngle[mask_down], x=Eout[mask_down]))

            # Update results:
            result.append(EinResults)
    if prob:
        dfGrouped = pd.DataFrame(result, columns=["mu", "Ein", "xs", "xs_up", "xs_down"]).groupby("Ein")
        xsDb = dfGrouped.apply(lambda group: pd.Series({
            'xs': np.trapz(group['xs'], x=group['mu']),
            'upscattering': np.trapz(group['xs_up'], x=group['mu']),
            'downscattering': np.trapz(group['xs_down'], x=group['mu'])
        }))
        xsDb['upscattering'] /= xsDb['xs']
        xsDb['downscattering'] /= xsDb['xs']
        xsDb['Ein=Eout'] = 1.0 - xsDb['upscattering'] - xsDb['downscattering']
        xsDb = xsDb[["xs", "downscattering", "upscattering", "Ein=Eout"]]
    else:
        dfGrouped = pd.DataFrame(result, columns=["mu", "Ein", "xs"]).groupby("Ein")
        xsDb = dfGrouped.apply(lambda group: pd.Series({'xs': np.trapz(group['xs'], x=group['mu'])}))
    return xsDb


def from_recoil(xs0K: pd.Series, EinGrid: np.ndarray, M: float, T: float,
                *args, thetaDiff: float = 1.0, EoutNum: int = 3000,
                model: str = "fgm", prob: bool = True, **kwargs) -> pd.DataFrame:
    """
    Generate doppler broadening cross section using 4PCF with SIGMA1 doppler
    broadened XsMat and scattering function based on the selected model.

    Parameters
    ----------
    xs0K: pd.Series
        Cross section at 0K
    EinGrid: np.ndarray
        Energies to calculate the cross section in eV
    M: float
        Mass of the target in amu
    T: float
        Temperature in K
    thetaDiff: float
        Angle step in degrees
    EoutNum: int
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
        Precision of the tauN functions. Default: 1.0e-6
    n_orderMax: int
        Maximum expansion order. Default: 5000
    threshold: float
        Threshold to calculate the tauN functions. Default: 0.0
    tauToFile: bool
        If True, the tauN functions are saved to a file. Default: False
    binary: bool
        If True, the tauN functions are saved in binary format. Default: False

    Returns
    -------
    pd.DataFrame
        Cross section at each energy in EinGrid for the selected temperature
        and the downscattering, upscattering and Ein=Eout probabilities if prob
        is True.

    Examples
    --------
    # 0K xs data for U238:
    >>> import os
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("db.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
    >>> os.chdir(wd)

    # Generate DDXS test variables:
    >>> T = 300
    >>> M = 238.05077040419212
    >>> Ein = np.array([6.67])
    >>> from_recoil(xs0K, Ein, M, T, model="fgm", EoutNum=1000)
                  xs  downscattering  upscattering  Ein=Eout
    Ein
    6.67  457.003682        0.846442      0.148459  0.005098

    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> from_recoil(xs0K, Ein, M, T, pdos, model="sct", EoutNum=1000)
                  xs  downscattering  upscattering  Ein=Eout
    Ein
    6.67  438.888555        0.837399      0.157379  0.005222

    #>>> from_recoil(xs0K, Ein, M, T, pdos, model="pdos", EoutNum=1000)
    #              xs  downscattering  upscattering  Ein=Eout
    #Ein
    #6.67  425.113084        0.845742      0.145287  0.008971
    """
    model = model.lower()
    if model == "pdos":
        return from_recoilClm(xs0K, EinGrid, M, T, thetaDiff, EoutNum, prob,
                              *args, **kwargs)
    elif model == "sct":
        return from_recoilSct(xs0K, EinGrid, M, T, thetaDiff, EoutNum, prob,
                              *args)
    else:
        return from_recoilFgm(xs0K, EinGrid, M, T, thetaDiff, EoutNum, prob)


def from_alpha0Clm(xs0K: pd.Series, EinGrid: np.ndarray, M: float, T: float,
                   EoutNum: int, pdos: Pdos) -> pd.Series:
    xsDb = {}
    recoil = get_gressierRecoil(EinGrid, T, M)
    alpha = recoil / (kb * T)
    DebyeWallerCoeff = pdos.DebyeWallerCoeff(T)
    nphonon = get_expansionOrder(alpha, DebyeWallerCoeff, 1.0e-6, 5000)
    tau1 = pdos.beta_grid(T).data.index.values
    tauN = pdos.get_tau(T, nphonon, 0.0, values=True)
    tauNbeta = get_tauNbeta(tau1, tauN.shape[1])
    for i in range(len(EinGrid)):
        Eout = np.linspace(EinGrid[i] * 0.95, EinGrid[i] * 1.05, EoutNum)
        beta = get_beta(Eout, EinGrid[i], T)
        scatfunc = Sab.from_tau(alpha[i], beta, tauN, tauNbeta, DebyeWallerCoeff).full
        EoutCalc = EinGrid[i] + scatfunc.index.values * kb * T
        # xs0K interpolation
        xs0Kinterp = reshape_differential(xs0K, EoutCalc + recoil[i])
        # XsMat
        dxs = scatfunc * xs0Kinterp
        dxs.index = pd.Index(EoutCalc, name="Eout")
        xsDb[EinGrid[i]] = integrate(dxs) / (kb * T)
    return pd.Series(xsDb, name="xs")
