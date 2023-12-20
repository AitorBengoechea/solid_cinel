

import numpy as np
import numba as nb
from scipy.constants import physical_constants as const
from solid_cinel.core.material.scattering_function.beta import get_beta
from solid_cinel.core.material.scattering_function.alpha import get_alpha_mat, get_alpha_from_Eout


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

@nb.jit(nopython=True, cache=True, nogil=True)
def get_sab_pdos(alpha: np.ndarray, beta: np.ndarray,
                 tau_n: np.ndarray, tau_n_beta: np.ndarray,
                 DebyeWallerCoeff: float) -> np.ndarray:
    """
    Generate the scattering function from a S(alpha, -beta) table based on
    the phonon expansion model using a single angle.

    Parameters
    ----------
    alpha : 'np.ndarray', (Z, N)
        alpha grid values.
    beta : 'np.ndarray', (N,)
        beta grid values.
    nphonon : 'int', optional
        Phonon expansion order.
    tau_n : 'np.ndarray', (M, T)
        all tau n functions in one array.
    tau_n_beta : 'np.ndarray', (M,)
        Space between beta grid points of tau n functions.
    DebyeWallerCoeff : 'float'
        Debye Waller Coefficient in LEAPR formalism.

    Returns
    -------
    S_diag : 'np.ndarray', (Z, N)
        S(alpha, -beta) values for the alpha and beta combinations.

    Examples
    --------
    >>> import pandas as pd
    >>> from solid_cinel.core.material.vibration.pdos import Pdos
    >>> Ein = 7.2
    >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
    >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> mu = np.cos(np.deg2rad([120]))
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> tau_n, delta_beta, debye_waller_coeff = pdos.get_clm_param(T, nphonon=1000, threshold=1.0e-14)
    >>> tau_n_beta = np.arange(tau_n.shape[1]) * delta_beta
    >>> beta = get_beta(Eout, Ein, T)
    >>> alpha_mat = get_alpha_mat(beta * kb * T + Ein, Ein, T, M, mu)
    >>> sab_values = get_sab_pdos(alpha_mat, beta, tau_n, tau_n_beta, debye_waller_coeff)
    >>> pd.DataFrame(sab_values, index=[120], columns=beta).T.iloc[::100].round(6)
                   120
    0.000000  0.210641
    0.399957  0.247226
    0.802224  0.269120
    1.204491  0.271591
    1.603331  0.254529
    2.000979  0.221922
    2.403246  0.179661
    2.805512  0.135357
    3.526166  0.068344
    4.330699  0.024589
    5.135233  0.006799
    """
    # Zero phonon expansion:
    iter_sum = np.log(alpha * DebyeWallerCoeff)
    alpha_mul = np.exp(- alpha * DebyeWallerCoeff + iter_sum)
    S_diag = alpha_mul * np.interp(beta, tau_n_beta, tau_n[0])

    # Higher phonon expansion (nphonon >= 1):
    for n in range(1, tau_n.shape[0]):
        # Compute S(alpha, -beta) for tau_n reshape
        iter_sum += np.log(alpha * DebyeWallerCoeff / (n + 1))
        alpha_mul = np.exp(- alpha * DebyeWallerCoeff + iter_sum)
        S_diag += alpha_mul * np.interp(beta, tau_n_beta, tau_n[n])
    return S_diag


@nb.jit(nopython=True, nogil=True, cache=True)
def normalization_factor(Eout_calc: np.ndarray, Ein: float, T: float, M: float) -> np.ndarray:
    M_div_m = M / m
    aws = ((M_div_m + 1) / M_div_m) ** 2
    two_kb_T = 2 * kb * T
    return aws * np.sqrt(Eout_calc / Ein) / two_kb_T


@nb.jit(nopython=True, nogil=True, cache=True)
def scatfunc_values_alpha_vec(Sab_mat: np.ndarray, beta: np.ndarray, Ein: float,
                        T: float, M: float) -> (np.ndarray, np.ndarray):
    """
    Generate the scattering function values from a S(alpha, -beta) table based on
    the phonon expansion model for a single angle

    Parameters
    ----------
    Sab_mat : 'np.ndarray', (N,)
        S(alpha, -beta) matrix values.
    beta: 'np.ndarray', (N,)
        Minus beta grid values.
    Ein : 'float'
        Incident energy in eV.
    T : 'float'
        Temperature in K.
    M : 'float'
        Mass of the target nucleus in amu.

    Returns
    -------
    'np.ndarray', (N, 2)
        Scattering function values for a single angle for tau_n expansion.

    Examples
    --------
    >>> import pandas as pd
    >>> from solid_cinel.core.material.vibration.pdos import Pdos
    >>> Ein = 7.2
    >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
    >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> mu = np.cos(np.deg2rad([120]))
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> tau_n, delta_beta, debye_waller_coeff = pdos.get_clm_param(T, nphonon=1000, threshold=1.0e-14)
    >>> tau_n_beta = np.arange(tau_n.shape[1]) * delta_beta
    >>> beta = get_beta(Eout, Ein, T)
    >>> from solid_cinel.core.material.scattering_function.alpha import get_alpha_from_Eout
    >>> alpha_mat = get_alpha_from_Eout(beta * kb * T + Ein, Ein, T, M, mu)
    >>> sab_values = get_sab_pdos(alpha_mat, beta, tau_n, tau_n_beta, debye_waller_coeff)
    >>> Eout_calc, scatfunc_values = scatfunc_values_alpha_vec(sab_values, beta, Ein, T, M)
    >>> pd.Series(scatfunc_values, index=Eout_calc).iloc[::200].round(6)
    6.755400  0.036933
    6.894059  0.381007
    6.991813  1.027909
    7.060847  1.470584
    7.129778  1.569325
    7.199108  1.238804
    7.268142  0.716554
    7.337073  0.307374
    7.406107  0.098213
    7.501782  0.012632
    7.640440  0.000258
    dtype: float64
    """
    # Scattering function values calculation:
    ScatFunc_values = np.concatenate((Sab_mat[::-1], Sab_mat[1::] * np.exp(-beta[1:])))

    # Eout calculation
    Eout_calc = Ein + np.concatenate((-beta[::-1], beta[1::])) * kb * T

    # Ensure the Eout values are positive:
    positive_mask = Eout_calc > 0
    Eout_calc = Eout_calc[positive_mask]

    # Normalization constant
    norm = normalization_factor(Eout_calc, Ein, T, M)

    return Eout_calc, ScatFunc_values[positive_mask] * norm

@nb.jit(nopython=True, cache=True, nogil=True)
def scatfunc_values_alpha_mat(Sab_values: np.ndarray, beta: np.ndarray, Ein: float,
                              T: float, M: float) -> (np.ndarray, np.ndarray):
    """
    Generate the scattering function from a S(alpha, -beta) table based on
    the phonon expansion model. The scattering function is calculated for all
    the angles and the outgoing energy grid is calculated based on the beta
    grid.

    Parameters
    ----------
    Sab_values: 'np.ndarray', (Z, N)
        S(alpha, -beta) for the selected alpha and beta.
    beta: 'np.ndarray', (N,)
        beta grid values.
    Ein: 'float'
        Incident energy in eV.
    T: 'float'
        Temperature in K.
    M: 'float'
        Mass of the target in amu.

    Returns
    -------
    Eout_calc: 'np.ndarray', (N,)
        Outgoing energy grid in eV.
    ScatFunc_values: 'np.ndarray', (Z, N)
        Scattering function values for all the angles and Eout calculation.

    Examples
    --------
    >>> import pandas as pd
    >>> from solid_cinel.core.material.vibration.pdos import Pdos
    >>> Ein = 7.2
    >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
    >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> mu = np.cos(np.deg2rad([120]))
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> tau_n, delta_beta, debye_waller_coeff = pdos.get_clm_param(T, nphonon=1000, threshold=1.0e-14)
    >>> tau_n_beta = np.arange(tau_n.shape[1]) * delta_beta
    >>> beta = get_beta(Eout, Ein, T)
    >>> alpha_mat = get_alpha_mat(beta * kb * T + Ein, Ein, T, M, mu)
    >>> sab_values = get_sab_pdos(alpha_mat, beta, tau_n, tau_n_beta, debye_waller_coeff)
    >>> Eout_calc, scatfunc_values = scatfunc_values_alpha_mat(sab_values, beta, Ein, T, M)
    >>> pd.DataFrame(scatfunc_values, index=[120], columns=Eout_calc).T.iloc[::200].round(6)
                   120
    6.755400  0.036933
    6.894059  0.381007
    6.991813  1.027909
    7.060847  1.470584
    7.129778  1.569325
    7.199108  1.238804
    7.268142  0.716554
    7.337073  0.307374
    7.406107  0.098213
    7.501782  0.012632
    7.640440  0.000258
    """
    ScatFunc_values = np.concatenate(
        (Sab_values[::, ::-1], Sab_values[::, 1:] * np.exp(-beta[1:])), axis=1)
    # Eout calculation
    Eout_calc = np.sort(Ein + np.concatenate((-beta[::-1], beta[1::])) * kb * T)

    # Ensure the Eout values are positive:
    positive_mask = Eout_calc > 0
    Eout_calc = Eout_calc[positive_mask]

    # Normalization constant
    norm = normalization_factor(Eout_calc, Ein, T, M)

    return Eout_calc, ScatFunc_values[::, positive_mask] * norm


@nb.jit(nopython=True, nogil=True, cache=True)
def get_scatfunc_pdos(Ein: float, M: float, T: float, Eout: np.ndarray,
                      mu: np.ndarray, tau_n: np.ndarray, delta_beta: float,
                      DebyeWallerCoeff: float) -> np.ndarray:
    """
    Generate the scattering function from a S(alpha, -beta) table based on
    the phonon expansion model.

    Parameters
    ----------
    Ein : float
        The incident energy of the neutron in eV
    M : float
        The mass of the target material in amu
    T : float
        Temperature of the material in K
    Eout : np.ndarray, (N,)
        The neutron outgoing energy grid in eV
    mu : float
        Cosine of the scattering angle
    tau_n : 'np.ndarray', (M, T)
        all tau n functions in one array.
    tau_n_beta : 'np.ndarray', (M,)
        Space between beta grid points of tau n functions.
    DebyeWallerCoeff : float
        Debye Waller coefficient

    Returns
    -------
    S_diag : 'np.ndarray', (N,)
        Scattering function values for a single angle.

    Examples
    --------
    >>> import pandas as pd
    >>> from solid_cinel.core.material.vibration.pdos import Pdos
    >>> Ein = 7.2
    >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
    >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> mu = np.cos(np.deg2rad([120]))
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> tau_n, delta_beta, debye_waller_coeff = pdos.get_clm_param(T, nphonon=1000, threshold=1.0e-14)
    >>> sd_pdf = get_scatfunc_pdos(Ein, M, T, Eout, mu, tau_n, delta_beta, debye_waller_coeff)
    >>> pd.Series(sd_pdf[0], index=Eout).loc[Eout_test].round(6)
    6.7554    0.034511
    6.9050    0.426488
    7.0439    1.383082
    7.2000    1.262613
    7.3157    0.415630
    7.4480    0.042074
    dtype: float64
    """
    tau_n_beta = np.arange(tau_n.shape[1]) * delta_beta
    beta = get_beta(Eout, Ein, T)
    alpha_mat = get_alpha_mat(beta * kb * T + Ein if len(beta) < len(Eout) else Eout,
                              Ein, T, M, mu)
    sab_values = get_sab_pdos(alpha_mat, beta, tau_n, tau_n_beta, DebyeWallerCoeff)
    Eout_calc, scatfunc_values = scatfunc_values_alpha_mat(sab_values, beta, Ein, T, M)
    # Interpolation for avoiding numerical fluctuations:
    select_scarfunc = np.zeros((len(mu), len(Eout)))
    for i in range(len(mu)):
        select_scarfunc[i] += np.interp(Eout, Eout_calc, scatfunc_values[i])
    return select_scarfunc


@nb.jit(nopython=True, nogil=True, cache=True)
def get_scatfunc_pdos_row(Ein: float, M: float, T: float, Eout: np.ndarray,
                 mu: float, tau_n: np.ndarray, delta_beta: float,
                 DebyeWallerCoeff: float) -> np.ndarray:
    """
    Generate the scattering function from a S(alpha, -beta) table based on
    the phonon expansion model.

    Parameters
    ----------
    Ein : float
        The incident energy of the neutron in eV
    M : float
        The mass of the target material in amu
    T : float
        Temperature of the material in K
    Eout : np.ndarray, (N,)
        The neutron outgoing energy grid in eV
    mu : float
        Cosine of the scattering angle
    tau_n : 'np.ndarray', (M, T)
        all tau n functions in one array.
    tau_n_beta : 'np.ndarray', (M,)
        Space between beta grid points of tau n functions.
    DebyeWallerCoeff : float
        Debye Waller coefficient

    Returns
    -------
    S_diag : 'np.ndarray', (N,)
        Scattering function values for a single angle.

    Examples
    --------
    >>> import pandas as pd
    >>> from solid_cinel.core.material.vibration.pdos import Pdos
    >>> Ein = 7.2
    >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
    >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> mu = np.cos(np.deg2rad(120))
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> tau_n, delta_beta, debye_waller_coeff = pdos.get_clm_param(T, nphonon=1000, threshold=1.0e-14)
    >>> sd_pdf = get_scatfunc_pdos_row(Ein, M, T, Eout, mu, tau_n, delta_beta, debye_waller_coeff)
    >>> pd.Series(sd_pdf, index=Eout).loc[Eout_test].round(6)
    6.7554    0.034511
    6.9050    0.426488
    7.0439    1.383082
    7.2000    1.262613
    7.3157    0.415630
    7.4480    0.042074
    dtype: float64
    """
    tau_n_beta = np.arange(tau_n.shape[1]) * delta_beta
    beta = get_beta(Eout, Ein, T)
    alpha = get_alpha_from_Eout(beta * kb * T + Ein if len(beta) < len(Eout) else Eout, Ein, T, M, mu)
    sab_values = get_sab_pdos(alpha, beta, tau_n, tau_n_beta, DebyeWallerCoeff)
    Eout_calc, scatfunc_values = scatfunc_values_alpha_vec(sab_values, beta, Ein, T, M)
    # Interpolation for avoiding numerical fluctuations:
    return np.interp(Eout, Eout_calc, scatfunc_values)
