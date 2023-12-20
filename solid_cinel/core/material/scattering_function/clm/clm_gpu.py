

import numpy as np
import cupy as cp
from scipy.constants import physical_constants as const
from solid_cinel.core.material.scattering_function.beta import get_beta
from solid_cinel.core.material.scattering_function.alpha import get_alpha_mat, get_alpha_from_Eout
from solid_cinel.core.material.scattering_function.clm.clm_cpu import normalization_factor


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


def get_sab_pdos(alpha: cp.ndarray, beta: cp.ndarray,
                 tau_n: cp.ndarray, tau_n_beta: cp.ndarray,
                 DebyeWallerCoeff: float) -> cp.ndarray:
    # Zero phonon expansion:
    iter_sum = cp.log(alpha * DebyeWallerCoeff)
    alpha_mul = cp.exp(- alpha * DebyeWallerCoeff + iter_sum)
    S_diag = alpha_mul * cp.interp(beta, tau_n_beta, tau_n[0])

    # Higher phonon expansion (nphonon >= 1):
    for n in range(1, tau_n.shape[0]):
        # Compute S(alpha, -beta) for tau_n reshape
        iter_sum += cp.log(alpha * DebyeWallerCoeff / (n + 1))
        alpha_mul = cp.exp(- alpha * DebyeWallerCoeff + iter_sum)
        S_diag += alpha_mul * cp.interp(beta, tau_n_beta, tau_n[n])
    return S_diag


def scatfunc_values_alpha_vec(Sab_mat: cp.ndarray, beta: cp.ndarray, Ein: float,
                        T: float, M: float) -> (cp.ndarray, cp.ndarray):
    # Scattering function values calculation:
    ScatFunc_values = cp.concatenate((Sab_mat[::-1], Sab_mat[1::] * np.exp(-beta[1:])))

    # Eout calculation
    Eout_calc = Ein + cp.concatenate((-beta[::-1], beta[1::])) * kb * T

    # Ensure the Eout values are positive:
    positive_mask = Eout_calc > 0
    Eout_calc = Eout_calc[positive_mask]

    # Normalization constant
    norm = normalization_factor(Eout_calc, Ein, T, M)

    return Eout_calc, ScatFunc_values[positive_mask] * norm


def scatfunc_values_alpha_mat(Sab_values: cp.ndarray, beta: cp.ndarray, Ein: float,
                              T: float, M: float) -> (np.ndarray, np.ndarray):
    ScatFunc_values = cp.concatenate(
        (Sab_values[::, ::-1], Sab_values[::, 1:] * np.exp(-beta[1:])), axis=1)
    # Eout calculation
    Eout_calc = cp.sort(Ein + cp.concatenate((-beta[::-1], beta[1::])) * kb * T)

    # Ensure the Eout values are positive:
    positive_mask = Eout_calc > 0
    Eout_calc = Eout_calc[positive_mask]

    # Normalization constant
    norm = normalization_factor(Eout_calc, Ein, T, M)

    return Eout_calc, ScatFunc_values[::, positive_mask] * norm


def get_scatfunc_pdos(Ein: float, M: float, T: float, Eout: cp.ndarray,
                      mu: cp.ndarray, tau_n: cp.ndarray, delta_beta: float,
                      DebyeWallerCoeff: float) -> cp.ndarray:
    tau_n_beta = np.arange(tau_n.shape[1]) * delta_beta
    beta = get_beta(Eout, Ein, T)
    alpha_mat = get_alpha_mat(beta * kb * T + Ein if len(beta) < len(Eout) else Eout,
                              Ein, T, M, mu)
    sab_values = get_sab_pdos(alpha_mat, beta, tau_n, tau_n_beta, DebyeWallerCoeff)
    Eout_calc, scatfunc_values = scatfunc_values_alpha_mat(sab_values, beta, Ein, T, M)
    # Interpolation for avoiding numerical fluctuations:
    def interp_row(row):
        return cp.interp(Eout, Eout_calc, row)
    return cp.apply_along_axis(interp_row, axis=1, arr=scatfunc_values)


def get_scatfunc_pdos_row(Ein: float, M: float, T: float, Eout: cp.ndarray,
                          mu: float, tau_n: cp.ndarray, delta_beta: float,
                          DebyeWallerCoeff: float) -> cp.ndarray:
    tau_n_beta = cp.arange(tau_n.shape[1]) * delta_beta
    beta = get_beta(Eout, Ein, T)
    alpha = get_alpha_from_Eout(beta * kb * T + Ein if len(beta) < len(Eout) else Eout, Ein, T, M, mu)
    sab_values = get_sab_pdos(alpha, beta, tau_n, tau_n_beta, DebyeWallerCoeff)
    Eout_calc, scatfunc_values = scatfunc_values_alpha_vec(sab_values, beta, Ein, T, M)
    # Interpolation for avoiding numerical fluctuations:
    return cp.interp(Eout, Eout_calc, scatfunc_values)
