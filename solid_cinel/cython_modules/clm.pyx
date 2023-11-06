import numpy as np
import numpy as np

cimport numpy as cnp
from numpy cimport (
    float64_t,
    int64_t,
    uint8_t,
    ndarray,
)

cnp.import_array()
from scipy.constants import physical_constants as const
from libc.math cimport sqrt, exp, pi, fmax, log
import cython
from cython.parallel import prange

cdef double m = const["neutron mass in u"][0]  # Define the value of m
cdef double kb = const["Boltzmann constant in eV/K"][0]  # Define the value of kb

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ndarray[float64_t, ndim=1] sigma1(ndarray[float64_t, ndim=1] Eout, float64_t Ein, float64_t T, float64_t M):
    """
    Sigma1 function for Energy differential scattering function
    ..math::
           S(E, E^\prime, M, T) = \frac{1}{2}\sqrt{\frac{M}{m\pi k_BT}}\frac{\sqrt{E^\prime}}{E}\left(exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} - \sqrt{E^\prime}\right)^2 \right) - exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} + \sqrt{E^\prime}\right)^2 \right)\right)

    Parameters
    ----------
    Eout : np.array
        Outgoing energy grid in eV
    Ein : float
        Incoming energy in eV
    T : float
        Temperature in K
    M :
        Mass of the target in amu

    Returns
    -------
    scattfunc : np.array
        Scattering function based on sigma1 model
    """
    cdef:
        int n = Eout.shape[0]
        ndarray[float64_t, ndim=1] exp_negative = exp(- M / (m * kb * T) * (sqrt(Ein) - sqrt(Eout)) ** 2)
        ndarray[float64_t, ndim=1] exp_positive = exp(- M / (m * kb * T) * (sqrt(Ein) + sqrt(Eout)) ** 2)
        ndarray[float64_t, ndim=1] scattfunc = 0.5 * (exp_negative - exp_positive) * sqrt(Eout) / Ein
    scattfunc *= sqrt(M / (pi * m * kb * T))
    return scattfunc

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ndarray[float64_t, ndim=1] get_scat_sct_angular(ndarray[float64_t, ndim=1] Eout, float64_t mu, float64_t Ein, float64_t T, float64_t M, float64_t Teff, float64_t ws):
    """
    Calculate the scattering function from the Short Collision Time model using
    a single angle.
    ..math::
        S(\theta, E^\prime, E, M, T) = \frac{1}{2 * k_B * T}\sqrt{\frac{E^\prime}{E}} \frac{1}{\sqrt{4 \pi w_s \alpha T_{eff} / T}} exp\left(\frac{(w_s\alpha +\beta)^2}{4 \alpha w_s T_{eff}/T}\right)

    Parameters
    ----------
    Eout : np.ndarray, (N,)
        The neutron outgoing energy grid in eV
    mu : float
        Cosine of the angle between the incident neutron direction and
        the outgoing neutron direction
    Ein : float
        The incident energy of the neutron in eV
    T : float
        Temperature of the material in K
    M : float
        The mass of the target material in amu
    Teff : float
        Effective temperature of the material in K
    ws : float
        Normalization for continuous (vibrational) part. For solid is 1.

    Returns
    -------
    np.array, (N,)
        The scattering function values for a single angle
    """
    cdef:
        float64_t awr = ((M / m + 1) / (M / m)) ** 2
        ndarray[float64_t, ndim=1] beta = (Eout - Ein) / (kb * T)
        ndarray[float64_t, ndim=1] alpha = Eout + Ein - 2 * mu * sqrt(Eout * Ein) / (M * kb * T / m)
        ndarray[float64_t, ndim=1] scattfunc = exp(-(ws * alpha + beta) ** 2 / (4 * alpha * Teff / T * ws))

    scattfunc /= sqrt(4 * pi * ws * alpha * Teff / T)
    scattfunc *= awr * sqrt(Eout / Ein) / (2 * kb * T)
    return scattfunc


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef ndarray[float64_t, ndim=1] default_Eout(float64_t Ein):
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
    """
    cdef:
        ndarray[float64_t, ndim=1] Eout_small = np.linspace(0, 0.99 * Ein, 2000)
        ndarray[float64_t, ndim=1] Eout_middle = np.linspace(0.99 * Ein, Ein * 1.01, 3000)
        ndarray[float64_t, ndim=1] Eout_great

    if Ein * 2 < 5.0:
        Eout_great = np.logspace(np.log10(Ein * 1.01), np.log10(5.0), 2000)
    else:
        Eout_great = np.logspace(np.log10(Ein * 1.01), np.log10(2 * Ein), 2000)

    return np.sort(np.concatenate((Eout_great, Eout_small, Eout_middle)))


cpdef float64_t Db(ndarray[float64_t, ndim=1] xs_values, ndarray[float64_t, ndim=1] xs_E, float64_t Ein, ndarray[float64_t, ndim=1] Eout, ndarray[float64_t, ndim=1] pdf):
    """
    Calculate the doppler broadening of a cross section for a pdf

    Parameters
    ----------
    xs_values: np.ndarray, (N,)
        Cross section values in barns
    xs_E: np.ndarray, (N,)
        Cross section energy grid in eV
    Ein: float
        The incident energy of the neutron in eV
    Eout: np.ndarray, (Z,)
        The neutron outgoing energy grid in eV
    pdf: np.ndarray, (Z,)
        Probability density function

    Returns
    -------
    Db_xs: float
        Doppler broadened cross section in barns
    """
    cdef:
        int64_t max_pos
        float64_t norm, recoil, Db_xs

    max_pos = np.argmax(pdf)
    if pdf[max_pos] > 1.0e308:  # Overflow found in pdf_val
        Db_xs = np.interp(Eout[max_pos], xs_E, xs_values)
    else:
        norm = np.trapz(pdf, x=Eout)
        # Recoil:
        recoil = Ein - Eout[max_pos]
        # xs:
        xs_Eout_arno = np.interp(Eout + recoil, xs_E, xs_values)
        Db_xs = np.trapz(xs_Eout_arno * pdf, x=Eout) / norm
    return Db_xs


cpdef ndarray[float64_t, ndim=2] get_Ein_arno(float64_t Ein, ndarray[float64_t, ndim=1] Eout, ndarray[float64_t, ndim=1] mu, float64_t M):
    """
    Get the incident energy matrix for the arno model.

    Parameters
    ----------
    Ein: float
        The incident energy of the neutron in eV
    Eout: np.ndarray, (Z,)
        The neutron outgoing energy grid in eV
    mu: np.ndarray, (M,)
        The neutron outgoing angle grid in degrees (0, 180]
    M: float
        Mass of the material in amu

    Returns
    -------
    Ein_arno: np.ndarray, (M, Z)
        Incident energy matrix for the arno model
    """
    cdef ndarray[float64_t, ndim=2] Ein_arno = np.empty((len(mu), len(Eout)))
    for i in range(len(mu)):
        alpha = (Ein + Eout - 2 * mu[i] * np.sqrt(Ein * Eout)) * m / M
        Ein_arno[i, :] = (Eout + Ein) / 2 - Ein * mu[i] * m / M
        Ein_arno[i, :] += 0.5 * alpha / (1 - mu[i])
    return Ein_arno

cpdef ndarray[float64_t, ndim=2] xs_matrix_sigma1(ndarray[float64_t, ndim=1] xs_values, ndarray[float64_t, ndim=1] xs_E,
                                                  float64_t Ein, float64_t M, ndarray[float64_t, ndim=1] T_arno,
                                                  ndarray[float64_t, ndim=1] Eout, ndarray[float64_t, ndim=1] mu):
    cdef:
        ndarray[float64_t, ndim=2] xs_mat = np.empty((len(mu), len(Eout)))
        ndarray[float64_t, ndim=2] Ein_arno = get_Ein_arno(Ein, Eout, mu, M)
        ndarray[float64_t, ndim=1] Eout_db = np.empty(7000)
        ndarray[float64_t, ndim=1] pdf = np.empty(7000)

    for i in range(len(mu)):
        if mu[i] == np.cos(np.pi):
            xs_mat[i, :] = np.interp(Ein_arno[i, :], xs_E, xs_values)
            continue
        for j in range(7000):
            Eout_db = default_Eout(Ein_arno[i, j])
            pdf = sigma1(Eout_db, Ein_arno[i, j], T_arno[i], M)
            xs_mat[i, j] = Db(xs_values, xs_E, Ein_arno[i, j], Eout_db, pdf)
    return xs_mat


cpdef ndarray[float64_t, ndim=2] xs_matrix_sct(ndarray[float64_t, ndim=1] xs_values, ndarray[float64_t, ndim=1] xs_E,
                                                  float64_t Ein, float64_t M, ndarray[float64_t, ndim=1] T_arno,
                                                  ndarray[float64_t, ndim=1] Eout, ndarray[float64_t, ndim=1] mu,
                                                  float64_t mu_fit, ndarray[float64_t, ndim=1] Teff, float64_t ws):
    cdef:
        ndarray[float64_t, ndim=2] xs_mat = np.empty((len(mu), len(Eout)))
        ndarray[float64_t, ndim=2] Ein_arno = get_Ein_arno(Ein, Eout, mu, M)
        ndarray[float64_t, ndim=1] Eout_db = np.empty(7000)
        ndarray[float64_t, ndim=1] pdf = np.empty(7000)

    for i in range(len(mu)):
        if mu[i] == np.cos(np.pi):
            xs_mat[i, :] = np.interp(Ein_arno[i, :], xs_E, xs_values)
            continue
        for j in range(7000):
            Eout_db = default_Eout(Ein_arno[i, j])
            pdf = get_scat_sct_angular(Eout_db, mu_fit, Ein_arno[i, j], T_arno[i], M, Teff[i], ws)
            xs_mat[i, j] = Db(xs_values, xs_E, Ein_arno[i, j], Eout_db, pdf)
    return xs_mat

# PDOS:

cpdef ndarray[float64_t, ndim=1] tau_n_CPU(float64_t delta_beta, double[:] tau1, double[:] tau_n_minus_1,
              float64_t threshold):
    """
    Get the tau_{n}(-beta) function values.

    Parameters
    ----------
    delta_beta : 'float'
        Interval of beta for the PDOS.
    tau1 : 'np.ndarray', (N,)
        Tau(-beta) function for n = 1 expansion.
    tau_n_minus_1 : 'np.ndarray', (N,)
        Tau(-beta) function for n - 1 expansion.
    threshold : 'float'
        Minimun value to take into account.

    Returns
    -------
    tau_n : 'np.ndarray', (N,)
        Tau(-beta) function for n expansion.
    """
    cdef:
        int Nnm1 = len(tau_n_minus_1), N = len(tau1), tau_n_len = Nnm1 + N - 1
        double[:] tau_n = np.zeros(tau_n_len)
        float64_t convol
        int k, l, i, j

    for i in prange(tau_n_len, nogil=True):  # loop for tau_n
        for j in range(N):  # loop for tau1
            convol = 0.

            k = i - j  # tau_n_minus_1(-(beta-beta^prime))
            if k >= 0 and k < Nnm1:
                convol = tau_n_minus_1[k]
            elif k < 0 and -k < Nnm1:  # tau(beta) = exp(-beta)Tau(-beta)
                convol = tau_n_minus_1[-k] * exp(k * delta_beta)

            l = i + j  # Tau_n_minus_1(-(beta+beta^prime))
            if l < Nnm1:
                convol += tau_n_minus_1[l] * exp(-j * delta_beta)

            if j == 0 or j == N - 1:
                tau_n[i] += tau1[j] * convol * delta_beta / 2 # trapz integrate
            else:
                tau_n[i] += tau1[j] * convol * delta_beta

    return np.asarray(tau_n)


cpdef ndarray[float64_t, ndim=1] get_diag_S_from_tau_n(ndarray[float64_t, ndim=1] tau, ndarray[float64_t, ndim=1] beta_tau,
                     float64_t debye_waller_coeff, ndarray[float64_t, ndim=1] iter_sum,
                     ndarray[float64_t, ndim=1] alpha, ndarray[float64_t, ndim=1] beta):
    """
    Generate the scattering function from a S(alpha, -beta) table based on
    the phonon expansion model using a single angle for tau_n function.

    Parameters
    ----------
    tau : 'np.ndarray', (T,)
        tau function values.
    beta_tau : 'np.ndarray', (T,)
        beta grid for tau function.
    debye_waller_coeff : 'float'
        Debye Waller Coefficient.
    alpha : 1D iterable, (N,)
        Alpha grid.
    beta: 1D iterable, (N,)
        beta grid.

    Returns
    -------
    'np.ndarray', (N,)
        Scattering function values for a single angle for tau_n function.
    """
    cdef:
        ndarray[float64_t, ndim=1] alpha_mul = exp(- alpha * debye_waller_coeff + iter_sum)
        ndarray[float64_t, ndim=1] tau_n_reshape = np.interp(beta, beta_tau, tau)
    # Bounds in nopython mode:
    if beta[-1] > beta_tau[-1]:
        tau_n_reshape[beta > beta_tau[-1]] = 0.0
    return alpha_mul * tau_n_reshape

cpdef ndarray[float64_t, ndim=1] get_diag_S_pdos(ndarray[float64_t, ndim=1] alpha, ndarray[float64_t, ndim=1] beta,
                    int64_t nphonon, ndarray[float64_t, ndim=1] tau1, float64_t delta_beta,
                    float64_t threshold, float64_t DebyeWallerCoeff):
    """
    Generate the scattering function from a S(alpha, -beta) table based on
    the phonon expansion model using a single angle.

    Parameters
    ----------
    alpha : 'np.ndarray', (N,)
        alpha grid values.
    beta : 'np.ndarray', (N,)
        beta grid values.
    nphonon : 'int', optional
        Phonon expansion order.
    tau1 : 'np.ndarray', (M,)
        tau1 function values.
    delta_beta : float
        Space between beta grid points.
    threshold : 'float', optional
        Minimun value to take into account in the creation of tau_n
        functions. For T>200 is convenient to set into 1.0e-14 to speed up
        the calculations.
    DebyeWallerCoeff : 'float'
        Debye Waller Coefficient in LEAPR formalism.

    Returns
    -------
    S_diag : 'np.ndarray', (N,)
        Scattering function values for a single angle.
    """
    cdef:
        ndarray[float64_t, ndim=1] iter_sum = log(alpha * DebyeWallerCoeff)
        ndarray[float64_t, ndim=1] beta_tau_1 = np.arange(len(tau1)) * delta_beta
        ndarray[float64_t, ndim=1] S_diag = get_diag_S_from_tau_n(tau1, beta_tau_1, DebyeWallerCoeff, iter_sum, alpha, beta)
        int n
        ndarray[float64_t, ndim=1] beta_tau_n, tau_n, tau_n_minus_1
        
    tau_n_minus_1 = tau1.copy()
    if len(alpha) != len(beta):
        raise ValueError("alpha and beta must have the same length")

    # Higher phonon expansion (nphonon >= 1):
    for n in range(1, nphonon + 1):
        # Tau_n(-beta)
        tau_n = tau_n_CPU(delta_beta, tau1, tau_n_minus_1, threshold)
        beta_tau_n = np.arange(len(tau_n)) * delta_beta

        # Compute S(alpha, -beta) for tau_n reshape
        iter_sum += log(alpha * DebyeWallerCoeff / (n + 1))
        S_diag += get_diag_S_from_tau_n(tau_n, beta_tau_n,
                                        DebyeWallerCoeff, iter_sum, alpha, beta)

        # Next tau_n
        tau_n_minus_1 = tau_n
    return S_diag

cpdef ndarray[float64_t, ndim=2] get_ScatFunc_values(ndarray[float64_t, ndim=1] Sab_mat, ndarray[float64_t, ndim=1] beta_grid, float64_t Ein,
                        float64_t T, float64_t M):
    """
    Generate the scattering function values from a S(alpha, -beta) table based on
    the phonon expansion model for a single angle

    Parameters
    ----------
    Sab_mat : 'np.ndarray', (N,)
        S(alpha, -beta) matrix values.
    beta_grid : 'np.ndarray', (N,)
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
    """
    cdef:
        ndarray[float64_t, ndim=1] Eout = np.sort(Ein + np.concatenate((-beta_grid[::-1], beta_grid[1::])) * kb * T)
        ndarray[uint8_t, ndim=1] positive_mask = Eout > 0
        float64_t aws = ((M / m + 1) / (M / m)) ** 2
        ndarray[float64_t, ndim=1] normalization_factor, ScatFunc_values
    # Scattering function values calculation:
    ScatFunc_values = np.concatenate((Sab_mat[::-1], Sab_mat[1::]))
    ScatFunc_values[len(Sab_mat)::] *= np.exp(-beta_grid[1::])

    # Ensure the Eout values are positive:
    ScatFunc_values = ScatFunc_values[positive_mask]
    Eout = Eout[positive_mask]

    # Handle nan values:
    ScatFunc_values[np.isnan(ScatFunc_values)] = 0

    # Normalization constant
    normalization_factor = aws * np.sqrt(Eout / Ein) / (2 * kb * T)

    return np.vstack((Eout, ScatFunc_values * normalization_factor)).T


cpdef ndarray[float64_t, ndim=1] get_ScatFunc_pdos_angle(float64_t Ein, float64_t M, float64_t T, Eout: np.ndarray,
                 float64_t mu, int64_t nphonon, ndarray[float64_t, ndim=1] tau1,
                 float64_t delta_beta, float64_t threshold,
                 float64_t DebyeWallerCoeff):
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
    Eout : np.ndarray
        The neutron outgoing energy grid in eV
    mu : float
        Cosine of the scattering angle
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
    """
    cdef: 
        ndarray[float64_t, ndim=1] Eout_, beta, alpha, Sab_values, sd_pdf
    beta = (Eout - Ein) / (kb * T)
    beta = np.unique(np.absolute(beta))
    if len(beta) < len(Eout): # same beta values but one negative and one positive
        Eout_ = beta * kb * T + Ein
    else:
        Eout_ = Eout.copy()
    alpha = Eout + Ein - 2 * mu * np.sqrt(Eout * Ein)
    alpha /= (M * kb * T / m)
    Sab_values = get_diag_S_pdos(alpha, beta, nphonon, tau1, delta_beta,
                                                            threshold, DebyeWallerCoeff)
    sd_pdf = get_ScatFunc_values(Sab_values, beta, Ein, T, M)
    # Interpolation for avoiding numerical fluctuations:
    return np.interp(Eout, sd_pdf[:, 0], sd_pdf[:, 1])

cpdef ndarray[float64_t, ndim=2] xs_matrix_clm(ndarray[float64_t, ndim=1] xs_values, ndarray[float64_t, ndim=1] xs_E,
                                               float64_t Ein, float64_t M, ndarray[float64_t, ndim=1] T_arno,
                                               ndarray[float64_t, ndim=1] Eout, ndarray[float64_t, ndim=1] mu,
                                               float64_t mu_fit, int64_t nphonon,  ndarray[float64_t, ndim=2] tau1,
                                               ndarray[float64_t, ndim=1] delta_beta, float64_t threshold,
                                               ndarray[float64_t, ndim=1] DebyeWallerCoeff):
    cdef:
        ndarray[float64_t, ndim=2] xs_mat = np.empty((len(mu), len(Eout)))
        ndarray[float64_t, ndim=2] Ein_arno = get_Ein_arno(Ein, Eout, mu, M)
        ndarray[float64_t, ndim=1] Eout_db = np.empty(7000)
        ndarray[float64_t, ndim=1] pdf = np.empty(7000)

    for i in range(len(mu)):
        if mu[i] == np.cos(np.pi):
            xs_mat[i, :] = np.interp(Ein_arno[i, :], xs_E, xs_values)
            continue
        for j in range(7000):
            Eout_db = default_Eout(Ein_arno[i, j])
            pdf = get_ScatFunc_pdos_angle(Ein_arno[i, j], M, T_arno[i],
                                              Eout_db, mu_fit, nphonon,
                                              tau1[i], delta_beta[i],
                                              threshold, DebyeWallerCoeff[i])
            xs_mat[i, j] = Db(xs_values, xs_E, Ein_arno[i, j], Eout_db, pdf)
    return xs_mat