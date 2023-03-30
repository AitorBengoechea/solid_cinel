import numba as nb
import numpy as np
from numba import prange
from numba import cuda
import math
from scipy.constants import physical_constants as const

kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]


@nb.jit(nopython=True, nogil=True)
def hklloop(d_min: float, hkl_max: np.array, rec_vecs: np.array, Bfac: dict,
            pos: dict, csl: dict, preferred_orientation: np.array,
            precision: np.array) -> dict:
    """
    Get the F_hkl and d_hkl for all the posible h, k, l plane combination that
    fill the condition of d_hkl > d_min
    .. math::
        d_{hkl} = \frac{2\pi}{\tau_{hkl}}
        F(\vec{\tau}_{hkl})=\sum_{j=1}^{N_{uc}}b_j\exp\left(-\dfrac{\hbar^2\tau_{hkl}^2}{4M_jk_BT}\Lambda_j(T)\right) e^{i\vec{\tau}_{hkl}\cdot\vec{p}_j}

    Parameters
    ----------
    d_min : 'float'
        The minimum dspacing for the LEAPR module of NJOY
    hkl_max : 'np.array'
        Maximun h, k, l index for generating a d > d_min
    rec_vecs : 'np.array'
        Reciprocal vectors
    Bfac : 'nb.typed.Dict'
        Dict with the B factor for Target_Material object elements.
    pos : 'nb.typed.Dict'
        Dict with atomic position of elements in Target_Material object.
    csl : 'nb.typed.Dict'
        Coherent elastic length for each element of Target_Material object.
    preferred_orientation: "np.array"
        Array with the preferred orientation of the solid.
    precision: "float"
        Precision of the rounding in the calculation to merge different plane
        values

    Returns
    --------
    "dict"
        Dictionary containing the hkl planes, the d_hkl, Fsq, orientation_angle.
    """
    hklM = {}
    hkldF = {}
    h_range, k_range, l_range = [np.arange(-x, x + 1) for x in hkl_max]

    for h in h_range[::-1]:  # to get positive hkl order
        for k in k_range[::-1]:
            for l in l_range[::-1]:
                if h ** 2 + k ** 2 + l ** 2 == 0:  # (0, 0, 0) is excluded
                    continue

                # d_hkl:
                vec_tau_hkl = h * rec_vecs[0] + k * rec_vecs[1] + l * rec_vecs[2]
                d_hkl = 2 * np.pi / np.linalg.norm(vec_tau_hkl)

                if d_hkl < d_min:  # d < d_min is excluded
                    continue

                Fsq = Fsq_hkl(vec_tau_hkl, Bfac, csl, pos)  # Fsquared

                # same dspacing and Fsquared with precision will be regrouped
                d_rnd = round(d_hkl, precision[0])
                Fsq_rnd = round(Fsq, precision[1])
                if (d_rnd, Fsq_rnd) in hkldF:
                    hklM[hkldF[(d_rnd, Fsq_rnd)]][-1] += 1
                else:
                    hkldF[(d_rnd, Fsq_rnd)] = (h, k, l)
                    OA_num = np.sum(vec_tau_hkl * preferred_orientation)
                    OA_den = np.linalg.norm(vec_tau_hkl) * np.linalg.norm(preferred_orientation)
                    orientation_angle_hkl = np.arccos(OA_num / OA_den) * 180 / np.pi
                    hklM[(h, k, l)] = np.array([d_hkl, Fsq, orientation_angle_hkl, 1])
    return hklM


@nb.jit(nopython=True, nogil=True, cache=True)
def Fsq_hkl(vec_tau_hkl: np.array, Bfac: dict, csl:dict, pos:dict) -> float:
    """
    Get F_hkl:
    .. math::
        F(\vec{\tau}_{hkl})=\sum_{j=1}^{N_{uc}}b_j\exp\left(-\dfrac{\hbar^2\tau_{hkl}^2}{4M_jk_BT}\Lambda_j(T)\right) e^{i\vec{\tau}_{hkl}\cdot\vec{p}_j}

    Parameters
    ----------
    vec_tau_hkl : 'np.array'
        Reciprocal vectors
    Bfac : 'nb.typed.Dict'
        Dict with the B factor for Target_Material object elements.
    pos : 'nb.typed.Dict'
        Dict with atomic position of elements in Target_Material object.
    csl : 'nb.typed.Dict'
        Coherent elastic length for each element of Target_Material object.

    Returns
    -------
    "float"
        Fsq_hkl value for that (h, k, l) plane
    """
    real = 0.
    imag = 0.
    for element in Bfac:
        expon_hkl = np.exp(-0.5 * np.linalg.norm(vec_tau_hkl) ** 2
                           * Bfac[element] / (8 * np.pi ** 2))
        element_position = pos[element]
        for iep in range(len(element_position)):
            cumulant_cos = np.cos(np.sum(vec_tau_hkl * element_position[iep]))
            cumulant_sin = np.sin(np.sum(vec_tau_hkl * element_position[iep]))
            real += csl[element] * 0.1 * expon_hkl * cumulant_cos
            imag += csl[element] * 0.1 * expon_hkl * cumulant_sin
    return real ** 2 + imag ** 2  # Fsquared


@nb.jit(nopython=True, nogil=False, cache=True, parallel=True)
def tau_n_CPU(delta_beta: float, tau1: np.array, tau_n_minus_1: np.array,
              threshold: float) -> np.array:
    """
    Get the tau_{n}(-beta) function values.

    Parameters
    ----------
    delta_beta : 'float'
        Interval of beta for the PDOS.
    tau1 : 'np.array' of 1D
        Tau(-beta) function for n = 1 expansion.
    tau_n_minus_1 : 'np.array' of 1D
        Tau(-beta) function for n - 1 expansion.
    threshold : 'float'
        Minimun value to take into account.

    Returns
    -------
    tau_n : 'np.array' of 1D
        Tau(-beta) function for n expansion.
    """
    tau_n = np.zeros(len(tau1) + len(tau_n_minus_1) - 1)
    Nnm1 = len(tau_n_minus_1)  # length of tau_n_minus_1
    N = len(tau1)

    for i in prange(len(tau_n)):  # loop for tau_n
        for j in prange(N):  # loop for tau1
            convol = 0.

            k = i - j  # tau_n_minus_1(-(beta-beta^pirme))
            if k >= 0 and k < Nnm1:
                convol = tau_n_minus_1[k]
            elif k < 0 and -k < Nnm1:  # tau(beta) = exp(-beta)Tau(-beta)
                convol = tau_n_minus_1[-k] * np.exp(k * delta_beta)

            l = i + j  # Tau_n_minus_1(-(beta+beta^pirme))
            if l < Nnm1:
                convol += tau_n_minus_1[l] * np.exp(-j * delta_beta)

            if j == 0 or j == N - 1:
                convol *= 0.5                      # trapz integrate

            tau_n[i] += tau1[j] * convol * delta_beta

    return tau_n[tau_n >= threshold]

@cuda.jit
def convolTaufunc(interv_in_beta, Tau1_neg, Taunm1_neg, Taun_neg):
    '''Tau function for n phonon (n > 1) in LEAPR formalism is 
    to be computed by convolution. Tau1_neg function is zero 
    beyond the maximum value of beta ((num_dos - 1) * interv_in_beta). 
    According to the calculation of Taun_neg by convolution, by recurrence,
    Taun_neg is nonzero within n * (num_dos - 1) * interv_in_beta.
    
    The convolution process is direct : Tau1_neg and Taunm1_neg will be
    extended to the same length as Taun_neg by adding 0 in the end;
    then the positive part is also computed by multiplying exp(-beta);
    finally Taun_neg can be computed by the convolution formula.
    
    interv_in_beta : interval of beta for the PDOS, float
    Tau1_neg : Tau function for n = 1 negative part, num_dos * 1 array
    Taunm1_neg : Tau function for n - 1 negative part, ((n - 1) * (num_dos - 1) + 1) * 1 array
    Taun_neg : Taun_neg function for n negative part, in LEAPR formalism, 
                (n * (num_dos - 1) + 1) * 1 array
    '''
    
    start = cuda.grid(1)      # 1 = one dimensional thread grid, returns a single value
    stride = cuda.gridsize(1) # ditto
    
    N = len(Tau1_neg)                             # length of Tau1_neg
    Nnm1 = len(Taunm1_neg)                        # length of Taunm1_neg
    Nn = len(Taun_neg)                            # length of Taun_neg
    
    for i in prange(start, Nn, stride):
        for j in prange(N):                       # loop for Tau1_neg
            convol = 0.
            k = i - j                             # indice of Taunm1_neg(-(beta-beta^pirme))
            if 0 <= k and k < Nnm1:
                convol = Taunm1_neg[k]
            elif -Nnm1 < k and k < 0:             # Tau(beta) = exp(-beta)Tau(beta)
                convol = Taunm1_neg[-k] * math.exp(k * interv_in_beta)
            l = i + j                             # indice of Taunm1_neg(-(beta+beta^pirme))
            if l < Nnm1:
                convol += Taunm1_neg[l] * math.exp(-j * interv_in_beta)
            convol *= Tau1_neg[j]
            if j == 0 or j == N - 1:
                convol *= 0.5                      # trapz integrate
            Taun_neg[i] += interv_in_beta * convol
    return Taun_neg


@nb.jit(nopython=True, nogil=False, cache=True, parallel=True)
def update_Sab_with_tau_n(n: int, alpha_grid: np.array, DebyeWallerCoeff: float,
                          tau_n: np.array, Sab: np.array,
                          iter_sum: np.array) -> np.array:
    """
    Iterative sum into a S(alpha, -beta) matrix of tau_n(-beta) functions. This
    function only add one term to term to the matrix.
    .. math::
        S(\alpha,\,-\beta)=\exp(-\alpha\lambda)\sum_{n=0}^{\infty}\dfrac{1}{n!}(\alpha\lambda)^n\mathcal{T}_n(-\beta)


    Numerical appoximation to get convergence in large exponentiation and
    factorial numbers. Each element of the array is related with one alpha
    and represent the following term of the previous equation:
    ..math::
       \sum_{n=0}^{\infty}\dfrac{1}{n!}(\alpha\lambda)^n = \exp(\log(\dfrac{1}{1}(\alpha\lambda)) + \log(\dfrac{1}{2}(\alpha\lambda)) + ...)

    Parameters
    ----------
    n : 'int'
        phonon expansion order in python nomenclature. For example, for
        convenience with arrays, the order 2 is represent with n=1
        (order = n + 1).
    alpha_grid : 'np.array[:]'
        alpha values of the matrix S(alpha, -beta).
    DebyeWallerCoeff : 'float'
        Debye Wallelr Coefficient.
    tau_n : 'np.array[:]'
        tau_n values.
    Sab : 'np.array[:, :]'
        S(alpha, -beta) matrix for n-1 expansion order.
    iter_sum : 'np.array[:]'
        Iterative coefficient explained above for n-1 expansion order.

    Returns
    -------
    Sab : 'np.array[:, :]'
        S(alpha, -beta) matrix for n expansion order..
    iter_sum : 'np.array[:]'
        Iterative coefficient explained above for n expansion order.

    """
    for alpha in prange(len(alpha_grid)):
        iter_sum[alpha] += np.log(alpha_grid[alpha] * DebyeWallerCoeff / (n + 1))
        alpha_mul = np.exp(-alpha_grid[alpha] * DebyeWallerCoeff + iter_sum[alpha])
        Sab[alpha, :] += alpha_mul * tau_n
    return Sab, iter_sum


@nb.jit(nopython=True, nogil=True)
def convolSab(alphai, SabSolid_alpha, delta_beta, SabDiff, Nb, beta,
              debye_waller_coeff, bmax, data_type = np.double):
    '''combined S(alpha, -beta) for a given alpha
    '''
    
    NbDiff = len(SabDiff)
    betaDiff = np.arange(NbDiff) * delta_beta # beta for Sab diffusion part    
    SabSolid_convol = np.zeros(NbDiff, dtype = data_type) # solid-type S(alpha, beta) for convolution
    SabComb_alpha = np.zeros(Nb, dtype = data_type)
    
    for ib in prange(Nb): # loop in beta
        # first part in the right hand side, S_t(alpha, -beta)exp(-alpha*lambda)
        if beta[ib] < np.max(betaDiff):
            j = np.searchsorted(betaDiff, beta[ib], side = 'right') # betaDiff[j-1] <= beta[ib] < betaDiff[j] 
            Sinterp = interp1(betaDiff[j - 1], SabDiff[j - 1],
                              betaDiff[j], SabDiff[j],
                              beta[ib], 'linlog')
            s = Sinterp * np.exp(-alphai * debye_waller_coeff)
            SabComb_alpha[ib] += s
                
        # second part in the right hand side
        SabSolid_convol[0] = SabSolid_alpha[ib] * 2 # beta^prime=0, avoid the problem in the right bound
        for ibD in prange(1, NbDiff): # loop in beta^prime in convolution                
            betapr_i = beta[ib] - betaDiff[ibD]
            if 0 <= betapr_i and betapr_i < bmax:
                j = np.searchsorted(beta, betapr_i, side = 'right') # beta[j-1] <= betapr_i < beta[j]
                if SabSolid_alpha[j - 1] > 0 and SabSolid_alpha[j] > 0:
                    convol1 = interp1(beta[j - 1], SabSolid_alpha[j - 1],
                                      beta[j], SabSolid_alpha[j],
                                      betapr_i, 'linlog')
                else:
                    convol1 = 0.
            elif -bmax < betapr_i and betapr_i < 0:
                j = np.searchsorted(beta, -betapr_i, side = 'right') # beta[j-1] <= -betapr_i < beta[j]                        
                if SabSolid_alpha[j - 1] > 0 and SabSolid_alpha[j] > 0:
                    convol1 = interp1(beta[j - 1], SabSolid_alpha[j - 1],
                                      beta[j], SabSolid_alpha[j],
                                      -betapr_i, 'linlog')
                    convol1 *= np.exp(betapr_i)
                else:
                    convol1 = 0.
            else:
                convol1 = 0.
                    
            betapr_i_neg = beta[ib] + betaDiff[ibD]
            if betapr_i_neg < bmax:
                j = np.searchsorted(beta, betapr_i_neg, side = 'right') # beta[j-1] <= betapr_i_neg < beta[j]
                if SabSolid_alpha[j - 1] > 0 and SabSolid_alpha[j] > 0:
                    convol2 = interp1(beta[j - 1], SabSolid_alpha[j - 1],
                                      beta[j], SabSolid_alpha[j],
                                      betapr_i_neg, 'linlog')
                    convol2 *= np.exp(-betaDiff[ibD])
                else:
                    convol2 = 0.
            else:
                convol2 = 0.
                
            SabSolid_convol[ibD] = convol1 + convol2
            
        s = integrate_trapz(SabDiff * SabSolid_convol, betaDiff)
        SabComb_alpha[ib] += s
            
    return SabComb_alpha

@nb.jit(nopython=True)
def integrate_trapz(y, x):
    return 0.5 * ((x[1:] - x[:-1]) * (y[1:] + y[:-1])).sum()

@nb.jit(nopython=True)
def interp1(x1, y1, x2, y2, x, mode):
    '''interpolation of one point according to different cases
    (x1, y1) and (x2, y2) are two points, (x, y) in which y is to
    be determined by interpolation.
    '''
    #assert x >= x1 and x <= x2 # to be verified 
    
    if mode == 'const' or y1 == y2 or x == x1:         # y constant
        return y1
    elif mode == 'linlin':                             # y linear in x
        assert x1 < x2
        return y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    elif mode == 'loglin':                             # y linear in ln(x)
        assert x1 < x2
        assert x1 > 0
        return y1 + math.log(x / x1) * (y2 - y1) / math.log(x2 / x1)
    elif mode == 'linlog':                             # ln(y) linear in x
        assert x1 < x2
        assert y1 > 0 and y2 > 0
        return y1 * math.exp((x - x1) * math.log(y2 / y1) / (x2 - x1))
    elif mode == 'loglog':                             # ln(y) linear in ln(x)
        assert x1 < x2
        assert x1 > 0
        assert y1 > 0 and y2 > 0
        return y1 * math.exp(math.log(x / x1) * math.log(y2 / y1) / math.log(x2 / x1))
    else:
        raise ValueError('Undefined interpolation mode, please check.')


@nb.jit(nopython=True, nogil=False, cache=True, parallel=True)
def get_alpha(Eout: np.array, Ein: np.array, T: np.array, M: np.array,
              mu: np.array) -> np.array:
    """
    Get all the posible alpha values from the parameters of the function:
    .. math::
        \alpha = \frac{E^\prime + E - 2 \mu\sqrt{E^\prime E}}{Ak_BT}

    Parameters
    ----------
    Eout : 1D iterable
        Output energy of the neutron.
    Ein : 1D iterable
        Incidente energy of the neutron.
    T : 1D iterable
        Temperature in K.
    M : "float"
        Mass in amu of the scatterer.
    mu : 1D iterable
        Cosine of the scattering angle.

    Returns
    -------
    "np.array"
        Array containing all posible alpha values for the input parameters.
    """
    alpha = []
    for i in prange(len(T)):
        for j in prange(len(Ein)):
            for k in prange(len(Eout)):
                for ll in prange(len(mu)):
                    alpha_value = Eout[k] + Ein[j]
                    alpha_value -= 2 * mu[ll] * np.sqrt(Eout[k] * Ein[j])
                    alpha_value /= (M * kb * T[i] / m)
                    alpha.append(alpha_value)
    return np.array(alpha)


@nb.jit(nopython=True, nogil=False, parallel=True, cache=True)
def get_beta(Eout: np.array, Ein: np.array, T: np.array) -> np.array:
    """
    Get all the posible beta values from the parameters of the function:
    .. math::
        \beta=\dfrac{E_{out} - E_{in}}{k_BT}

    Parameters
    ----------
    Eout : 1D iterable
        Output energy of the neutron.
    Ein : 1D iterable
        Incidente energy of the neutron.
    T : 1D iterable
        Temperature in K.

    Returns
    -------
    "np.array"
        Array containing all posible beta values for the input parameters.
    """
    beta = []
    for i in prange(len(T)):
        for j in prange(len(Ein)):
            for k in prange(len(Eout)):
                beta_value = Eout[k] - Ein[j]
                beta_value /= kb * T[i]
                beta.append(beta_value)
    return np.array(beta)


@nb.jit(nopython=True, nogil=False, cache=False, parallel=True)
def get_S_fgm_from_alpha_beta(alpha: np.array, beta: np.array,
                              wt:float) -> np.array:
    """
    Get the S(alpha, beta) matrix values using Free Gas Model.
    .. math::
        S(\alpha,\beta)=\dfrac{1}{\sqrt{4\pi w_t\alpha}}\exp\left(-\dfrac{(w_t\alpha+\beta)^2}{4w_t\alpha}\right)

    Parameters
    ----------
    alpha : 1D iterable
        alpha grid values.
    beta : 1D iterable
        beta grid values.
    w_t: 'float', optional
        normalization for continuous (vibrational) part. For solid is 1.

    Returns
    -------
    "np.array"
        S(alpha, beta) matrix values.
    """
    Sab = np.zeros((len(alpha), len(beta)))
    for i in prange(len(alpha)):
        for j in prange(len(beta)):
            Sab[i, j] = np.exp(-(wt * alpha[i] + beta[j]) ** 2 / (4 * wt * alpha[i]))
            Sab[i, j] /= np.sqrt(4 * np.pi * alpha[i] * wt)
    return Sab


@nb.jit(nopython=True, nogil=False, cache=True, parallel=True)
def get_S_fgm_from_parameters(Eout: np.array, Ein: float, T: float, M: float,
                              theta: np.array, wt: float) -> np.array:
    """
    Generate the S(alpha, beta) matrix values using Free Gas Model from base
    parameters:
    .. math::
        \beta=\dfrac{E_{out} - E_{in}}{k_BT}
        \alpha = \frac{E^\prime + E - 2 \mu\sqrt{E^\prime E}}{Ak_BT}
        S(\alpha,\beta)=\dfrac{1}{\sqrt{4\pi w_t\alpha}}\exp\left(-\dfrac{(w_t\alpha+\beta)^2}{4w_t\alpha}\right)

    Parameters
    ----------
    Eout : 1D iterable or 'float'
        Neutron output energies in eV.
    Ein : 'float'
        Neutron incident energy in eV.
    T : 'float'
        Temperature in Kelvin.
    M : 'float'
        Atom mass, amu
    theta : 1D iterable or 'float'
        scattering angle in Degrees.
    w_t: 'float', optional
        normalization for continuous (vibrational) part. For solid is 1.

    Returns
    -------
    "np.array"
        S(theta, Eout) matrix values.
    """
    Sab = np.zeros((len(theta), len(Eout)))
    for j in prange(len(theta)):
        for i in prange(len(Eout)):
            beta = get_beta(Eout[i], Ein, T)[0]
            alpha = get_alpha(Eout[i], Ein, T, M, np.cos(theta[j]))[0]
            Sab[j, i] = get_S_fgm_from_alpha_beta(alpha, beta, wt)
    return Sab


@nb.jit(nopython=True, nogil=False, cache=False, parallel=True)
def get_S_sct_from_alpha_beta(alpha: np.array, beta: np.array, Tratio: float,
                              ws:float) -> np.array:
    """
    Generate S(alpha, beta) matrix using Short Collision Time:
    .. math::
        S(\alpha, \beta)=\dfrac{1}{\sqrt{4\pi\omega_{s}\alpha T_{\textrm{eff}}/T}}\exp\left(-\dfrac{(\mid\beta\mid - \omega_{s}\alpha)^2}{4\omega_{s}\alpha T_{\textrm{eff}}/T} - \frac{\mid\beta\mid - \beta}{2}\right)

    Parameters
    ----------
    alpha : 1D iterable
        alpha grid values.
    beta : 1D iterable
        beta grid values
    Tratio : "float"
        Effective temperature divide by the temperature.
    ws: 'float', optional
        normalization for continuous (vibrational) part. For solid is 1.

    Returns
    -------
    "np.array"
        S(alpha, beta) matrix values.
    """
    Sab = np.zeros((len(alpha), len(beta)))
    for i in prange(len(alpha)):
        for j in prange(len(beta)):
            Sab[i, j] = np.exp(-(abs(beta[j]) - alpha[i] * ws) ** 2 / (4 * alpha[i] * ws * Tratio))
            Sab[i, j] *= np.exp(- (abs(beta[j]) + beta[j]) / 2)
            Sab[i, j] /= np.sqrt(4 * np.pi * ws * alpha[i] * Tratio)
    return Sab


@nb.jit(nopython=True, nogil=False, cache=True, parallel=True)
def get_S_sct_from_parameters(Eout: np.array, Ein: float, T: float, M: float,
                              theta: np.array, ws: float,
                              Teff: float) -> np.array:
    """
    Generate the S(alpha, beta) matrix values using Short Collision Time from
    base parameters:
    .. math::
        \beta=\dfrac{E_{out} - E_{in}}{k_BT}
        \alpha = \frac{E^\prime + E - 2 \mu\sqrt{E^\prime E}}{Ak_BT}
        S(\alpha, \beta)=\dfrac{1}{\sqrt{4\pi\omega_{s}\alpha T_{\textrm{eff}}/T}}\exp\left(-\dfrac{(\mid\beta\mid - \omega_{s}\alpha)^2}{4\omega_{s}\alpha T_{\textrm{eff}}/T} - \frac{\mid\beta\mid - \beta}{2}\right)

    Parameters
    ----------
    Eout : 1D iterable or 'float'
        Neutron output energies in eV.
    Ein : 'float'
        Neutron incident energy in eV.
    T : 'float'
        Temperature in Kelvin.
    M : 'float'
        Atom mass, amu
    theta : 1D iterable or 'float'
        scattering angle in Degrees.
    ws : 'float', optional
        normalization for continuous (vibrational) part. For solid is 1.
    Teff : "float"
        Effective temperature.

    Returns
    -------
    "np.array"
        S(theta, Eout) matrix values.
    """
    Sab = np.zeros((len(theta), len(Eout)))
    Tratio = Teff / T
    for j in prange(len(theta)):
        for i in prange(len(Eout)):
            beta = get_beta(Eout[i], Ein, T)[0]
            alpha = get_alpha(Eout[i], Ein, T, M, np.cos(theta[j]))[0]
            Sab[j, i] = get_S_sct_from_alpha_beta(alpha, beta, Tratio, ws)
    return Sab
