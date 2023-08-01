"""
Python file for working with numba function coming from cinel.

@author: AB272525
"""
import numpy as np
import numba as nb
from numba import prange
from numba import cuda
import math
from scipy.constants import physical_constants as const

kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]


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
def update_Sab_with_tau_n(n: int, alpha_grid: np.ndarray,
                          DebyeWallerCoeff: float, tau_n: np.ndarray,
                          Sab: np.ndarray,
                          iter_sum: np.ndarray) -> tuple:
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
    alpha_grid : 'np.ndarray', (N,)
        alpha values of the matrix S(alpha, -beta).
    DebyeWallerCoeff : 'float'
        Debye Wallelr Coefficient.
    tau_n : 'np.ndarray', (N,)
        tau_n values.
    Sab : 'np.ndarray', (N, M)
        S(alpha, -beta) matrix for n-1 expansion order.
    iter_sum : 'np.ndarray', (n,)
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
