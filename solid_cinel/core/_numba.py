import numba as nb
import numpy as np
from numba import prange
from numba import cuda
import math


@nb.jit(nopython=True, nogil=True)
def hklloop(d_min, hkl_max, rec_vecs, Bfac, pos, csl, preferred_orientation,
            precision) -> dict:
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
    Bfac : 'pd.Series'
        Pandas series with the B factor for Target_Material object elements.
    pos : 'pd.Series'
        Pandas series with atomic position of elements in Target_Material 
        object.
    csl : 'pd.Series'
        Coherent elastic length for each element of Target_Material object.
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

                # Fsq_hkl
                real = 0.
                imag = 0.
                for element in Bfac:
                    expon_hkl = np.exp(-0.5 * np.linalg.norm(vec_tau_hkl) ** 2
                                       * Bfac[element] / (8 * np.pi ** 2))
                    element_position = pos[element]
                    scalar = np.sum(vec_tau_hkl * element_position, axis=1)
                    cumulant_cos = np.sum(np.cos(scalar))
                    cumulant_sin = np.sum(np.sin(scalar))
                    real += csl[element] * 0.1 * expon_hkl * cumulant_cos
                    imag += csl[element] * 0.1 * expon_hkl * cumulant_sin
                Fsq_hkl = real ** 2 + imag ** 2  # Fsquared

                # same dspacing and Fsquared with precision will be regrouped
                d_rnd = round(d_hkl, precision[0])
                Fsq_rnd = round(Fsq_hkl, precision[1])
                if (d_rnd, Fsq_rnd) in hkldF:
                    hklM[hkldF[(d_rnd, Fsq_rnd)]][-1] += 1
                else:
                    hkldF[(d_rnd, Fsq_rnd)] = (h, k, l)
                    OA_num = np.sum(vec_tau_hkl * preferred_orientation)
                    OA_den = np.linalg.norm(vec_tau_hkl) * np.linalg.norm(preferred_orientation)
                    orientation_angle_hkl = np.arccos(OA_num / OA_den) * 180 / np.pi
                    hklM[(h, k, l)] = np.array([d_hkl, Fsq_hkl, orientation_angle_hkl, 1])
    return hklM

@nb.jit(nopython=True, nogil=True)
def convolTaufunc_CPU(interv_in_beta, Tau1_neg, Taunm1_neg, Taun_neg):
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
    
    N = len(Tau1_neg)                             # length of Tau1_neg
    Nnm1 = len(Taunm1_neg)                        # length of Taunm1_neg
    Nn = len(Taun_neg)                            # length of Taun_neg
    
    for i in prange(Nn):                           # loop for Taun_neg            
        for j in prange(N):                        # loop for Tau1_neg
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

@nb.jit(nopython=True, nogil=True)
def Sabadd(i, Na, Nb, alpha, debye_waller_coeff,
          beta_pip1, beta_Tauip1, Tauip1, SabSolid, xa):
    '''increase of S(alpha, -beta) of (i + 1) phonon expansion
    order
    '''
    
    for ia in prange(Na):
        xa[ia] += math.log(alpha[ia] * debye_waller_coeff / (i + 1))
        ex = math.exp(-alpha[ia] * debye_waller_coeff + xa[ia])
        for ib in prange(len(beta_pip1)): # interpolation of Tau(beta)
            j = np.searchsorted(beta_Tauip1, beta_pip1[ib], side = 'right') # beta_Tauip1[j-1] <= beta_pip1[ib] < beta_Tauip1[j]
            Tau_interp = interp1(beta_Tauip1[j - 1], Tauip1[j - 1],
                                 beta_Tauip1[j], Tauip1[j],
                                 beta_pip1[ib], 'linlin')
            add = Tau_interp * ex # add adopted from LEAPR, but asymmetric form
            SabSolid[ia, ib] += add
    return SabSolid

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