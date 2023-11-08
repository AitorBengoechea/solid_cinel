# -*- coding: utf-8 -*-
"""
Python file for working with the functions for getting clm.

@author: AB272525
"""
import numpy as np
import numba as nb
from math import exp, sqrt, pi
from numba import prange
from scipy.constants import physical_constants as const
from solid_cinel.core.material.vibration.pdos import Pdos


# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]

# Avoid numba fast math:
nb.config.FASTMATH_DEFAULT = False

# Examples variables:
rho_in_energy_str = '''
    0 .0066 .0264 .0594 .1055 .1649 .2374 .3232 .4221
    .5342 .6595 .7980 .9497 1.1146 1.2927 1.4839 1.6884
    2.0169 2.4373 2.9366 3.6133 4.6775 7.1346 7.3650
    7.5156 7.6733 7.8309 8.0740 8.4419 9.0595 9.6773
    7.3645 6.2674 5.1965 4.7958 4.8024 4.6841 4.4673
    4.1914 3.8169 3.3439 2.7855 3.2782 5.3082 8.5930
    12.3377 8.4616 5.6695 4.1585 2.6081 0.0
'''
rho_in_energy = np.fromstring(rho_in_energy_str, dtype=np.float64, sep=' ')
interv_in_energy = 0.0008
alpha0_str = '''
  .005 .010 .015 .020 .025 .030 .035 .040 .045 .050
  .060 .070 .080 .090 .100 .125 .150 .175 .200 .225
  .250 .275 .300 .325 .350 .375 .400 .425 .450 .475
  .500 .525 .550 .575 .600 .625 .675 .700 .725 .750
  .800 .850 .900 .950 1.00 1.05 1.10 1.15 1.20 1.25
  1.30 1.35 1.40 1.50 1.60 1.70 1.80 1.90 2.00 2.10
  2.20 2.30 2.40 2.50 2.60 2.70 2.80 2.90 3.00 3.10
  3.20 3.30 3.40 3.50 3.60 3.80 4.00 4.20 4.40 4.60
  4.80 5.00 5.20 5.40 5.60 5.80 6.00 6.20 6.40 6.60
  6.80 7.00 7.40 7.80 8.20 8.60 9.00 9.40 9.80 10.2
  10.6 11.0 11.5 12.0 12.5 13.0 13.5 14.0 14.5 15.0
  15.5 16.0 16.5 17.0 17.5 18.0 18.5 19.0 19.5 20.0
  21.0 22.0 23.0 24.0 24.5 25.0 26.0 27.0 28.0 29.0
  30.0 32.5 35.0 37.5 40.0 42.5 45.0 47.5 50.0 52.5
  55.0 57.5 60.0 62.5 65.0 67.5 70.0 72.5 75.0
'''
alpha0_ = np.fromstring(alpha0_str, dtype=np.float64, sep=' ')
beta0_str = '''
  .000 .025 .050 .075 .100 .125 .150 .175 .200 .225
  .250 .275 .300 .325 .350 .375 .400 .425 .450 .475
  .500 .525 .550 .575 .600 .625 .650 .675 .700 .725
  .750 .775 .800 .825 .850 .875 .900 .925 .950 .975
  1.00 1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40 1.45
  1.50 1.55 1.60 1.70 1.80 1.90 2.00 2.10 2.20 2.30
  2.40 2.50 2.60 2.70 2.80 2.90 3.00 3.10 3.20 3.30
  3.40 3.50 3.60 3.70 3.80 3.90 4.00 4.10 4.20 4.30
  4.40 4.50 4.60 4.70 4.80 4.90 5.00 5.10 5.20 5.30
  5.40 5.50 5.60 5.70 5.80 5.90 6.00 6.25 6.50 6.75
  7.00 7.50 8.00 8.50 9.00 10.0 11.0 12.0 13.0 14.0
  15.0 16.0 17.0 18.0 19.0 20.0 22.5 25.0 27.5 30.0
  32.5 35.0 37.5 40.0 42.5 45.0 47.5 50.0 52.5 55.0
  57.5 60.0 62.5 65.0 67.5 70.0 72.5 75.0 77.5 80.0
  82.5 85.0 87.5 90.0
'''
beta0_ = np.fromstring(beta0_str, dtype=np.float64, sep=' ')

alpha0_str_U238 = '''
 1.14731156e-04 1.22909925e-04 1.31671728e-04 1.41058129e-04
 1.51113653e-04 1.61885998e-04 1.73426265e-04 1.85789196e-04
 1.99033435e-04 2.13221808e-04 2.28421619e-04 2.44704969e-04
 2.62149101e-04 2.80836761e-04 3.00856598e-04 3.22303575e-04
 3.45279431e-04 3.69893152e-04 3.96261495e-04 4.24509543e-04
 4.54771292e-04 4.87190291e-04 5.21920324e-04 5.59126134e-04
 5.98984212e-04 6.41683629e-04 6.87426932e-04 7.36431110e-04
 7.88928618e-04 8.45168483e-04 9.05417485e-04 9.69961421e-04
 1.03910646e-03 1.11318060e-03 1.19253521e-03 1.27754673e-03
 1.36861841e-03 1.46618227e-03 1.57070109e-03 1.68267069e-03
 1.80262219e-03 1.93112460e-03 2.06878749e-03 2.21626386e-03
 2.37425329e-03 2.54350521e-03 2.72482249e-03 2.91906522e-03
 3.12715481e-03 3.35007836e-03 3.58889332e-03 3.84473254e-03
 4.11880960e-03 4.41242463e-03 4.72697041e-03 5.06393902e-03
 5.42492890e-03 5.81165245e-03 6.22594411e-03 6.66976913e-03
 7.14523283e-03 7.65459062e-03 8.20025868e-03 8.78482543e-03
 9.41106384e-03 1.00819445e-02 1.08006498e-02 1.15705891e-02
 1.23954145e-02 1.32790387e-02 1.42256533e-02 1.52397487e-02
 1.63261352e-02 1.74899663e-02 1.87367627e-02 2.00724386e-02
 2.15033301e-02 2.30362247e-02 2.46783938e-02 2.64376272e-02
 2.83222699e-02 3.03412620e-02 3.25041807e-02 3.48212861e-02
 3.73035696e-02 3.99628060e-02 4.28116098e-02 4.58634945e-02
 4.91329370e-02 5.26354463e-02 5.63876367e-02 6.04073072e-02
 6.47135255e-02 6.93267185e-02 7.42687693e-02 7.95631210e-02
 8.52348880e-02 9.13109746e-02 9.78202035e-02 1.04793452e-01
 1.12263798e-01 1.20266677e-01 1.28840053e-01 1.38024594e-01
 1.47863868e-01 1.58404548e-01 1.69696635e-01 1.81793694e-01
 1.94753108e-01 2.08636352e-01 2.23509283e-01 2.39442450e-01
 2.56511436e-01 2.74797208e-01 2.94386506e-01 3.15372255e-01
 3.37854001e-01 3.61938390e-01 3.87739668e-01 4.15380225e-01
 4.44991178e-01 4.76712988e-01 5.10696131e-01 5.47101810e-01
 5.86102718e-01 6.27883859e-01 6.72643427e-01 7.20593742e-01
 7.71962261e-01 8.26992657e-01 8.85945970e-01 9.49101851e-01
 1.01675989e+00 1.08924102e+00 1.16688907e+00 1.25007237e+00
 1.33918550e+00 1.43465119e+00 1.53692228e+00 1.64648390e+00
 1.76385578e+00 1.88959467e+00 2.02429704e+00 2.16860185e+00
 2.32319362e+00 2.48880569e+00 2.66622364e+00 2.85628907e+00
 3.05990358e+00 3.27803303e+00
'''
beta0_str_U238 = '''
0.00000000e+00 2.57878269e-02 5.15756538e-02 7.73634807e-02
 1.03151308e-01 1.28939135e-01 1.54726961e-01 1.80514788e-01
 2.06302615e-01 2.32090442e-01 2.57878269e-01 2.83666096e-01
 3.09453923e-01 3.35241750e-01 3.61029577e-01 3.86817404e-01
 4.12605231e-01 4.38393058e-01 4.64180884e-01 4.89968711e-01
 5.15756538e-01 5.41544365e-01 5.67332192e-01 5.93120019e-01
 6.18907846e-01 6.44695673e-01 6.70483500e-01 6.96271327e-01
 7.22059154e-01 7.47846980e-01 7.73634807e-01 7.99422634e-01
 8.25210461e-01 8.50998288e-01 8.76786115e-01 9.02573942e-01
 9.28361769e-01 9.54149596e-01 9.79937423e-01 1.00572525e+00
 1.03151308e+00 1.05730090e+00 1.08308873e+00 1.10887656e+00
 1.13466438e+00 1.16045221e+00 1.18624004e+00 1.21202786e+00
 1.23781569e+00 1.26360352e+00 1.28939135e+00 1.31517917e+00
 1.34096700e+00 1.36675483e+00 1.39254265e+00 1.41833048e+00
 1.44411831e+00 1.46990613e+00 1.49569396e+00 1.52148179e+00
 1.54726961e+00 1.57305744e+00 1.59884527e+00 1.62463310e+00
 1.65042092e+00 1.67620875e+00 1.70199658e+00 1.72778440e+00
 1.75357223e+00 1.77936006e+00 1.80514788e+00 1.83093571e+00
 1.85672354e+00 1.88251136e+00 1.90829919e+00 1.93408702e+00
 1.95987485e+00 1.98566267e+00 2.01145050e+00 2.03723833e+00
 2.06302615e+00 2.08881398e+00 2.11460181e+00 2.14038963e+00
 2.16617746e+00 2.19196529e+00 2.21775311e+00 2.24354094e+00
 2.26932877e+00 2.29511660e+00 2.32090442e+00 2.34669225e+00
 2.37248008e+00 2.39826790e+00 2.42405573e+00 2.44984356e+00
 2.47563138e+00 2.50141921e+00 2.52720704e+00 2.55299486e+00
 2.57878269e+00 2.60457052e+00 2.63035835e+00 2.65614617e+00
 2.68193400e+00 2.70772183e+00 2.73350965e+00 2.75929748e+00
 2.78508531e+00 2.81087313e+00 2.83666096e+00 2.86244879e+00
 2.88823661e+00 2.91402444e+00 2.93981227e+00 2.96560009e+00
 2.99138792e+00 3.01717575e+00 3.04296358e+00 3.06875140e+00
 3.09453923e+00 3.26083370e+00 3.43606452e+00 3.62071189e+00
 3.81528185e+00 4.02030762e+00 4.23635107e+00 4.46400427e+00
 4.70389111e+00 4.95666900e+00 5.22303068e+00 5.50370611e+00
 5.79946450e+00 6.11111635e+00 6.43951577e+00 6.78556272e+00
 7.15020557e+00 7.53444360e+00 7.93932983e+00 8.36597386e+00
 8.81554490e+00 9.28927500e+00 9.78846243e+00 1.03144752e+01
 1.08687549e+01 1.14528205e+01 1.20682726e+01 1.27167979e+01
 1.34001737e+01 1.41202728e+01 1.48790686e+01 1.56786406e+01
 1.65211800e+01 1.74089958e+01 1.83445212e+01 1.93303198e+01
 2.03690933e+01 2.14636884e+01 2.26171050e+01 2.38325039e+01
 2.51132159e+01 2.64627509e+01 2.78848073e+01 2.93832822e+01
 3.09622822e+01 3.26261345e+01 3.43793991e+01 3.62268806e+01
 3.81736422e+01 4.02250189e+01 4.23866326e+01 4.46644071e+01
 4.70645848e+01 4.95937434e+01 5.22588139e+01 5.50671002e+01
 5.80262982e+01 6.11445178e+01 6.44303043e+01 6.78926626e+01
 7.15410812e+01 7.53855587e+01 7.94366309e+01 8.37053998e+01
 8.82035639e+01 9.29434505e+01 9.79380494e+01 1.03201048e+02
 1.08746870e+02 1.14590714e+02 1.20748594e+02 1.27237387e+02
 1.34074875e+02 1.41279796e+02 1.48871895e+02 1.56871979e+02
 1.65301972e+02 1.74184976e+02 1.83545335e+02 1.93408702e+02
'''
alpha0_U238 = np.fromstring(alpha0_str_U238, dtype=np.float64, sep=' ')
beta0_U238 = np.fromstring(beta0_str_U238, dtype=np.float64, sep=' ')

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


@nb.jit(nopython=True, nogil=True, cache=True, parallel=False)
def get_S_from_tau_n(tau: np.ndarray, beta_tau: np.ndarray,
                     debye_waller_coeff: float,  iter_sum: float,
                     alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Generate S(alpha, -beta) matrix using phonon expansion tau_n function.
    .. math::
        S(\alpha,\,-\beta)=\exp(-\alpha\lambda)\sum_{n=0}^{\infty}\dfrac{1}{n!}(\alpha\lambda)^n\mathcal{T}_n(-\beta)

    Numerical appoximation to get convergence in large exponentiation and
    factorial numbers. Each element of the array is related with one alpha
    and represent the following term of the previous equation:
    ..math::
        \sum_{n=0}^{\infty}\dfrac{1}{n!}(\alpha\lambda)^n = \exp(\log(\dfrac{1}{1}(\alpha\lambda)) + \log(\dfrac{1}{2}(\alpha\lambda)) + ...)

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
    beta: 1D iterable, (M,)
        beta grid.

    Returns
    -------
    'np.ndarray', (N, M)
        S(alpha, beta) matrix values for the n phonon expansion using tau_n.

    Example
    -------
    >>> import pandas as pd
    >>> from solid_cinel.core.material.scattering_function.beta import Beta
    >>> from solid_cinel.core.material.scattering_function.alpha import Alpha
    >>> T = 800
    >>> pdos = Pdos.from_dE(rho_in_energy, interv_in_energy)
    >>> tau1 = pdos.get_tau_1(T)
    >>> debye_waller_coeff = pdos.DebyeWallerCoeff(T)
    >>> alpha_grid = Alpha(alpha0_).scale(T).data
    >>> beta_grid = Beta(beta0_).scale(T).data
    >>> iter_sum = np.log(alpha_grid * debye_waller_coeff)
    >>> S_mat = get_S_from_tau_n(tau1.values, np.arange(len(tau1)) * beta_grid[1], debye_waller_coeff, iter_sum, alpha_grid, beta_grid)
    >>> pd.DataFrame(S_mat.round(6)).iloc[:10, :5] #doctest: +NORMALIZE_WHITESPACE
              0         1         2         3         4
    0  0.036967  0.037182  0.037398  0.037614  0.037796
    1  0.070694  0.071105  0.071517  0.071932  0.072279
    2  0.101393  0.101982  0.102574  0.103168  0.103666
    3  0.129265  0.130016  0.130771  0.131528  0.132163
    4  0.154499  0.155397  0.156299  0.157204  0.157963
    5  0.177272  0.178303  0.179337  0.180376  0.181247
    6  0.197752  0.198902  0.200056  0.201215  0.202186
    7  0.216097  0.217353  0.218614  0.219880  0.220942
    8  0.232453  0.233804  0.235161  0.236523  0.237664
    9  0.246960  0.248396  0.249837  0.251284  0.252497
    """
    alpha_mul = np.exp(- alpha * debye_waller_coeff + iter_sum)
    # Interpolate tau_n(-beta):
    tau_n_reshape = np.interp(beta, beta_tau, tau)
    # Bounds in nopython mode:
    if beta[-1] > beta_tau[-1]:
        tau_n_reshape[beta > beta_tau[-1]] = 0.0
    return np.outer(alpha_mul, tau_n_reshape)


@nb.jit(nopython=True, nogil=True, cache=True, parallel=False)
def get_S_pdos_from_alpha_beta(alpha: np.ndarray, beta: np.ndarray,
                               nphonon: int, tau1: np.ndarray, delta_beta: float,
                               threshold: float,
                               DebyeWallerCoeff: float) -> np.ndarray:
    """
    Generate S(alpha, -beta) matrix using phonon expansion.
    .. math::
        S(\alpha,\,-\beta)=\exp(-\alpha\lambda)\sum_{n=0}^{\infty}\dfrac{1}{n!}(\alpha\lambda)^n\mathcal{T}_n(-\beta)

    Numerical appoximation to get convergence in large exponentiation and
    factorial numbers. Each element of the array is related with one alpha
    and represent the following term of the previous equation:
    ..math::
        \sum_{n=0}^{\infty}\dfrac{1}{n!}(\alpha\lambda)^n = \exp(\log(\dfrac{1}{1}(\alpha\lambda)) + \log(\dfrac{1}{2}(\alpha\lambda)) + ...)

    Parameters
    ----------
    alpha : 'np.ndarray', (N,)
        alpha grid values.
    beta : 'np.ndarray', (M,)
        beta grid values.
    nphonon : 'int', optional
        Phonon expansion order.
    tau1 : 'np.ndarray', (Z,)
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
    'np.ndarray', (N, M)
        S(alpha, beta) matrix values

    Example
    -------
    >>> import pandas as pd
    >>> from solid_cinel.core.material.scattering_function.beta import Beta
    >>> from solid_cinel.core.material.scattering_function.alpha import Alpha
    >>> T = 800
    >>> pdos = Pdos.from_dE(rho_in_energy, interv_in_energy)
    >>> tau1 = pdos.get_tau_1(T)
    >>> debye_waller_coeff = pdos.DebyeWallerCoeff(T)
    >>> alpha_grid = Alpha(alpha0_).scale(T).data
    >>> beta_grid = Beta(beta0_).scale(T).data
    >>> S_mat = get_S_pdos_from_alpha_beta(alpha_grid, beta_grid, 10, tau1.values, beta_grid[1], 1.0e-14, debye_waller_coeff)
    >>> pd.DataFrame(S_mat.round(6)).iloc[:10, :5] #doctest: +NORMALIZE_WHITESPACE
              0         1         2         3         4
    0  0.037829  0.038039  0.038243  0.038444  0.038611
    1  0.074018  0.074411  0.074776  0.075133  0.075422
    2  0.108603  0.109154  0.109645  0.110117  0.110490
    3  0.141624  0.142311  0.142895  0.143444  0.143867
    4  0.173120  0.173922  0.174570  0.175165  0.175607
    5  0.203131  0.204030  0.204716  0.205328  0.205763
    6  0.231696  0.232675  0.233377  0.233982  0.234387
    7  0.258856  0.259901  0.260599  0.261175  0.261531
    8  0.284651  0.285748  0.286425  0.286954  0.287244
    9  0.309121  0.310260  0.310900  0.311366  0.311575
    """
    tau_n_minus_1 = tau1.copy()
    # Zero phonon expansion:
    iter_sum = np.log(alpha * DebyeWallerCoeff)
    S_values = get_S_from_tau_n(tau1, np.arange(len(tau1)) * delta_beta,
                                DebyeWallerCoeff, iter_sum, alpha, beta)

    # Higher phonon expansion (nphonon >= 1):
    for n in range(1, nphonon + 1):
        # Tau_n(-beta)
        tau_n = tau_n_CPU(delta_beta, tau1, tau_n_minus_1, threshold)
        check_tau_n(tau_n, delta_beta)

        # Compute S(alpha, -beta) for tau_n reshape
        iter_sum += np.log(alpha * DebyeWallerCoeff / (n + 1))
        S_values += get_S_from_tau_n(tau_n, np.arange(len(tau_n)) * delta_beta,
                                     DebyeWallerCoeff, iter_sum, alpha, beta)

        # Next tau_n
        tau_n_minus_1 = tau_n
    return S_values


@nb.jit(nopython=True, nogil=True, cache=True, parallel=True)
def get_S_sct_from_alpha_beta(alpha: np.ndarray, beta: np.ndarray,
                              Tratio: float,
                              ws: float) -> np.ndarray:
    """
    Generate S(alpha, beta) matrix using Short Collision Time:
    .. math::
        S(\alpha, \beta)=\dfrac{1}{\sqrt{4\pi\omega_{s}\alpha T_{\textrm{eff}}/T}}\exp\left(-\dfrac{(\mid\beta\mid - \omega_{s}\alpha)^2}{4\omega_{s}\alpha T_{\textrm{eff}}/T} - \frac{\mid\beta\mid - \beta}{2}\right)

    Parameters
    ----------
    alpha : 'np.ndarray', (N,)
        alpha grid values.
    beta : 'np.ndarray', (M,)
        beta grid values.
    Tratio : "float"
        Effective temperature divide by the temperature.
    ws: 'float', optional
        normalization for continuous (vibrational) part. For solid is 1.

    Returns
    -------
    'np.ndarray', (N, M)
        S(alpha, beta) matrix values.
    """
    Sab = np.empty((len(alpha), len(beta)))
    for i in prange(len(alpha)):
        for j in prange(len(beta)):
            Sab[i, j] = exp(-(abs(beta[j]) - alpha[i] * ws) ** 2 / (4 * alpha[i] * ws * Tratio))
            Sab[i, j] *= exp(- (abs(beta[j]) + beta[j]) / 2)
            Sab[i, j] /= sqrt(4 * pi * ws * alpha[i] * Tratio)
    return Sab


@nb.jit("float64[:](float64, float64[:], float64[:], float64)",
    nopython=True, nogil=True, cache=True, parallel=True)
def tau_n_CPU(delta_beta: float, tau1: np.ndarray, tau_n_minus_1: np.ndarray,
              threshold: float) -> np.ndarray:
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
    tau_n = np.zeros(len(tau1) + len(tau_n_minus_1) - 1)
    Nnm1 = len(tau_n_minus_1)  # length of tau_n_minus_1
    N = len(tau1)

    for i in prange(len(tau_n)):  # loop for tau_n
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
                convol *= 0.5                      # trapz integrate

            tau_n[i] += tau1[j] * convol * delta_beta

    return tau_n if threshold == 0.0 else tau_n[tau_n >= threshold]


@nb.jit('(float64[:], float64)',
    nopython=True, cache=True)
def check_tau_n(tau_n: np.ndarray, delta_beta: float) -> None:
    """
    Check if the tau function created in solid_cinel.core._numba.tau_n_CPU is
    normalized to the unity.

    Parameters
    ----------
    tau_n : 1D iterable, (N,)
        tau_n function values.
    delta_beta : float
        Space between beta grid points.

    Returns
    -------
    "None"
        If the normalization is not satisfied with good accuracy a warning
        is raise. If the accuracy is very low, a ValueError will be raise.

    Raises
    ------
    ValueError
        Tau function doesnt satisfy the normalization condition.
    """
    if np.trapz(tau_n, dx=delta_beta) < 1.e-5:
        raise ValueError("Tau function doesnt satisfy the normalization condition")
    return


@nb.jit("float64[:, :](float64[:], float64[:], float64, float64, float64)",
        nopython=True, nogil=True, cache=True)
def get_ScatFunc_values(Sab_diag: np.ndarray, beta_grid: np.ndarray, Ein: float,
                        T: float, M: float) -> np.ndarray:
    """
    Generate the scattering function values from a S(alpha, -beta) table based on
    the phonon expansion model for a single angle

    Parameters
    ----------
    Sab_diag : 'np.ndarray', (N,)
        S(alpha, -beta) matrix diagonal values.
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
    # Scattering function values calculation:
    ScatFunc_values = np.concatenate((Sab_diag[::-1], Sab_diag[1::]))
    ScatFunc_values[len(Sab_diag)::] *= np.exp(-beta_grid[1::])

    # Eout calculation
    Eout = np.sort(
        Ein + np.concatenate((-beta_grid[::-1], beta_grid[1::])) * kb * T)

    # Ensure the Eout values are positive:
    positive_mask = Eout > 0
    ScatFunc_values = ScatFunc_values[positive_mask]
    Eout = Eout[positive_mask]

    # Handle nan values:
    ScatFunc_values[np.isnan(ScatFunc_values)] = 0

    # Normalization constant
    aws = ((M / m + 1) / (M / m)) ** 2
    normalization_factor = aws * np.sqrt(Eout / Ein) / (2 * kb * T)

    return np.vstack((Eout, ScatFunc_values * normalization_factor)).T


@nb.jit("float64[:](float64[:], float64, float64, float64)",
    nopython=True, nogil=True, cache=True)
def sigma1(Eout: np.array, Ein: float, T: float, M: float) -> np.array:
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

    Examples
    --------
    >>> import pandas as pd
    >>> Ein = 7.2
    >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> pd.Series(sigma1(Eout, Ein, T, M), index=Eout).round(6)
    6.7554    0.000000
    6.9050    0.001153
    7.0439    0.522804
    7.2000    5.501786
    7.3157    1.568599
    7.4480    0.017808
    dtype: float64
    """
    exp_negative = np.exp(
        - M / (m * kb * T) * (sqrt(Ein) - np.sqrt(Eout)) ** 2)
    exp_positive = np.exp(
        - M / (m * kb * T) * (sqrt(Ein) + np.sqrt(Eout)) ** 2)
    scattfunc = 0.5 * (exp_negative - exp_positive) * np.sqrt(Eout) / Ein
    scattfunc *= sqrt(M / (pi * m * kb * T))
    return scattfunc


@nb.jit("float64[:](float64[:], float64, float64, float64, float64, float64, float64)",
    nopython=True, nogil=True, cache=True, parallel=False)
def get_scat_sct_angular(Eout: np.ndarray, mu: float, Ein: float, T: float,
                M: float, Teff: float, ws: float) -> np.array:
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
    awr = ((M / m + 1) / (M / m)) ** 2
    beta = (Eout - Ein) / (kb * T)
    alpha = Eout + Ein - 2 * mu * np.sqrt(Eout * Ein)
    alpha /= (M * kb * T / m)
    scattfunc = np.exp(-(ws * alpha + beta) ** 2 / (4 * alpha * Teff / T * ws))
    scattfunc /= np.sqrt(4 * pi * ws * alpha * Teff / T)
    scattfunc *= awr * np.sqrt(Eout / Ein) / (2 * kb * T)
    return scattfunc


def scat_from_pdos(Ein: float, M: float, T: float, Eout: np.array,
                       theta: np.array, pdos: Pdos, threshold: float = 0.0,
                       nphonon: int = 1000) -> list:
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
    Eout : np.array
        The neutron outgoing energy grid in eV
    theta : np.array
        Grid of cosine of the scattering angle
    pdos : 'solid_cinel.core.material.Pdos'
        Pdos object.
    threshold : 'float', optional
        Minimun value to take into account in the creation of tau_n
        functions. For T>200 is convenient to set into 1.0e-14 to speed up
        the calculations. The default is 0.0.
    nphonon : 'int', optional
        Phonon expansion order. The default is 1000.

    Returns
    -------
    dd_pdf : dict
        Dictionary with the scattering function for each angle

    Examples
    --------
    >>> import pandas as pd
    >>> Ein = 7.2
    >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
    >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> theta = np.array([40, 80, 120, 160])
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> dd_pdf = scat_from_pdos(Ein, M, T, Eout, theta, pdos, threshold=1.0e-14)
    >>> pd.DataFrame(dd_pdf, index=np.cos(np.deg2rad(theta)), columns=Eout).loc[:, Eout_test].round(6)
               6.7554    6.9050    7.0439    7.2000    7.3157    7.4480
     0.766044  0.000000  0.000012  0.077506  4.022814  0.127645  0.000019
     0.173648  0.000519  0.073364  1.103240  1.912878  0.440892  0.013328
    -0.500000  0.034511  0.426488  1.383082  1.262613  0.415630  0.042074
    -0.939693  0.109061  0.644157  1.346118  1.029210  0.373644  0.053219
        """
    dd_pdf = []
    tau1 = pdos.get_tau_1(T)
    debye_waller_coeff = pdos.DebyeWallerCoeff(T)
    for mu in np.cos(np.deg2rad(theta)):
        dd_pdf.append(get_ScatFunc_pdos_angle(Ein, M, T, Eout, mu, nphonon,
                                              tau1.values, tau1.index[1],
                                              threshold, debye_waller_coeff))
    return dd_pdf

@nb.jit("float64[:](float64[:], float64[:], float64, float64[:], float64[:], float64[:])",
    nopython=True, nogil=True, cache=True, parallel=False)
def get_diag_S_from_tau_n(tau: np.ndarray, beta_tau: np.ndarray,
                     debye_waller_coeff: float,  iter_sum: np.ndarray,
                     alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
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
    alpha_mul = np.exp(- alpha * debye_waller_coeff + iter_sum)
    # Interpolate tau_n(-beta):
    tau_n_reshape = np.interp(beta, beta_tau, tau)
    # Bounds in nopython mode:
    if beta[-1] > beta_tau[-1]:
        tau_n_reshape[beta > beta_tau[-1]] = 0.0
    return alpha_mul * tau_n_reshape

@nb.jit("float64[:](float64[:], float64[:], int32, float64[:], float64, float64, float64)",
    nopython=True, nogil=True, cache=True, parallel=False)
def get_diag_S_pdos(alpha: np.ndarray, beta: np.ndarray,
                    nphonon: int, tau1: np.ndarray, delta_beta: float,
                    threshold: float, DebyeWallerCoeff: float) -> np.ndarray:
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
    if len(alpha) != len(beta):
        raise ValueError("alpha and beta must have the same length")

    tau_n_minus_1 = tau1.copy()
    # Zero phonon expansion:
    iter_sum = np.log(alpha * DebyeWallerCoeff)
    beta_tau_1 = np.arange(len(tau1)) * delta_beta
    S_diag = get_diag_S_from_tau_n(tau1, beta_tau_1,
                                DebyeWallerCoeff, iter_sum, alpha, beta)

    # Higher phonon expansion (nphonon >= 1):
    for n in range(1, nphonon + 1):
        # Tau_n(-beta)
        tau_n = tau_n_CPU(delta_beta, tau1, tau_n_minus_1, threshold)
        beta_tau_n = np.arange(len(tau_n)) * delta_beta

        # Compute S(alpha, -beta) for tau_n reshape
        iter_sum += np.log(alpha * DebyeWallerCoeff / (n + 1))
        S_diag += get_diag_S_from_tau_n(tau_n, beta_tau_n,
                                        DebyeWallerCoeff, iter_sum, alpha, beta)

        # Next tau_n
        tau_n_minus_1 = tau_n
    return S_diag

@nb.jit("float64[:](float64, float64, float64, float64[:], float64, int32, float64[:], float64, float64, float64)",
        nopython=True, nogil=True, cache=True, parallel=False)
def get_ScatFunc_pdos_angle(Ein: float, M: float, T: float, Eout: np.ndarray,
                 mu: float, nphonon: int, tau1: np.ndarray,
                 delta_beta: float, threshold: float,
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

    Examples
    --------
    >>> import pandas as pd
    >>> Ein = 7.2
    >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
    >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
    >>> T = 1000
    >>> M = 238.05077040419212
    >>> mu = np.cos(np.deg2rad(120))
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> tau1 = pdos.get_tau_1(T)
    >>> debye_waller_coeff = pdos.DebyeWallerCoeff(T)
    >>> sd_pdf = get_ScatFunc_pdos_angle(Ein, M, T, Eout, mu, 1000, tau1.values, tau1.index[1], 1.0e-14, debye_waller_coeff)
    >>> pd.Series(sd_pdf, index=Eout).loc[Eout_test].round(6)
    6.7554    0.034511
    6.9050    0.426488
    7.0439    1.383082
    7.2000    1.262613
    7.3157    0.415630
    7.4480    0.042074
    dtype: float64
    """
    beta = (Eout - Ein) / (kb * T)
    beta = np.unique(np.absolute(beta))
    if len(beta) < len(Eout): # same beta values but one negative and one positive
        Eout_ = beta * kb * T + Ein
    else:
        Eout_ = Eout.copy()
    alpha = Eout_ + Ein - 2 * mu * np.sqrt(Eout_ * Ein)
    alpha /= (M * kb * T / m)
    Sab_values = get_diag_S_pdos(alpha, beta, nphonon, tau1, delta_beta,
                                 threshold, DebyeWallerCoeff)
    sd_pdf = get_ScatFunc_values(Sab_values, beta, Ein, T, M)
    # Interpolation for avoiding numerical fluctuations:
    return np.interp(Eout, sd_pdf[:, 0], sd_pdf[:, 1])