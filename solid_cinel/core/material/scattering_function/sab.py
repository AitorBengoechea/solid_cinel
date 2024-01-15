"""
Python file for working with S(alpha, -beta) matrixs.

@author: AB272525
"""
from scipy.constants import physical_constants as const
from scipy.integrate import trapezoid
from solid_cinel.core.generic import integrate, reshape_differential, gpu_available, optional_jit
from solid_cinel.core.material.vibration.pdos import Pdos
from solid_cinel.core.material.vibration.tau import tau_n_functions, save_tau
from solid_cinel.core.material.scattering_function.beta import Beta
from solid_cinel.core.material.scattering_function.alpha import Alpha
from solid_cinel.core.material.scattering_function.scatfunc import scatfunc_values_alpha_vec
from typing import Iterable, Union
import numpy as np
import pandas as pd
import numba as nb
from math import exp, sqrt, pi
from numba import prange
import warnings
try:
    import cupy as cp
    xp = cp
except ImportError:
    xp = np


kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]

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
3.550530E+00 3.34990.5 / incident_neutron_energy * np.sqrt(M / (np.pi * const["neutron mass in u"][0] * kbT))06960E-01 1.452214E-01 1.246671E-01
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


class Sab:
    """
    Class containing all the methods and properties of a asymmetric
    S(alpha, beta) matrix.

    Attributes
    ----------
    data : pd.DataFrame
        Dataframe with the S(alpha, -beta) matrix values.
    alpha : Alpha
        Initialize the Alpha class with the information of S(alpha, beta)
    beta : Beta
        Initialize the Beta class with the information of S(alpha, beta)

    Methods
    -------
    to_sym -> pd.DataFrame
        Return the symmetric S(alpha, -beta) matrix
    to_full -> pd.DataFrame
        Return the full S(alpha, beta) matrix
    from_fgm -> Sab
        Return the S(alpha, beta) matrix from the FGM model
    from_sct -> Sab
        Return the S(alpha, beta) matrix from the SCT model
    from_pdos -> Sab
        Return the S(alpha, beta) matrix from the PDOS model
    sum_rule_check -> bool
        Check if the sum rule is satisfied
    normalization_check -> bool
        Check if the normalization is satisfied
    get_momentum -> Sab or pd.DataFrame
        Return the S(alpha, beta) matrix n momentum
    get_beta -> Sab or pd.DataFrame
        Quadratic interpolation to get the probability of the new beta value
    get_alpha -> Sab or pd.DataFrame
        Unit base interpolation to get the probability of the new alpha values
    get_value_from_alpha_beta -> pd.DataFrame
        Return the S(alpha, beta) matrix value for a given alpha and beta
    get_matrix_from_parameters -> Sab or pd.DataFrame
        Based on the set of variables introduced, interpolate the existing
        S(alpha, -beta) to make a new S(alpha, beta) matrix with the alpha and
        beta values created from the set of variables
    get_scattering_function -> pd.DataFrame
        Return the scattering function from S(alpha, beta) matrix
    get_inelastic_Xs -> pd.DataFrame
        Return the inelastic cross section from S(alpha, beta) matrix
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the S(alpha, beta) matrix class.

        Parameters
        ----------
        args : "np.array"
            Array containing the S(alpha, -beta) matrix.
        kwargs : "dict"
            Dictionary containing the S(alpha, -beta) matrix.
        """
        self.data = pd.DataFrame(*args, **kwargs)

    @property
    def alpha(self) -> Alpha:
        """
        Initialize the Alpha class with the information of S(alpha, beta)
        matrix.
        """
        return Alpha(self.data.index.values)

    @property
    def beta(self) -> Beta:
        """
        Initialize the Beta class with the information of S(alpha, -beta).
        matrix
        """
        return Beta(self.data.columns.values)

    @property
    def data(self) -> pd.DataFrame:
        """Dataframe with the S(alpha, -beta) matrix values."""
        return self._data

    @data.setter
    def data(self, df: Iterable):
        """
        Construct the S(alpha, -beta) matrix and check if the data achieve the
        normalization and sum rule constrain.

        Parameters
        ----------
        df : 2D iterable, (N, M)
            Iterable containing the S(alpha, -beta) matrix.
        """
        df_ = pd.DataFrame(df).sort_index(axis=0).sort_index(axis=1)
        df_.index.name = "alpha"
        df_.columns.name = "beta"
        # Normalization constrains:
        self.normalization_check(df_)
        self.sum_rule_check(df_)
        # DataFrame:
        self._data = df_

    def to_sym(self, detail_balance: bool = True) -> pd.DataFrame:
        """
        Generate the symmetric S(alpha, -beta) matrix from the asymmetric
        S(alpha, -beta) matrix.

        Parameters
        ----------
        detail_balance : 'bool', optional
            Relationships between upscatter and downscatter. The default is
            True.

        Returns
        -------
        "pd.DataFrame"
            Dataframe containing the symmetric S(alpha, -beta) matrix.

        Example:
        --------
        >>> beta_grid = Beta.generate_grid(300).data
        >>> alpha_grid = Alpha.generate_grid(300, 26).data
        >>> Sab.from_fgm(alpha_grid, beta_grid).to_sym().iloc[:10, :5].round(6) #doctest: +NORMALIZE_WHITESPACE
        beta      0.000000  0.012894  0.025788  0.038682  0.051576
        alpha
        0.001050  8.701463  8.363896  7.427755  6.094516  4.620122
        0.001087  8.553363  8.232522  7.340419  6.063184  4.639515
        0.001125  8.407781  8.102844  7.252800  6.029567  4.655632
        0.001164  8.264674  7.974859  7.164978  5.993787  4.668553
        0.001205  8.124000  7.848564  7.077028  5.955964  4.678361
        0.001247  7.985718  7.723951  6.989018  5.916214  4.685142
        0.001291  7.849787  7.601016  6.901017  5.874652  4.688985
        0.001336  7.716166  7.479752  6.813088  5.831389  4.689983
        0.001382  7.584817  7.360149  6.725291  5.786534  4.688230
        0.001431  7.455701  7.242199  6.637682  5.740191  4.683821
        """
        S_asym = self.data
        if detail_balance:
            beta = self.data.columns.values
            S_sym = S_asym * np.exp(-beta / 2)
        else:
            S_sym = S_asym
        return S_sym

    def to_full(self) -> pd.DataFrame:
        """
        Get the full S(alpha, beta) matrix.

        Returns
        -------
        'pd.DataFrame'
            the full S(alpha, beta) matrix

        Example
        -------
        >>> beta_grid = Beta.generate_grid(300)
        >>> alpha_grid = Alpha.generate_grid(300, 26)
        >>> Sab_matrix = Sab.from_fgm(alpha_grid, beta_grid)
        >>> Sab_matrix_norm = Sab_matrix.data.apply(_normalization, axis=1)
        >>> Sab_matrix_full = Sab_matrix.to_full()
        >>> assert (Sab_matrix_full.apply(integrate, axis=1).round(6) == Sab_matrix_norm.round(6)).all()

        >>> Sab_matrix_sum = Sab_matrix.data.apply(_sum_rule, axis=1)
        >>> assert (Sab_matrix_full.apply(lambda x: - integrate(x * x.index), axis=1).round(6) == Sab_matrix_sum.round(6)).all()
        """
        beta = self.beta.to_index
        Sab_negative = self.data.set_axis(-beta, axis=1).sort_index(axis=1)
        Sab_positive = self.data.apply(lambda x: x * np.exp(-beta), axis=1)
        return pd.concat([Sab_negative, Sab_positive.iloc[::, 1::]], axis=1)

    def to_ScatFunc(self, Ein, T, M, mu=None) -> pd.Series:
        """
        Get the scattering function from the S(alpha, -beta) matrix.

        Parameters
        ----------
        Ein : 'float'
            Incident energy in eV.
        T : 'float'
            Temperature in K.
        mu : 'float', optional
            The Cosine of the scattering angle used for the creation of the
            S(alpha, -beta) table. The default is None.

        Returns
        -------
        'pd.Series'
            Scattering function of these S(alpha, -beta) table

        Example
        -------
        >>> T = 300
        >>> M = 26
        >>> Ein = 3
        >>> Eout = np.linspace(Ein, Ein * 1.05, 1000)
        >>> beta_grid = Beta.from_Eout(Eout, Ein, T).data
        >>> alpha_grid = Alpha.from_parameters(Eout, Ein, T, M, 60).data
        >>> Sab.from_fgm(alpha_grid, beta_grid).to_ScatFunc(Ein, T, M).iloc[295:305].round(6) #doctest: +NORMALIZE_WHITESPACE
        Eout
        2.894294    2.665969
        2.894444    2.665249
        2.894595    2.664520
        2.894745    2.663782
        2.894895    2.663034
        2.895045    2.662277
        2.895195    2.661511
        2.895345    2.660736
        2.895495    2.659952
        2.895646    2.659158
        dtype: float64
        """
        Ein, T, M = float(Ein), float(T), float(M)
        # Get the scattering function values:
        sab_diag = np.array(np.diag(self.data), order='C')
        beta = self.beta.data[:len(sab_diag)]
        Eout_calc, scatfunc_values = scatfunc_values_alpha_vec(sab_diag, beta, Ein, T, M)

        # Change the data type:
        scattfunc = pd.Series(scatfunc_values, index=Eout_calc)
        scattfunc = scattfunc[~scattfunc.index.duplicated(keep='first')]
        # Output style:
        scattfunc.index.name = 'Eout'
        if mu:
            scattfunc.name = mu
        return scattfunc.sort_index()

    @classmethod
    def from_fgm(cls, alpha_grid: Union[Alpha, Iterable],
                 beta_grid: Union[Beta, Iterable], T: float = None,
                 wt: float = 1):
        """
        Generate S(alpha, -beta) matrix using Free Gas Model.
        .. math::
            S_t(\alpha,\,-\beta)=\dfrac{1}{\sqrt{4\pi w_t\alpha}}\exp\left(-\dfrac{(w_t\alpha+\beta)^2}{4w_t\alpha}\right)\end{equation}

        Parameters
        ----------
        alpha_grid : 1D iterable or "Alpha", (N,)
            Alpha grid.
        beta_grid : 1D iterable or "Beta", (M,)
            Absolute beta grid.
        model : 'str', optional
            The model to calculate matrix values. The default is "FGM".
        wt: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Returns
        -------
        "Sab", (N, M)
            S(alpha, -beta) based on Free Gas Model.

        Example
        -------
        FGM:
        >>> beta_grid = Beta.generate_grid(300).data
        >>> alpha_grid = Alpha.generate_grid(300, 26).data
        >>> Sab.from_fgm(alpha_grid, beta_grid).data.iloc[:10, :5].round(6) #doctest: +NORMALIZE_WHITESPACE
        beta	      0.000000	0.012894	 0.025788	0.038682 	0.051576
        alpha
        0.001050	  8.701463	8.417992 7.524148	6.213536	    4.740815
        0.001087  8.553363	8.285768	 7.435678	6.181592	    4.760714
        0.001125	  8.407781	8.155251	 7.346923	6.147319	    4.777252
        0.001164	  8.264674	8.026439	 7.257961	6.110841	    4.790511
        0.001205	  8.124000	7.899326	 7.168869	6.072279	    4.800575
        0.001247	  7.985718	7.773908	 7.079717	6.031753	    4.807533
        0.001291	  7.849787	7.650178	 6.990574	5.989379	    4.811476
        0.001336	  7.716166	7.528129	 6.901504	5.945271	    4.812500
        0.001382	  7.584817	7.407753	 6.812568	5.899540	    4.810701
        0.001431	  7.455701	7.289040	 6.723822	5.852292	    4.806177
        """
        beta_grid_ = beta_grid if isinstance(beta_grid, Beta) else Beta(
            beta_grid)
        alpha_grid_ = alpha_grid if isinstance(alpha_grid, Alpha) else Alpha(
            alpha_grid)
        if beta_grid_.kind == "abs":
            S_values = get_sab_sct(alpha_grid_.data,
                                   - beta_grid_.data,
                                   1.0,
                                   wt)
        else:
            raise ValueError(
                "The beta grid contains negative values and the input is the absolute beta grid")
        return cls(S_values, index=alpha_grid_.data, columns=beta_grid_.data)

    @classmethod
    def from_sct(cls, alpha_grid: Union[Alpha, Iterable[int]],
                 beta_grid: Union[Beta, Iterable[int]], T: float, pdos: Pdos,
                 ws: float = 1):
        """
        Generate S(alpha, -beta) matrix using Short Collision Time.
        .. math::
            S(\alpha,\,-\beta)=\sqrt{\dfrac{1}{4\pi\omega_{s}\alpha T_{\textrm{eff}}/T}}\exp\left(-\dfrac{(\omega_{s}\alpha-\beta)^2}{4\omega_{s}\alpha T_{\textrm{eff}}/T}\right)

        Parameters
        ----------
        alpha_grid : 1D iterable or "Alpha", (N,)
            Alpha grid.
        beta_grid : 1D iterable or "Beta", (M,)
            beta grid.
        T : 'float'
            Temperature in K.
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        ws: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Returns
        -------
        "Sab", (N, M)
            S(alpha, -beta) based on Short Collision Time

        Example
        -------
        SCT:
        Dont fit the normalization and sum rule with the correct precision
        >>> T = 300
        >>> pdos = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> beta_grid = Beta.generate_grid(T)
        >>> alpha_grid = Alpha.generate_grid(T, 26)
        >>> S = Sab.from_sct(alpha_grid, beta_grid, T, pdos)
        >>> S.data.iloc[:10, :5].round(6) #doctest: +NORMALIZE_WHITESPACE
        beta      0.000000  0.012894  0.025788  0.038682  0.051576
        alpha
        0.001050  8.342190  8.092079  7.298835  6.121534  4.773978
        0.001087  8.200211  7.964121  7.209904  6.084151  4.785744
        0.001125  8.060646  7.837859  7.120876  6.044744  4.794361
        0.001164  7.923454  7.713288  7.031821  6.003428  4.799921
        0.001205  7.788595  7.590401  6.942804  5.960320  4.802517
        0.001247  7.656028  7.469191  6.853888  5.915532  4.802243
        0.001291  7.525715  7.349649  6.765132  5.869173  4.799196
        0.001336  7.397618  7.231765  6.676593  5.821349  4.793476
        0.001382  7.271698  7.115530  6.588322  5.772162  4.785181
        0.001431  7.147919  7.000933  6.500370  5.721713  4.774412
        """
        # Save the Phonon Density of States for extrapolation
        cls.pdos = pdos

        # Start the calculation:
        ratio = pdos.Teff(T) / T

        beta_grid_ = beta_grid if isinstance(beta_grid, Beta) else Beta(
            beta_grid)
        alpha_grid_ = alpha_grid if isinstance(alpha_grid, Alpha) else Alpha(
            alpha_grid)
        if beta_grid_.kind == "abs":
            S_values = get_sab_sct(alpha_grid_.data,
                                   - beta_grid_.data,  # S(alpha, -beta)
                                   ratio,
                                   ws)
        else:
            raise ValueError(
                "The beta grid contains negative values and the input is the absolute beta grid")

        return cls(S_values, index=alpha_grid_.data, columns=beta_grid_.data)

    @classmethod
    def from_pdos(cls, alpha_grid: Union[Alpha, Iterable],
                  beta_grid: Union[Beta, Iterable], T: float, pdos: Pdos,
                  threshold: float = 0.0, nphonon: int = 1000,
                  tau_to_file: bool = False, binary: bool = False):
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
        alpha_grid : 1D iterable or "Alpha", (N,)
            Alpha grid.
        beta_grid : 1D iterable or "Beta", (M,)
            beta grid.
        T : 'float'
            Temperature in K.
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        threshold : 'float', optional
            Minimun value to take into account in the creation of tau_n
            functions. For T>200 is convenient to set into 1.0e-14 to speed up
            the calculations. The default is 0.0.
        nphonon : 'int', optional
            Phonon expansion order. The default is 1000.
        tau_to_file : 'bool', optional
            Save the tau_n functions into a file. The default is False.
        binary : 'bool', optional
            Save the tau_n functions into a binary file. The default is False.

        Returns
        -------
        "Sab", (N, M)
            S(alpha, -beta) based on Phonon Density Of States model.

        Example
        -------
        >>> T = 800
        >>> pdos = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> alpha = Alpha(alpha0_).scale(T)
        >>> beta = Beta(beta0_).scale(T)
        >>> S_mat = Sab.from_pdos(alpha, beta, T, pdos, nphonon=700)
        >>> S_mat.data.round(6).iloc[:10, :5]
        beta      0.000000  0.009175  0.018350  0.027524  0.036699
        alpha
        0.001835  0.038004  0.038171  0.038333  0.038492  0.038645
        0.003670  0.074701  0.075013  0.075307  0.075590  0.075857
        0.005505  0.110103  0.110542  0.110941  0.111315  0.111663
        0.007340  0.144226  0.144776  0.145255  0.145693  0.146093
        0.009175  0.177088  0.177733  0.178272  0.178749  0.179174
        0.011010  0.208709  0.209435  0.210015  0.210509  0.210937
        0.012845  0.239108  0.239904  0.240509  0.241002  0.241412
        0.014680  0.268310  0.269164  0.269779  0.270255  0.270631
        0.016515  0.296336  0.297239  0.297853  0.298297  0.298625
        0.018350  0.323212  0.324156  0.324758  0.325158  0.325425
        """
        beta_grid_ = beta_grid if isinstance(beta_grid, Beta) else Beta(
            beta_grid)
        alpha_grid_ = alpha_grid if isinstance(alpha_grid, Alpha) else Alpha(
            alpha_grid)

        # Save Debye wallerr coefficient of the S(alpha, -beta) matrix for
        # interpolation and normalization check
        cls.DebyeWallerCoeff = pdos.DebyeWallerCoeff(T)

        # Save the Phonon Density of States for extrapolation
        cls.pdos = pdos

        # Get the parameters for calculation:
        tau_n, delta_beta, DebyeWallerCoeff = pdos.get_clm_param(T, nphonon=nphonon, threshold=threshold)
        save_tau(tau_n, nphonon, T, tau_to_file, binary)
        S_values = phonon_expansion(alpha_grid_.data,
                                    beta_grid_.data,
                                    nphonon,
                                    tau_n,
                                    delta_beta,
                                    DebyeWallerCoeff)
        return cls(S_values, columns=beta_grid_.data, index=alpha_grid_.data)

    @classmethod
    def from_model(cls, *args, model: str = "pdos", **kwargs):
        """
        Create Sab object from different models. The models available are:
            - "phonon expansion": Phonon expansion model.
            - "fgm": Free Gas Model.
            - "sct": Short Collision Time model.

        Parameters
        ----------
        model : 'str'
            The model to calculate matrix values. The default is "pdos". The
            available models are:
                - "pdos": Phonon expansion model
                - "fgm" : Free Gas Model
                - "sct" : Short Collision Time model

        Parameters for FGM model:
        -------------------------
        alpha_grid : 1D iterable or "Alpha", (N,)
            Alpha grid.
        beta_grid : 1D iterable or "Beta", (M,)
            Absolute beta grid.
        model : 'str', optional
            The model to calculate matrix values. The default is "FGM".
        wt: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Parameters for SCT model:
        -------------------------
        alpha_grid : 1D iterable or "Alpha", (N,)
            Alpha grid.
        beta_grid : 1D iterable or "Beta", (M,)
            beta grid.
        T : 'float'
            Temperature in K.
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        ws: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Parameters for Phonon Expansion model:
        --------------------------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        T : 'float'
            Temperature in K.
        alpha_grid : 1D iterable or "Alpha", (N,)
            Alpha grid.
        beta_grid : 1D iterable or "Beta", (M,)
            beta grid.
        threshold : 'float', optional
            Minimun value to take into account in the creation of tau_n
            functions. For T>200 is convenient to set into 1.0e-14 to speed up
            the calculations. The default is 0.0.
        nphonon : 'int', optional
            Phonon expansion order. The default is 1000.

        Returns
        -------
        Sab
            S(alpha, -beta) matrix based on the chosen model.

        Examples
        --------
        FGM:
        >>> T = 300
        >>> beta_grid = Beta.generate_grid(T).data
        >>> alpha_grid = Alpha.generate_grid(T, 26).data
        >>> Sab.from_model(alpha_grid, beta_grid, model="fgm").data.iloc[:10, :5].round(6) #doctest: +NORMALIZE_WHITESPACE
        beta	      0.000000	0.012894	0.025788	0.038682	0.051576
        alpha
        0.001050	8.701463	8.417992	7.524148	6.213536	4.740815
        0.001087	8.553363	8.285768	7.435678	6.181592	4.760714
        0.001125	8.407781	8.155251	7.346923	6.147319	4.777252
        0.001164	8.264674	8.026439	7.257961	6.110841	4.790511
        0.001205	8.124000	7.899326	7.168869	6.072279	4.800575
        0.001247	7.985718	7.773908	7.079717	6.031753	4.807533
        0.001291	7.849787	7.650178	6.990574	5.989379	4.811476
        0.001336	7.716166	7.528129	6.901504	5.945271	4.812500
        0.001382	7.584817	7.407753	6.812568	5.899540	4.810701
        0.001431	7.455701	7.289040	6.723822	5.852292	4.806177

        SCT:
        >>> pdos = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> Sab.from_model(alpha_grid, beta_grid, T, pdos, model="sct").data.iloc[:10, :5].round(6) #doctest: +NORMALIZE_WHITESPACE
        beta      0.000000  0.012894  0.025788  0.038682  0.051576
        alpha
        0.001050  8.342190  8.092079  7.298835  6.121534  4.773978
        0.001087  8.200211  7.964121  7.209904  6.084151  4.785744
        0.001125  8.060646  7.837859  7.120876  6.044744  4.794361
        0.001164  7.923454  7.713288  7.031821  6.003428  4.799921
        0.001205  7.788595  7.590401  6.942804  5.960320  4.802517
        0.001247  7.656028  7.469191  6.853888  5.915532  4.802243
        0.001291  7.525715  7.349649  6.765132  5.869173  4.799196
        0.001336  7.397618  7.231765  6.676593  5.821349  4.793476
        0.001382  7.271698  7.115530  6.588322  5.772162  4.785181
        0.001431  7.147919  7.000933  6.500370  5.721713  4.774412

        Phonon Expansion:
        >>> T = 800
        >>> beta_grid = Beta(beta0_).scale(T).data
        >>> alpha_grid = Alpha(alpha0_).scale(T).data
        >>> Sab.from_model(alpha_grid, beta_grid, T, pdos, model="pdos", nphonon=700).data.iloc[:10, :5].round(6) #doctest: +NORMALIZE_WHITESPACE
        beta      0.000000  0.009175  0.018350  0.027524  0.036699
        alpha
        0.001835  0.038004  0.038171  0.038333  0.038492  0.038645
        0.003670  0.074701  0.075013  0.075307  0.075590  0.075857
        0.005505  0.110103  0.110542  0.110941  0.111315  0.111663
        0.007340  0.144226  0.144776  0.145255  0.145693  0.146093
        0.009175  0.177088  0.177733  0.178272  0.178749  0.179174
        0.011010  0.208709  0.209435  0.210015  0.210509  0.210937
        0.012845  0.239108  0.239904  0.240509  0.241002  0.241412
        0.014680  0.268310  0.269164  0.269779  0.270255  0.270631
        0.016515  0.296336  0.297239  0.297853  0.298297  0.298625
        0.018350  0.323212  0.324156  0.324758  0.325158  0.325425
        """
        if model.lower() == "fgm":
            return cls.from_fgm(*args, **kwargs)
        elif model.lower() == "sct":
            return cls.from_sct(*args, **kwargs)
        elif model.lower() == "pdos":
            return cls.from_pdos(*args, **kwargs)

    @classmethod
    def from_tau(cls, alpha_grid: Union[Alpha, Iterable],
                 beta_grid: Union[Beta, Iterable], tau_n: np.ndarray,
                 delta_beta: float, DebyeWallerCoeff: float):
        """
        Generate S(alpha, -beta) matrix using tau_n functions.
        .. math::
            S(\alpha,\,-\beta)=\exp(-\alpha\lambda)\sum_{n=0}^{\infty}\dfrac{1}{n!}(\alpha\lambda)^n\mathcal{T}_n(-\beta)

        Numerical appoximation to get convergence in large exponentiation and
        factorial numbers. Each element of the array is related with one alpha
        and represent the following term of the previous equation:
        ..math::
           \sum_{n=0}^{\infty}\dfrac{1}{n!}(\alpha\lambda)^n = \exp(\log(\dfrac{1}{1}(\alpha\lambda)) + \log(\dfrac{1}{2}(\alpha\lambda)) + ...)

        Parameters
        ----------
        alpha_grid : 1D iterable or "Alpha", (N,)
            Alpha grid.
        beta_grid : 1D iterable or "Beta", (M,)
            beta grid.
        tau_n: np.ndarray, (Z, T)
            tau_n functions. The first dimension is the number of the expansion
            and the second dimension is the number of the beta grid.
        delta_beta: float
            Delta beta value.
        DebyeWallerCoeff: float
            Debye Waller coefficient.

        Returns
        -------
        "Sab", (N, M)
            S(alpha, -beta) based on Phonon Density Of States model.

        Example
        -------
        >>> T = 800
        >>> pdos = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> alpha = Alpha(alpha0_).scale(T)
        >>> beta = Beta(beta0_).scale(T)
        >>> DebyeWallerCoeff = pdos.DebyeWallerCoeff(T)
        >>> delta_beta = pdos.to_beta_grid(T).grid
        >>> tau1 = pdos.get_tau_1(T).values
        >>> tau_n = tau_n_functions(tau1, delta_beta, 700, 0.0)
        >>> tau_n = tau_n.get() if gpu_available else tau_n
        >>> S_mat = Sab.from_tau(alpha, beta, tau_n, delta_beta, DebyeWallerCoeff)
        >>> S_mat.data.round(6).iloc[:10, :5]#doctest: +NORMALIZE_WHITESPACE
        beta      0.000000  0.009175  0.018350  0.027524  0.036699
        alpha
        0.001835  0.038004  0.038171  0.038333  0.038492  0.038645
        0.003670  0.074701  0.075013  0.075307  0.075590  0.075857
        0.005505  0.110103  0.110542  0.110941  0.111315  0.111663
        0.007340  0.144226  0.144776  0.145255  0.145693  0.146093
        0.009175  0.177088  0.177733  0.178272  0.178749  0.179174
        0.011010  0.208709  0.209435  0.210015  0.210509  0.210937
        0.012845  0.239108  0.239904  0.240509  0.241002  0.241412
        0.014680  0.268310  0.269164  0.269779  0.270255  0.270631
        0.016515  0.296336  0.297239  0.297853  0.298297  0.298625
        0.018350  0.323212  0.324156  0.324758  0.325158  0.325425
        """
        beta_grid_ = beta_grid if isinstance(beta_grid, Beta) else Beta(
            beta_grid)
        alpha_grid_ = alpha_grid if isinstance(alpha_grid, Alpha) else Alpha(
            alpha_grid)

        # Save Debye wallerr coefficient of the S(alpha, -beta) matrix for
        # interpolation and normalization check
        cls.DebyeWallerCoeff = DebyeWallerCoeff

        S_values = phonon_expansion(alpha_grid_.data,
                                    beta_grid_.data,
                                    tau_n.shape[0],
                                    tau_n,
                                    delta_beta,
                                    DebyeWallerCoeff)
        return cls(S_values, columns=beta_grid_.data, index=alpha_grid_.data)

    @staticmethod
    def sum_rule_check(S: pd.DataFrame) -> None:
        """
        Check if the S(alpha, beta) matrix satifies the sum rule constrain.
        .. math::
            \int_{-\infty}^{\infty}\beta S(\alpha,\,\beta)d\beta = \int_{0}^{\infty}\beta S(\alpha,\,-\beta)(1-\exp(-\beta))d\beta = \alpha

        For SCT and for phonon expansion, the normalization is only satisfy to
        large alpha values, for the rest:
        .. math::
            \int_{0}^{\infty}\beta S(\alpha,\,-\beta)(-1+\exp(-\beta))d\beta = -\alpha

        Parameters
        ----------
        S : 'pd.DataFrame', (N, M)
            S(alpha, beta) matrix.

        Returns
        -------
        "None"
            If the sum rule is not satisfied with good accuracy a warning is
            raise. If the accuracy is very low, a ValueError will be raise.

        Raises
        ------
        ValueError
            The sum rule constrain is not satified.

        """
        sum_rule = S.apply(_sum_rule, axis="columns")
        sum_rule /= S.index.values
        if (abs(1 - abs(sum_rule)) > 0.6).any():
            raise ValueError("Sum rule of S(alpha, -beta) not satisfied")
        if (abs(1 - abs(sum_rule)) > 1.0e-3).any():
            warnings.warn(
                "Sum rule of S(alpha, -beta) not satisfied with an precision of 1.0e-3")
        return

    def normalization_check(self, S: pd.DataFrame) -> None:
        """
        Check if the S(alpha, beta) matrix satifies the normalization constrain.
        .. math::
            \int_{-\infty}^{\infty}S(\alpha,\,\beta)d\beta = \int_{0}^{\infty}S(\alpha,\,-\beta)(1+\exp(-\beta))d\beta = 1

        For SCT and for phonon expansion, the normalization is only satisfy to
        large alpha values, for the rest:
        .. math::
            \int_{0}^{\infty}S(\alpha,\,-\beta)(1+\exp(-\beta))d\beta = 1 - \exp(-\alpha\lambda)

        Parameters
        ----------
        S : 'pd.DataFrame', (N, M)
            S(alpha, beta) matrix.

        Returns
        -------
        "None"
            If the normalization is not satisfied with good accuracy a warning
            is raise. If the accuracy is very low, a ValueError will be raise.

        Raises
        ------
        ValueError
            The normalization constrain is not satified.

        """
        normalization = S.apply(_normalization, axis="columns")
        if hasattr(self, "DebyeWallerCoeff"):
            normalization /= (
                        1 - np.exp(- S.index.values * self.DebyeWallerCoeff))
        if (abs(normalization - 1.0) > 1.0e-2).any():
            warnings.warn(
                "Normalization of S(alpha, -beta) not satisfied with an precision of 1.0e-2")
        return

    def get_beta(self, beta_new: Union[Iterable, float],
                 add: bool = False) -> pd.DataFrame:
        """
        Quadratic interpolation to get the probability of the new beta value
        for all the alpha existing in the S(alpha, -beta) matrix:
        .. math::
            \left\{ \mid\beta_{new}\mid = P\left(\mid\beta_new\mid, \alpha_k\right)\right\} \text{ for }k=0, 1, ...

        The method do not make extrapolation.

        Parameters
        ----------
        beta_new : "float" or 1D iterable
            New beta values.
        add : "bool", optional
            Optional argument to add the output to the existing S(alpha, -beta)
            matrix or only get the pd.Dataframe. The default is False.

        Returns
        -------
        "pd.Dataframe" or "Sab"
            pd.Dataframe with the requested probabilities for the new betas for
            all the existing alpha. If add is True, the pd.Dataframe is merge
            in the S(alpha, -beta) matrix.

        Example
        -------
        >>> T = 300
        >>> pdos = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> alpha = Alpha(alpha0_).scale(T)
        >>> beta = Beta(beta0_).scale(T)
        >>> S_mat = Sab.from_pdos(alpha, beta, T, pdos, threshold=1.0e-14)
        >>> beta_new = 0.01
        >>> S_mat.get_beta(beta_new).iloc[0:10]
        beta          0.01
        alpha
        0.004893  0.005423
        0.009786  0.010765
        0.014680  0.016027
        0.019573  0.021208
        0.024466  0.026309
        0.029359  0.031331
        0.034253  0.036275
        0.039146  0.041141
        0.044039  0.045929
        0.048932  0.050641

        >>> beta_new = [0.01, 0.03]
        >>> S_mat.get_beta(beta_new, add=True).data.iloc[0:10, 0:4] #doctest: +NORMALIZE_WHITESPACE
        beta      0.000000  0.010000  0.024466  0.030000
        alpha
        0.004893  0.005396  0.005423  0.005462  0.005477
        0.009786  0.010712  0.010765  0.010842  0.010872
        0.014680  0.015948  0.016027  0.016140  0.016184
        0.019573  0.021104  0.021208  0.021357  0.021414
        0.024466  0.026181  0.026309  0.026493  0.026563
        0.029359  0.031179  0.031331  0.031549  0.031632
        0.034253  0.036100  0.036275  0.036526  0.036621
        0.039146  0.040943  0.041141  0.041423  0.041530
        0.044039  0.045710  0.045929  0.046243  0.046361
        0.048932  0.050400  0.050641  0.050984  0.051114
        """
        beta_new_ = beta_new if hasattr(beta_new, '__len__') else [beta_new]
        beta_values = self.data.apply(lambda x: reshape_differential(x,
                                                                     beta_new_,
                                                                     bounds_error=True,
                                                                     kind="quadratic"),
                                      axis=1)
        beta_df = pd.DataFrame.from_records(beta_values.values,
                                            index=beta_values.index,
                                            columns=pd.Index(beta_new_,
                                                             name="beta"))
        if add:
            return Sab(pd.concat([beta_df, self.data], axis=1))
        else:
            return beta_df

    def get_alpha(self, alpha_new: Union[Iterable, float],
                  add: bool = False) -> pd.DataFrame:
        """
        Unit base interpolation to get the probability of the new alpha values
        for all the beta existing in the S(alpha, -beta) matrix.
        .. math::
            \left\{ \mid\alpha_{new}\mid = P\left(\mid\beta_{k}\mid, \alpha_{new}\right) \text{ for }k=0, 1, ...

        The method do not make extrapolation.

        Parameters
        ----------
        alpha_new : "float" or 1D iterable
            New alpha values.
        add : "bool", optional
            Optional argument to add the output to the existing S(alpha, -beta)
            matrix or only get the pd.Dataframe. The default is False.

        Returns
        -------
        "pd.Dataframe" or "Sab"
            pd.Dataframe with the requested probabilities for the new alpha for
            all the existing alpha. If add is True, the pd.Dataframe is merge
            in the S(alpha, -beta) matrix.

        Example
        -------
        >>> T = 300
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> beta_grid = Beta(beta0_U238).scale(T)
        >>> alpha_grid = Alpha(alpha0_U238).scale(T)
        >>> S_mat = Sab.from_pdos(alpha_grid, beta_grid, T, pdos, threshold=1.0e-14)
        >>> alpha_new = 0.00013
        >>> S_mat.get_alpha(alpha_new).data.iloc[::, 0:5] #doctest: +NORMALIZE_WHITESPACE
        beta     0.000000  0.025237  0.050474  0.075712  0.100949
        alpha
        0.00013  0.000498  0.000504  0.000467  0.000461  0.000462

        >>> alpha_new = [1.25e-4, 1.35e-4]
        >>> S_mat.get_alpha(alpha_new).data.iloc[::, 0:5] #doctest: +NORMALIZE_WHITESPACE
        beta      0.000000  0.025237  0.050474  0.075712  0.100949
        alpha
        0.000125  0.000479  0.000484  0.000449  0.000444  0.000445
        0.000135  0.000517  0.000523  0.000485  0.000479  0.000480

        >>> alpha_new = [1.25e-4, 1.35e-4]
        >>> S_mat.get_alpha(alpha_new, add=True).data.iloc[0:5, 0:5] #doctest: +NORMALIZE_WHITESPACE
        beta	    0.000000	0.025237	0.050474	0.075712	0.100949
        alpha
        0.000112	0.000430	0.000435	0.000403	0.000399	0.000399
        0.000120	0.000460	0.000466	0.000432	0.000427	0.000428
        0.000125	0.000479	0.000484	0.000449	0.000444	0.000445
        0.000129	0.000493	0.000499	0.000463	0.000457	0.000458
        0.000135	0.000517	0.000523	0.000485	0.000479	0.000480
        """
        alpha_new_ = alpha_new if hasattr(alpha_new, '__len__') else [alpha_new]
        alpha_vector = []
        for new_alpha in alpha_new_:
            alpha_vector.append(self._get_single_alpha(new_alpha))
        alpha_new_df = pd.concat(alpha_vector, axis=1).T
        if add:
            return Sab(pd.concat([self.data, alpha_new_df]))
        else:
            return Sab(alpha_new_df)

    def _get_single_alpha(self, alpha_new: float) -> pd.DataFrame:
        """
        Interpolate S(alpha, -beta) using unit base interpolation to get the
        probabilities for the new alpha values:
        .. math::
            \left\{ \mid\alpha_{new}\mid = P\left(\mid\beta_{k}\mid, \alpha_{new}\right) \text{ for }k=0, 1, ...

        The method do not make extrapolation.

        Parameters
        ----------
        alpha_new : "float"
            Alpha value to get interpolated.

        Returns
        -------
        "pd.Series"
            Interpolated beta grid vector for the introduced alpha.

        Example
        -------
        >>> T = 300
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> beta_grid = Beta(beta0_U238).scale(T)
        >>> alpha_grid = Alpha(alpha0_U238).scale(T)
        >>> S_mat = Sab.from_pdos(alpha_grid, beta_grid, T, pdos, threshold=1.0e-14)
        >>> alpha_new = 0.00013
        >>> alpha_vector = S_mat._get_single_alpha(alpha_new)
        >>> alpha_vector.iloc[0:10]  #doctest: +NORMALIZE_WHITESPACE
        alpha     0.00013
        beta
        0.000000  0.000498
        0.025237  0.000504
        0.050474  0.000467
        0.075712  0.000461
        0.100949  0.000462
        0.126186  0.000472
        0.151423  0.000490
        0.176660  0.000509
        0.201898  0.000528
        0.227135  0.000553

        Check the contrains:
        >>> debye_weller = pdos.DebyeWallerCoeff(T)
        >>> round(integrate(alpha_vector.iloc[::, 0] * (1 + np.exp(-beta_grid.data))) / (1 - np.exp(-debye_weller * alpha_new)), 6)
        1.005976

        >>> round(integrate(alpha_vector.iloc[::, 0] * beta_grid.data * (1 -  np.exp( - beta_grid.data))), 6)
        0.000131
        """
        alpha_grid = self.alpha.data
        if alpha_new > alpha_grid.max():
            raise SyntaxError("alpha out of range(alpha_max = "
                              + str(alpha_grid.max()) + ")")
        elif alpha_new < alpha_grid.min():
            raise SyntaxError("alpha out of range (alpha_min = "
                              + str(alpha_grid.min()) + ")")
        elif alpha_new in alpha_grid:
            return self.data.loc[alpha_new]

        beta = self.beta.data
        upper_bound = alpha_grid.searchsorted(alpha_new, side="right")
        alpha_0 = alpha_grid[upper_bound - 1]
        alpha_2 = alpha_grid[upper_bound]
        prob = self.data.loc[[alpha_0, alpha_2]].T

        if hasattr(self, "DebyeWallerCoeff"):
            debye_weller = self.DebyeWallerCoeff
            prob_norm = prob.apply(lambda x: (1 + np.exp(-x.index)) * x / (
                        1 - np.exp(-debye_weller * x.name)))
        else:
            prob_norm = prob.apply(lambda x: (1 + np.exp(-x.index)) * x)

        q = proportionality_factor(alpha_new, alpha_0, alpha_2, mode="linlog")
        alpha_new_escale = (1 - q) * prob_norm.loc[::, alpha_0] + q * prob_norm.loc[::, alpha_2]

        alpha_new_vector = alpha_new_escale / (1 + np.exp(-beta))
        if hasattr(self, "DebyeWallerCoeff"):
            alpha_new_vector *= (1 - np.exp(- debye_weller * alpha_new))
        return pd.DataFrame(alpha_new_vector,
                            columns=pd.Index([alpha_new], name="alpha"))

    def get_Alpha_Beta(self, alpha: Union[Iterable, float],
                       beta: Union[Iterable, float]) -> pd.DataFrame:
        """
        Get intepolated values for the beta and alpha values from the
        S(alpha, beta) matrix. This method take into account the sing of the
        beta that is introduced.
        .. math::
            Beta < 0:
                \left\{ \mid\alpha\mid = P\left(- \mid\beta\mid, \alpha\right)
            Beta > 0:
                \left\{ \mid\alpha\mid = P\left(-\mid\beta\mid, \alpha\right) * exp(-\mid\beta_{new}\mid)

        Parameters
        ----------
        alpha : "float" or 1D iterable
            Alpha values to interpolate.
        beta : "float" or 1D iterable
            Beta values to interpolate.

        Returns
        -------
        "pd.Dataframe"
            Interpolated S(alpha, beta)

        Example
        -------
        >>> T = 300
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> beta_grid = Beta(beta0_U238).scale(T)
        >>> alpha_grid = Alpha(alpha0_U238).scale(T)
        >>> S_mat = Sab.from_pdos(alpha_grid, beta_grid, T, pdos, threshold=1.0e-14)
        >>> alpha_new = [1.25e-4, 1.35e-4]
        >>> beta_new = [0.01, 0.03, -0.01, -0.03]
        >>> S_mat.get_Alpha_Beta(alpha_new, beta_new) #doctest: +NORMALIZE_WHITESPACE
        beta         -0.03     -0.01      0.01      0.03
        alpha
        0.000125  0.000479  0.000487  0.000483  0.000465
        0.000135  0.000518  0.000526  0.000521  0.000503
        """
        alpha_ = alpha if hasattr(alpha, '__len__') else [alpha]
        beta_ = np.array(beta) if hasattr(beta, '__len__') else np.array([beta])
        interp_Alpha_Beta = self.get_alpha(alpha_) \
            .get_beta(abs(beta_)) \
            .set_axis(pd.Index(beta_, name="beta"), axis=1)
        if (beta_ > 0).any():
            interp_Alpha_Beta.loc[::, beta_ > 0] *= np.exp(- beta_[beta_ > 0])
        return interp_Alpha_Beta.sort_index(axis=0).sort_index(axis=1)


def _sum_rule(x: pd.Series, n: int = 1) -> float:
    """
    Calculate the "n" sum rule value for a fix alpha value.
    .. math::
        \int_{-\infty}^{\infty}\beta^n S(\alpha,\,\beta)d\beta = \int_{0}^{\infty}\beta^n S(\alpha,\,-\beta)(1-\exp(-\beta))d\beta
    Parameters
    ----------
    x : 'pd.Series', (N)
        S(alpha, beta) matrix values for fix alpha.
    n: 'int', optional
        The number of the sum_rule

    Returns
    -------
    "float"
        Sum rule value for a fix alpha.

    Example
    -------
    >>> beta_grid = Beta.generate_grid(300)
    >>> alpha_grid = Alpha.generate_grid(300, 26)
    >>> s = Sab.from_fgm(alpha_grid, beta_grid).data
    >>> _sum_rule(s.iloc[1, ::]).round(6)
    0.001087
    """
    beta = x.index.values
    S_values = x.values
    return trapezoid(np.power(beta, n) * S_values * (1 - np.exp(-beta)), beta)


def _normalization(x: pd.Series) -> float:
    """
    Normalization rule value for a fix alpha value of the S(alpha, beta) matrix.

    Parameters
    ----------
    x : 'pd.Series', (N,)
        S(alpha, beta) matrix values for fix alpha.

    Returns
    -------
    normalization_values : "float"
        Normalization value for a fix alpha.

    Example
    -------
    >>> beta_grid = Beta.generate_grid(300)
    >>> alpha_grid = Alpha.generate_grid(300, 26)
    >>> s = Sab.from_fgm(alpha_grid, beta_grid).data
    >>> _normalization(s.iloc[0, ::]).round(6)
    1.0
    """
    beta = x.index.values
    S_asymm_values = x.values
    S = pd.Series((1 + np.exp(-beta)) * S_asymm_values, index=beta)
    return integrate(S)


def proportionality_factor(alpha: float, alpha_i: float,
                           alpha_i_plus_one: float,
                           mode: str = "linlog") -> float:
    """
    Get the proportionality factor for unit-base interpolation.

    Parameters
    ----------
    alpha : "float"
        Alpha value to be interpolated.
    alpha_i : "float"
        lower alpha bound.
    alpha_i_plus_one : "float"
        upper alpha bound.
    mode : "str", optional
        Is define by how the probability table is interpolated. The default is
        "linlog".

    Returns
    -------
    "float"
        Proportionality factor for unit-base interpolation
    """
    if mode == "linlog":
        q = np.log(alpha / alpha_i) / np.log(alpha_i_plus_one / alpha_i)
    elif mode == "linlin":
        q = (alpha - alpha_i) / (alpha_i_plus_one - alpha_i)
    elif mode == "const":
        q = 1
    return q


@optional_jit
def _phonon_expansion(alpha: xp.ndarray, beta: xp.ndarray, nphonon: int,
                      tau_n: xp.ndarray, delta_beta: float,
                      DebyeWallerCoeff: float) -> xp.ndarray:
    """
    Generate S(alpha, -beta) matrix using tau_n functions:
    .. math::
        S(\alpha,\,-\beta)=\exp(-\alpha\lambda)\sum_{n=0}^{\infty}\dfrac{1}{n!}(\alpha\lambda)^n\mathcal{T}_n(-\beta)

    Parameters
    ----------
    alpha: 'xp.ndarray', (N,)
        alpha grid values in the cpu as numpy array or in the gpu as cupy array.
    beta: 'xp.ndarray', (M,)
        beta grid values in the cpu as numpy array or in the gpu as cupy array.
    nphonon: 'int'
        Number of phonon expansion.
    tau_n: 'xp.ndarray', (Z, T)
        tau_n functions. The first dimension is the number of the expansion
        and the second dimension is the number of the beta grid. In the cpu as
        numpy array or in the gpu as cupy array.
    delta_beta: 'float'
        Delta beta value.
    DebyeWallerCoeff: 'float'
        Debye Waller coefficient.

    Returns
    -------
    'xp.ndarray', (N, M)
        S(alpha, -beta) matrix values.
    """
    tau_n_beta = xp.arange(tau_n.shape[1]) * delta_beta
    # Zero phonon expansion:
    iter_sum = xp.log(alpha * DebyeWallerCoeff)
    alpha_mul = xp.exp(- alpha * DebyeWallerCoeff + iter_sum)
    sab_values = xp.outer(alpha_mul, xp.interp(beta, tau_n_beta, tau_n[0]))

    # Higher phonon expansion (nphonon >= 1):
    for n in range(1, nphonon):
        # Compute S(alpha, -beta) for tau_n reshape
        iter_sum += xp.log(alpha * DebyeWallerCoeff / (n + 1))
        alpha_mul = xp.exp(- alpha * DebyeWallerCoeff + iter_sum)
        sab_values += xp.outer(alpha_mul, xp.interp(beta, tau_n_beta, tau_n[n]))
    return sab_values


def phonon_expansion(*args) -> np.ndarray:
    """
    Generate S(alpha, -beta) matrix using tau_n functions:
    .. math::
        S(\alpha,\,-\beta)=\exp(-\alpha\lambda)\sum_{n=0}^{\infty}\dfrac{1}{n!}(\alpha\lambda)^n\mathcal{T}_n(-\beta)

    Parameters
    ----------
    alpha: 'xp.ndarray', (N,)
        alpha grid values
    beta: 'xp.ndarray', (M,)
        beta grid values
    nphonon: 'int'
        Number of phonon expansion.
    tau_n: 'xp.ndarray', (Z, T)
        tau_n functions. The first dimension is the number of the expansion
        and the second dimension is the number of the beta grid.
    delta_beta: 'float'
        Delta beta value.
    DebyeWallerCoeff: 'float'
        Debye Waller coefficient.

    Returns
    -------
    'xp.ndarray', (N, M)
        S(alpha, -beta) matrix values.
    """
    if gpu_available:
        arg_gpu = [xp.asarray(arg) if isinstance(arg, np.ndarray) else arg
                   for arg in args]
        return _phonon_expansion(*arg_gpu).get()
    else:
        return _phonon_expansion(*args)


@nb.jit(nopython=True, nogil=True, cache=True, parallel=True)
def get_sab_sct(alpha: np.ndarray, beta: np.ndarray, Tratio: float,
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
