from scipy.constants import physical_constants as const
from solid_cinel.core._numba import get_alpha
from solid_cinel.core.beta import Beta
from typing import Iterable, Union
import numpy as np
import pandas as pd

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


class Alpha:
    """
    Class with all the method for the creation and manipulation of alpha
    grids

    Attributes
    ----------
    data : 'np.ndarray'
        Array of alpha values.
    to_index : 'pd.Index'
        pandas Index of alpha values.

    Methods
    -------
    generate_grid -> 'Alpha
        Generate a alpha grid for a given temperature and atomic mass
    from_parameters -> 'Alpha'
        Generate the alpha values for the given combination of the input
        parameters
    scale -> 'Alpha'
        Scale alpha or beta spectrum
    get_theta -> pd.Series
        Based on the S(alpha, -beta) matrix, get the posible scattering angles
        for a scattering atom, temperature and incident neutron energy.
     """

    def __init__(self, array: Iterable):
        """
        Initialize the Alpha class

        Parameters
        ----------
        array : Iterable
            Array of alpha values
        """
        self.data = np.unique(array)

    @property
    def to_index(self) -> pd.Index:
        """Tranform the Beta class data into a pandas Index."""
        return pd.Index(self.data, name="alpha")

    @classmethod
    def generate_grid(cls, T: float, M: float, num_grid: int = 300,
                      min_E: float = 2.8e-3, thermal_threshold: float = 5.,
                      scale: bool = False, **kwargs):
        """
        Generate a alpha grid for a given temperature and atomic mass.

        Parameters
        ----------
        T : 'float'
            Temperature in K.
        M : 'float'
            atomic mass of scatterer in amu.
        num_grid : 'int', optional
            Number of grid. The default is 400.
        mid_E : 'float', optional
            minimum of energy transfer in eV. The default is 0.08.
        thermal_threshold : 'float', optional
            thermal energy threshold in eV. The default is 5.
        scale : 'bool', optional
            Option to scale beta and alpha grid with the method scale_grid. The
            default is False.

        Parameters for scale_grid
        -------------------------
        therm : 'float', optional
            factor for regrid alpha and beta. The default is 0.0253.

        Returns
        "Alpha"
            Generate grid from minimun alpha to maximun alpha for a certain
            range of energies.

        Example
        -------
        >>> Alpha.generate_grid(300, 26, num_grid=10).data.round(6)
        array([1.0500000e-03, 3.2850000e-03, 1.0270000e-02, 3.2114000e-02,
               1.0041300e-01, 3.1397500e-01, 9.8174500e-01, 3.0697450e+00,
               9.5985550e+00, 3.0013001e+01])
        """
        AkT = M * kb * T / m
        min_alpha = min_E / 4 / AkT
        max_alpha = 4 * thermal_threshold / AkT
        alpha_grid = np.logspace(np.log10(min_alpha), np.log10(max_alpha),
                                 num=num_grid)
        if scale:
            return cls(alpha_grid).scale(T, **kwargs)
        else:
            return cls(alpha_grid)

    @classmethod
    def from_parameters(cls, Eout: Union[Iterable, float],
                        Ein: Union[Iterable, float],
                        T: Union[Iterable, float], M: float,
                        theta: Union[Iterable, float]):
        """
        Generate the alpha values for the given combination of the input
        parameters:
        .. math::
            \alpha = \frac{E^\prime + E - 2 \mu\sqrt{E^\prime E}}{Ak_BT}

        Parameters
        ----------
        Eout : 1D iterable or 'float'
            Neutron output energies in eV.
        Ein : 1D iterable or 'float'
            Neutron incident energy in eV.
        T : 1D iterable or 'float'
            Temperature in Kelvin.
        M : 'float'
            Atom mass, amu
        theta : 1D iterable or 'float'
            scattering angle in Degrees.

        Returns
        -------
        "Alpha"
            Alpha grid generated for the given combination of the input
            parameters.

        Example
        -------
        >>> T = 800
        >>> Ein = 0.33118
        >>> Eout = [0.331180, 0.331812, 0.332445, 0.333077, 0.333710]
        >>> M = 26.98153433356103
        >>> theta = 0.101125 * 180 / np.pi
        >>> Alpha.from_parameters(Eout, Ein, T, M, theta).data.round(6)
        array([0.001835, 0.001837, 0.001839, 0.001842, 0.001845])
        """
        Eout_ = np.array(Eout) if hasattr(Eout, '__len__') else np.array([Eout])
        Ein_ = np.array(Ein) if hasattr(Ein, '__len__') else np.array([Ein])
        T_ = np.array(T) if hasattr(T, '__len__') else np.array([T])
        mu = np.cos(theta * np.pi / 180) if hasattr(theta,
                                                    '__len__') else np.cos(
            np.array([theta]) * np.pi / 180)
        return cls(get_alpha(Eout_, Ein_, T_, M, mu))

    def get_theta(self, T: float, Ein: float, M: float,
                  beta_grid: Union[Beta, Iterable]) -> pd.Series:
        """
        Based on the S(alpha, -beta) matrix, get the posible scattering angles
        for a scattering atom, temperature and incident neutron energy.
        .. math::
            \mu = \frac{E^\prime + E - \alpha Ak_BT}{2\sqrt{E^\prime E}}
            \theta = \arccos(\mu)

        Parameters
        ----------
        beta_grid: 'Beta' or 1D iterable
            Beta grid.
        T : 'float'
            Temperature in K.
        Ein : 'float'
            Incident neutron energy in eV.
        m : 'float'
            Atom mass, amu.

        Returns
        -------
        "pd.Series"
            Series with the theta values for a range of alpha and a fix Ein, M
            and Beta grid.

        Example
        -------
        >>> T = 800
        >>> M = 26.98153433356103
        >>> Ein = 0.33118
        >>> beta_grid = Beta(beta0_).scale(T)
        >>> alpha_grid = Alpha(alpha0_).scale(T)
        >>> alpha_grid.get_theta(T, Ein, M, beta_grid).iloc[0:5].round(6)
        alpha
        0.001835    0.101125
        0.003670    0.143002
        0.005505    0.175125
        0.007340    0.202199
        0.009175    0.226045
        Name: mu, dtype: float64

        >>> T = 800
        >>> Ein = 0.33118
        >>> Eout = [0.331180, 0.331812, 0.332445, 0.333077, 0.333710]
        >>> beta_grid = Beta.from_parameters(Eout, Ein, T)
        >>> M = 26.98153433356103
        >>> theta = 45
        >>> alpha = Alpha.from_parameters(Eout, Ein, T, M, theta)
        >>> theta = alpha.get_theta(T, Ein, M, beta_grid)
        >>> import numpy as np
        >>> theta * 180 / np.pi
        alpha
        0.105201    45.0
        0.105302    45.0
        0.105403    45.0
        0.105504    45.0
        0.105605    45.0
        Name: mu, dtype: float64
        """
        alpha = self.data
        A = M / m

        beta = beta_grid if isinstance(beta_grid, Beta) else Beta(beta_grid)
        E_prima = beta.get_Eout(T, Ein).values

        if len(E_prima) > len(alpha):
            E_prima = E_prima[:len(alpha)]
        elif len(E_prima) < len(alpha):
            alpha = alpha[:len(E_prima)]

        mu = E_prima + Ein - alpha * A * kb * T
        mu /= 2 * np.sqrt(E_prima * Ein)
        mu = np.arccos(mu[abs(mu) <= 1])
        return pd.Series(mu, index=Alpha(alpha[:len(mu)]).to_index, name="mu")

    def scale(self, T: float, therm: float = 0.0253):
        """
        Scale alpha or beta spectrum.
        .. math::
            \alpha_{esc}= \alpha * \dfrac{therm}{k_BT}

        Parameters
        ----------
        grid : 'np.ndarray' of 1D or 2D
            Alpha o Beta grid.
        T : 'float'
            Temperature in K.
        therm : 'float', optional
            factor for regrid alpha and beta. The default is 0.0253.

        Returns
        -------
        "Alpha"
            Scaled alpha grid

        Example
        -------
        >>> T = 300
        >>> alpha0 = Alpha.generate_grid(T, 26, num_grid=10)
        >>> alpha0.scale(T).data.round(6)
        array([1.0280000e-03, 3.2140000e-03, 1.0051000e-02, 3.1428000e-02,
               9.8269000e-02, 3.0727100e-01, 9.6078300e-01, 3.0041990e+00,
               9.3936040e+00, 2.9372154e+01])
        """
        return Alpha(Beta(self.data).scale(T, therm=therm).data)
