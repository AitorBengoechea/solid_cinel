# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:46:42 2022
@author: Aitor Bengoechea
"""
from scipy.constants import physical_constants as const
from scipy.integrate import trapezoid
from solid_cinel.core.generic import normalization_coeff, reshape_differential
from solid_cinel.core._numba import tau_n_CPU
import numpy as np
import pandas as pd
import scipy as sp
import warnings

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


class S():
    """
    Class containing all the methods and properties of a asymmetric
    S(alpha, beta) matrix.
    """

    def __init__(self, *args, **kwargs):
        self.data = pd.DataFrame(*args, **kwargs)

    @property
    def data(self) -> pd.DataFrame:
        """Dataframe with the S(alpha, beta) matrix values."""
        return self._data

    @data.setter
    def data(self, df):
        """
        Construct the S(alpha, beta) matrix and check if the data achieve the
        normalization and sum rule constrain.

        Parameters
        ----------
        df : 2D iterable
            Iterable containing the S(alpha, beta) matrix.
        """
        df_ = pd.DataFrame(df).sort_index(axis=0).sort_index(axis=1)
        df_.index.name = "alpha"
        df_.columns.name = "beta"
        # Normalization constrains:
        self.normalization_check(df_)
        self.sum_rule_check(df_)
        # DataFrame:
        self._data = df_

    def to_sym(self, detail_balance=True) -> pd.DataFrame:
        """
        Generate the symmetric S(alpha, beta) matrix from the asymmetric
        S(alpha, -beta) matrix.

        Parameters
        ----------
        detail_balance : 'bool', optional
            Relationships between upscatter and downscatter. The default is
            True.

        Example
        -------
        >>> beta_grid = gen_beta(300)
        >>> alpha_grid = gen_alpha(300, 26)
        >>> S.from_fgm(alpha_grid, beta_grid).to_sym().iloc[:10, :5].round(6)
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

    @classmethod
    def from_fgm(cls, alpha_grid, beta_grid, T=None, w_t=1):
        """
        Generate S(alpha, -beta) matrix using Free Gas Model.
        .. math::
            S_t(\alpha,\,-\beta)=\dfrac{1}{\sqrt{4\pi w_t\alpha}}\exp\left(-\dfrac{(w_t\alpha+\beta)^2}{4w_t\alpha}\right)\end{equation}

        Parameters
        ----------
        alpha_grid : 1D iterable
            Alpha grid.
        beta_grid : 1D iterable
            beta grid.
        model : 'str', optional
            The model to calculate matrix values. The default is "FGM".
        T : 'float', optional
            Option to scale beta and alpha grid with the method scale_grid. The
            default is None.
        w_t: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Example
        -------
        FGM:
        >>> beta_grid = gen_beta(300)
        >>> alpha_grid = gen_alpha(300, 26)
        >>> S.from_fgm(alpha_grid, beta_grid).data.iloc[:10, :5].round(6)
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
        """
        if T is not None:
            alpha_grid = scale_grid(alpha_grid, T)
            beta_grid = scale_grid(beta_grid, T)

        f = lambda i, j: np.exp(-(alpha_grid[i] * w_t - beta_grid[j]) ** 2 / (4 * w_t * alpha_grid[i])) / np.sqrt(4 * np.pi * w_t * alpha_grid[i])

        S_values = np.fromfunction(np.vectorize(f),
                                   (len(alpha_grid), len(beta_grid)),
                                   dtype=int
                                   )
        return cls(S_values, index=alpha_grid, columns=beta_grid)

    @classmethod
    def from_sct(cls, alpha_grid, beta_grid, T, pdos, scale=False, w_s=1):
        """
        Generate S(alpha, -beta) matrix using Short Collision Time.
        .. math::
            S(\alpha,\,-\beta)=\sqrt{\dfrac{1}{4\pi\omega_{s}\alpha T_{\textrm{eff}}/T}}\exp\left(-\dfrac{(\omega_{s}\alpha-\beta)^2}{4\omega_{s}\alpha T_{\textrm{eff}}/T}\right)

        Parameters
        ----------
        alpha_grid : 1D iterable
            Alpha grid.
        beta_grid : 1D iterable
            beta grid.
        T : 'float'
            Temperature in K.
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        w_s: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Example
        -------
        SCT:
        Dont fit the normalization and sum rule with the correct precision
        >>> T = 300
        >>> from solid_cinel.core.material.pdos import Pdos
        >>> pdos = Pdos.from_data(rho_in_energy, interv_in_energy)
        >>> beta_grid = gen_beta(T)
        >>> alpha_grid = gen_alpha(T, 26)
        >>> S = S.from_sct(alpha_grid, beta_grid, T, pdos)
        >>> S.data.iloc[:10, :5].round(6)
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
        if scale:
            alpha_grid = scale_grid(alpha_grid, T)
            beta_grid = scale_grid(beta_grid, T)
        ratio = pdos.Teff(T) / T

        f = lambda i, j: np.exp(-(alpha_grid[i] * w_s - beta_grid[j]) ** 2 / (4 * ratio * w_s * alpha_grid[i])) / np.sqrt(4 * np.pi * w_s * alpha_grid[i] * ratio)

        S_values = np.fromfunction(np.vectorize(f),
                                   (len(alpha_grid), len(beta_grid)),
                                   dtype=int
                                   )
        return cls(S_values, index=alpha_grid, columns=beta_grid)

    @classmethod
    def from_pdos(cls, alpha_grid, beta_grid, T, pdos, scale=False,
                  threshold=0.0, nphonon=1000):
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
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        T : 'float'
            Temperature in K.
        alpha_grid : 1D iterable
            Alpha grid.
        beta_grid : 1D iterable
            beta grid.
        scale : 'bool', optional
            Option to scale beta and alpha grid with the method scale_grid. The
            default is False.
        threshold : 'float', optional
            Minimun value to take into account in the creation of tau_n
            functions. For T>200 is convenient to set into 1.0e-14 to speed up
            the calculations. The default is 0.0.
        nphonon : 'int', optional
            Phonon expansion order. The default is 1000.

        Example
        -------
        >>> T = 800
        >>> from solid_cinel.core.material.pdos import Pdos
        >>> pdos = Pdos.from_data(rho_in_energy, interv_in_energy)
        >>> S_mat = S.from_pdos(alpha0_, beta0_, T, pdos, scale=True, threshold=1.0e-14)
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
        if scale:
            alpha_grid = scale_grid(alpha_grid, T)
            beta_grid = scale_grid(beta_grid, T)
        debye_waller_coeff = pdos.DebyeWallerCoeff(T)
        tau1 = pdos._get_tau_1(T)
        delta_beta = tau1.index[1]
        S_values, iter_sum = cls._S_from_tau1(tau1, debye_waller_coeff,
                                             alpha_grid, beta_grid)
        if nphonon > 1:
            tau1 = tau_n_minus_1 = tau1.values
            for n in range(1, nphonon):
                # Tau_n(-beta)
                tau_n = tau_n_CPU(delta_beta, tau1, tau_n_minus_1, threshold)
                beta_tau_n = np.arange(len(tau_n)) * delta_beta
                check_tau_n(tau_n, beta_tau_n)

                # Interpolate tau_n(-beta):
                tau_n_reshape = reshape_differential(beta_tau_n, tau_n,
                                                     beta_grid)

                # Compute S(alpha, -beta) for tau_n reshape
                iter_sum += np.log(alpha_grid * debye_waller_coeff / (n + 1))
                alpha_mul = np.exp(-alpha_grid * debye_waller_coeff + iter_sum)
                S_values += np.outer(alpha_mul, tau_n_reshape)

                # Next tau_n
                tau_n_minus_1 = tau_n
        return cls(S_values, columns=beta_grid, index=alpha_grid)

    @staticmethod
    def _S_from_tau1(tau1, debye_waller_coeff, alpha_grid, beta_grid):
        """
        Generate S(alpha, -beta) matrix using first phonon expansion.
        .. math::
            S(\alpha,\,-\beta)=\exp(-\alpha\lambda)(\alpha\lambda)\mathcal{T}_1(-\beta)

        Parameters
        ----------
        tau1 : 'pd.Series'
            $\mathcal{T}_1(-\beta)$ function values and in the index beta
            values.
        debye_waller_coeff : 'float'
            Debye Waller Coefficient.
        alpha_grid : 1D iterable
            Alpha grid.
        beta_grid : 1D iterable
            beta grid.

        Returns
        -------
        S_values : 'np.array[:, :]'
            S(alpha, -beta) matrix values for the firts phonon expansion.
        iter_sum : 'np.array[:]'
            iterative sum array for $\sum_{n=0}^{\infty}\dfrac{1}{n!}(\alpha\lambda)^n$.

        Example
        -------
        >>> T = 800
        >>> from solid_cinel.core.material.pdos import Pdos
        >>> pdos = Pdos.from_data(rho_in_energy, interv_in_energy)
        >>> tau1 = pdos._get_tau_1(T)
        >>> debye_waller_coeff = pdos.DebyeWallerCoeff(T)
        >>> alpha_grid = scale_grid(alpha0_, T)
        >>> beta_grid = scale_grid(beta0_, T)
        >>> S_mat, iter_sum = S._S_from_tau1(tau1, debye_waller_coeff, alpha_grid, beta_grid)
        >>> pd.DataFrame(S_mat.round(6)).iloc[:10, :5]
              0         1         2         3         4
        0  0.036967  0.037137  0.037308  0.037478  0.037644
        1  0.070694  0.071019  0.071345  0.071671  0.071988
        2  0.101393  0.101859  0.102326  0.102795  0.103249
        3  0.129265  0.129859  0.130455  0.131052  0.131631
        4  0.154499  0.155209  0.155921  0.156635  0.157327
        5  0.177272  0.178087  0.178904  0.179724  0.180517
        6  0.197752  0.198661  0.199573  0.200487  0.201372
        7  0.216097  0.217090  0.218086  0.219085  0.220053
        8  0.232453  0.233521  0.234593  0.235667  0.236708
        9  0.246960  0.248095  0.249234  0.250375  0.251481

        >>> pd.Series(np.exp(iter_sum.round(6))).iloc[:10]
        0    0.044821
        1    0.089642
        2    0.134463
        3    0.179284
        4    0.224106
        5    0.268927
        6    0.313748
        7    0.358569
        8    0.403390
        9    0.448211
        dtype: float64
        """
        iter_sum = np.log(alpha_grid * debye_waller_coeff)
        alpha_mul = np.exp(-alpha_grid * debye_waller_coeff + iter_sum)
        tau1_reshape = reshape_differential(tau1.index.values,
                                            tau1.values,
                                            beta_grid)
        S_values = np.outer(alpha_mul, tau1_reshape)
        return S_values, iter_sum

    @staticmethod
    def sum_rule_check(S) -> None:
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
        S : 'pd.DataFrame'
            S(alpha, beta) matrix.

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
            warnings.warn("Sum rule of S(alpha, -beta) not satisfied with an precision of 1.0e-3")
        return

    @staticmethod
    def normalization_check(S, pdos=None, T=None) -> None:
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
        S : 'pd.DataFrame'
            S(alpha, beta) matrix.

        Raises
        ------
        ValueError
            The normalization constrain is not satified.

        """
        normalization = S.apply(_normalization, axis="columns")
        if pdos and T:
            normalization /= (1 - np.exp(S.index.values * pdos.DebyeWallerCoeff(T)))
        if (abs(normalization - 1.0) > 5.0e-3).any():
            warnings.warn("Normalization of S(alpha, -beta) not satisfied with an precision of 1.0e-3")
        return

    def get_output_energy(self, T, incident_neutron_energy) -> np.array:
        """
        Based on the S(alpha, -beta) matrix, get the posible
        output energies for a incident neutron energy and that beta grid.
        .. math::
            E^\prime = \beta k_B T + E

        Parameters
        ----------
        T : 'float'
            Temperature in K.
        incident_neutron_energy : 'float'
            Incident neutron energy in eV.

        Example
        -------
        >>> T = 800
        >>> incident_neutron_energy = 0.33118
        >>> beta_grid = scale_grid(beta0_, T)
        >>> alpha_grid = scale_grid(alpha0_, T)
        >>> Sab = S.from_fgm(alpha_grid, beta_grid)
        >>> Sab.get_output_energy(T, incident_neutron_energy)[0:5]
        array([0.33118  , 0.3318125, 0.332445 , 0.3330775, 0.33371  ])
        """
        beta = self.data.columns.values
        return beta * const["Boltzmann constant in eV/K"][0] * T + incident_neutron_energy

    def get_theta(self, T, incident_neutron_energy, m) -> np.array:
        """
        Based on the S(alpha, -beta) matrix, get the posible scattering angles
        for a scattering atom, temperature and incident neutron energy.
        .. math::
            \mu = \frac{E^\prime + E - \alpha Ak_BT}{2\sqrt{E^\prime E}}
            \theta = \arccos(\mu)

        Parameters
        ----------
        T : 'float'
            Temperature in K.
        incident_neutron_energy : 'float'
            Incident neutron energy in eV.
        m : 'float'
            Atom mass, amu.

        Example
        -------
        >>> T = 800
        >>> m = 26.98153433356103
        >>> incident_neutron_energy = 0.33118
        >>> beta_grid = scale_grid(beta0_, T)
        >>> alpha_grid = scale_grid(alpha0_, T)
        >>> Sab = S.from_fgm(alpha_grid, beta_grid)
        >>> Sab.get_theta(T, incident_neutron_energy, m)[0:5].round(6)
        array([0.101125, 0.143002, 0.175125, 0.202199, 0.226045])
        """
        A = A = m / const["neutron mass in u"][0]
        E_prima = self.get_output_energy(T, incident_neutron_energy)
        alpha = self.data.index.values
        if len(E_prima) > len(alpha):
            E_prima = E_prima[:len(alpha)]
        elif len(E_prima) < len(alpha):
            alpha = alpha[:len(E_prima)]
        mu = E_prima + incident_neutron_energy - alpha * A * const["Boltzmann constant in eV/K"][0] * T
        mu /= (2 * np.sqrt(E_prima * incident_neutron_energy))
        return np.arccos(mu[abs(mu) <= 1])

    def get_inelastic_Xs(self, T, m, boundXs, incident_neutron_energy) -> pd.DataFrame:
        """
        Get inelastic scattering for a atom with a certain bound Xs and mass
        and for a certain incident energy.
        .. math::

        Parameters
        ----------
        T : 'float'
            Temperature in K.
        m : 'float'
            Atom mass, amu.
        boundXs : 'float'
            Bound total scattering cross section in barn.
        incident_neutron_energy : 'float'
            Incident neutron energy in eV.

        Example
        -------
        >>> T = 800
        >>> m = 26.98153433356103
        >>> boundXs = 1.5030808051112423
        >>> incident_neutron_energy = 0.33118
        >>> beta_grid = scale_grid(beta0_, T)
        >>> alpha_grid = scale_grid(alpha0_, T)
        >>> from solid_cinel.core.material.pdos import Pdos
        >>> pdos = Pdos.from_data(rho_in_energy, interv_in_energy)
        >>> Sab = S.from_pdos(alpha_grid, beta_grid, T, pdos, threshold=1.0e-14)
        >>> Sab.get_inelastic_Xs(T, m, boundXs, incident_neutron_energy).iloc[:5, :5].round(6)
        E_out     0.331180  0.331812  0.332445  0.333077  0.333710
        theta
        0.101125  0.414306  0.416519  0.418685  0.420825  0.422894
        0.143002  0.814356  0.818542  0.822529  0.826404  0.830112
        0.175125  1.200295  1.206236  1.211737  1.216984  1.221948
        0.202199  1.572293  1.579791  1.586531  1.592831  1.598713
        0.226045  1.930542  1.939419  1.947155  1.954222  1.960724
        """
        # Get the output energies and the scatterig angle
        E_prima = self.get_output_energy(T, incident_neutron_energy)
        theta = self.get_theta(T, incident_neutron_energy, m)
        if len(E_prima) > len(theta):
            E_prima = E_prima[:len(theta)]

        vector = boundXs / (2 * const["Boltzmann constant in eV/K"][0] * T) * np.sqrt(E_prima / incident_neutron_energy)

        inelastic_xs = vector * self.data.iloc[:len(theta), :len(E_prima)]
        inelastic_xs.index = pd.Index(theta, name="theta")
        inelastic_xs.columns = pd.Index(E_prima, name="E_out")
        return inelastic_xs


def _sum_rule(x) -> float:
    """
    Calculate the sum rule value for a fix alpha value.

    Parameters
    ----------
    x : 'pd.Series'
        S(alpha, beta) matrix values for fix alpha.


    Returns
    -------
    sum_rule_values : "float"
        Sum rule value for a fix alpha.

    Example
    -------
    >>> beta_grid = gen_beta(300)
    >>> alpha_grid = gen_alpha(300, 26)
    >>> s = S.from_fgm(alpha_grid, beta_grid).data
    >>> _sum_rule(s.iloc[1, ::]).round(6)
    0.001087
    """
    beta = x.index.values
    S_values = x.values
    sum_rule_values = trapezoid(beta * S_values * (1 - np.exp(-beta)), beta)
    return sum_rule_values


def _normalization(x) -> float:
    """
    Normalization rule value for a fix alpha value of the S(alpha, beta) matrix.

    Parameters
    ----------
    x : 'pd.Series'
        S(alpha, beta) matrix values for fix alpha.

    Returns
    -------
    normalization_values : "float"
        Normalization value for a fix alpha.

    Example
    -------
    >>> beta_grid = gen_beta(300)
    >>> alpha_grid = gen_alpha(300, 26)
    >>> s = S.from_fgm(alpha_grid, beta_grid).data
    >>> _normalization(s.iloc[0, ::]).round(6)
    1.0
    """
    beta = x.index.values
    S_asymm_values = x.values
    S = pd.Series((1 + np.exp(-beta)) * S_asymm_values, index=beta)
    return normalization_coeff(S)


def gen_beta(T, num_grid=400, mid_E=0.08,
             thermal_threshold=5., scale=False, **kwargs) -> np.ndarray:
    """
    Generate beta grid for a given temperature

    Parameters
    ----------
    T : 'float'
        Temperature in K.
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

    Example
    -------
    >>> gen_beta(300, num_grid=10).round(6)
    array([  0.      ,   0.515756,   1.031513,   1.547269,   2.063025,
             2.578782,   3.094538,  12.280683,  48.735922, 193.408635])
    """
    kT = const["Boltzmann constant in eV/K"][0] * T
    mid_beta = mid_E / kT
    max_beta = thermal_threshold / kT
    first_half = np.linspace(0, mid_beta,
                             num=int(num_grid * 0.6),
                             endpoint=False)
    second_half = np.logspace(np.log10(mid_beta), np.log10(max_beta),
                              num=int(num_grid * 0.4),
                              endpoint=True)
    beta_grid = np.concatenate((first_half, second_half))
    if scale:
        beta_grid = scale_grid(beta_grid, T, **kwargs)
    return beta_grid


def gen_alpha(T, atom_mass, num_grid=300, min_E=2.8e-3,
              thermal_threshold=5., scale=False, **kwargs) -> np.ndarray:
    """
    Generate a alpha grid for a given temperature and atomic mass.

    Parameters
    ----------
    T : 'float'
        Temperature in K.
    atom_mass : 'float'
        atomic mass in amu.
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

    Example
    -------
    >>> gen_alpha(300, 26, num_grid=10).round(6)
    array([1.0500000e-03, 3.2850000e-03, 1.0270000e-02, 3.2114000e-02,
           1.0041300e-01, 3.1397500e-01, 9.8174500e-01, 3.0697450e+00,
           9.5985550e+00, 3.0013001e+01])
    """
    A = atom_mass / const["neutron mass in u"][0]
    AkT = A * const["Boltzmann constant in eV/K"][0] * T
    min_alpha = min_E / 4 / AkT
    max_alpha = 4 * thermal_threshold / AkT
    alpha_grid = np.logspace(np.log10(min_alpha), np.log10(max_alpha),
                             num=num_grid)
    if scale:
        alpha_grid =  scale_grid(alpha_grid, T, **kwargs)
    return alpha_grid

def scale_grid(grid, T, therm=0.0253) -> np.ndarray:
    """
    Scale alpha or beta spectrum.

    Parameters
    ----------
    grid : 'np.ndarray' of 1D or 2D
        Alpha o Beta grid.
    T : 'float'
        Temperature in K.
    therm : 'float', optional
        factor for regrid alpha and beta. The default is 0.0253.

    Example
    -------
    >>> T = 300
    >>> alpha0 = gen_alpha(T, 26, num_grid=10)
    >>> scale_grid(alpha0, T).round(6)
    array([1.0280000e-03, 3.2140000e-03, 1.0051000e-02, 3.1428000e-02,
           9.8269000e-02, 3.0727100e-01, 9.6078300e-01, 3.0041990e+00,
           9.3936040e+00, 2.9372154e+01])
        
    >>> beta0 = gen_beta(T, num_grid=10)
    >>> scale_grid(beta0, T).round(6)
    array([  0.      ,   0.504744,   1.009488,   1.514231,   2.018975,
           2.523719,   3.028463,  12.018462,  47.695298, 189.278915])

    >>> grid0 = np.array([alpha0, beta0])
    >>> scale_grid(grid0, T).round(6)
    array([[1.02800000e-03, 3.21400000e-03, 1.00510000e-02, 3.14280000e-02,
            9.82690000e-02, 3.07271000e-01, 9.60783000e-01, 3.00419900e+00,
            9.39360400e+00, 2.93721540e+01],
           [0.00000000e+00, 5.04744000e-01, 1.00948800e+00, 1.51423100e+00,
            2.01897500e+00, 2.52371900e+00, 3.02846300e+00, 1.20184620e+01,
            4.76952980e+01, 1.89278915e+02]])
    """
    return grid * therm / (const["Boltzmann constant in eV/K"][0] * T)


def check_tau_n(tau_n, beta) -> None:
    """
    Check if the tau function created in solid_cinel.core._numba.tau_n_CPU is
    normalized to the unity.

    Parameters
    ----------
    tau_n : TYPE
        DESCRIPTION.
    beta : 1D iterable
        beta grid.

    Raises
    ------
    ValueError
        Tau function doesnt satisfy the normalization condition.
    """
    if sp.integrate.trapezoid(tau_n, x=beta) < 1.e-5:
        raise ValueError("Tau function doesnt satisfy the normalization condition")
    return
