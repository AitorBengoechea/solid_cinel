# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:46:42 2022
@author: Aitor Bengoechea
"""
from scipy.constants import physical_constants as const
from scipy.integrate import trapezoid
from solid_cinel.core.generic import normalization_coeff
import numpy as np
import pandas as pd
import scipy as sp


class S():
    """
    Class containing all the methods and properties of a asymmetric
    S(alpha, beta) matrix.
    """
    def __init__(self, *args, **kwargs):
        self.data = pd.DataFrame(*args, **kwargs)

    @property
    def data(self) -> pd.DataFrame:
        """
        Dataframe with the S(alpha, beta) matrix values.
        """
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
        df_ = pd.DataFrame(df)
        df_.index.name = "alpha"
        df_.columns.name = "beta"
        # Normalization constrains:
        self.normalization_check(df_)
        self.sum_rule_check(df_)
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
        >>> S.from_model(alpha_grid, beta_grid).to_sym().iloc[:10, :5].round(6)
        beta	      0.000000	0.012894	0.025788	0.038682	0.051576
        alpha					
        0.001050	8.701463	8.310148	7.332597	5.977775	4.502503
        0.001087	8.553363	8.179618	7.246379	5.947043	4.521401
        0.001125	8.407781	8.050773	7.159884	5.914070	4.537109
        0.001164	8.264674	7.923611	7.073187	5.878976	4.549701
        0.001205	8.124000	7.798127	6.986363	5.841878	4.559259
        0.001247	7.985718	7.674316	6.899481	5.802889	4.565867
        0.001291	7.849787	7.552171	6.812607	5.762123	4.569612
        0.001336	7.716166	7.431685	6.725805	5.719689	4.570585
        0.001382	7.584817	7.312851	6.639132	5.675693	4.568876
        0.001431	7.455701	7.195659	6.552646	5.630238	4.564579
        """
        S_asym = self.data
        if detail_balance:
            beta = self.data.columns.values
            S_sym = S_asym * np.exp(-beta)
        else:
            S_sym = S_asym
        return S_sym

    @classmethod
    def from_model(cls, alpha_grid, beta_grid, model="FGM", **kwargs):
        """
        Generate S(alpha, -beta) matrix using physical models for a given
        alpha and beta grid.

        Available models:
            - FGM: Free Gas Model.
                .. math::
                    S_t(\alpha,\,\beta)=\dfrac{1}{\sqrt{4\pi w_t\alpha}}\exp\left(-\dfrac{(w_t\alpha+\beta)^2}{4w_t\alpha}\right)\end{equation}
            - SCT: Short Collision Time

        Parameters
        ----------
        alpha_grid : 1D iterable
            Alpha grid.
        beta_grid : 1D iterable
            beta grid.
        model : 'str', optional
            The model to calculate matrix values. The default is "FGM".

        Parameters for FGM
        ------------------
        w_t: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Parameters for SCT
        ------------------
        ratio: 'float'
            Ratio between Effective Temperature and Temperature
        w_s: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Example
        -------
        FGM:
        >>> beta_grid = gen_beta(300)
        >>> alpha_grid = gen_alpha(300, 26)
        >>> S.from_model(alpha_grid, beta_grid).data.iloc[:10, :5].round(6)
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
        Dont fit the normalization and sum rule with the correct precision
        >>> #ratio = 1.0880348914731839
        >>> #S = S.from_model(alpha_grid, beta_grid, model="SCT", ratio=ratio)
        """
        model_ = model.lower()
        if model_ == "fgm":
            w_t = kwargs.get("w_t", 1)
            f = lambda i, j: np.exp(-(alpha_grid[i] * w_t - beta_grid[j]) ** 2 / (4 * w_t * alpha_grid[i])) / np.sqrt(4 * np.pi * w_t * alpha_grid[i])
        elif model_ == "sct":
            w_s = kwargs.get("w_s", 1)
            ratio = kwargs.pop("ratio")
            f = lambda i, j: np.exp(-(alpha_grid[i] * w_s - beta_grid[j]) ** 2 / (4 * ratio * w_s * alpha_grid[i])) / np.sqrt(4 * np.pi * w_s * alpha_grid[i] * ratio)

        S_values = np.fromfunction(np.vectorize(f),
                                   (len(alpha_grid), len(beta_grid)),
                                   dtype=int
                                   )
        return cls(S_values, index=alpha_grid, columns=beta_grid)

    @staticmethod
    def sum_rule_check(S) -> None:
        """
        Check if the S(alpha, beta) matrix satifies the sum rule constrain.
        .. math::
            \int_{-\infty}^{\infty}\beta S(\alpha,\,\beta)d\beta = \int_{0}^{\infty}\beta S(\alpha,\,-\beta)(-1+\exp(-\beta))d\beta = -\alpha

        Parameters
        ----------
        S : 'pd.DataFrame'
            S(alpha, beta) matrix.

        Raises
        ------
        ValueError
            The sum rule constrain is not satified.

        """
        sum_rule = S.apply(_sum_rule, axis="columns") - S.index.values
        if (abs(sum_rule) > 5.0e-3).any():
            raise ValueError("Sum rule of S(alpha, beta) not satisfied")
        return

    @staticmethod
    def normalization_check(S) -> None:
        """
        Check if the S(alpha, beta) matrix satifies the normalization constrain.
        .. math::
            \int_{-\infty}^{\infty}S(\alpha,\,\beta)d\beta = \int_{0}^{\infty}S(\alpha,\,-\beta)(1+\exp(-\beta))d\beta = 1

        Parameters
        ----------
        S : 'pd.DataFrame'
            S(alpha, beta) matrix.

        Raises
        ------
        ValueError
            The normalization constrain is not satified.

        """
        normalization = S.apply(_normalization, axis="columns") - 1.0
        if (abs(normalization) > 5.0e-3).any():
            raise ValueError("Normalization of S(alpha, beta) not satisfied")
        return

    @staticmethod
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
        >>> S.scale_grid(alpha0, T).round(6)
        array([1.0280000e-03, 3.2140000e-03, 1.0051000e-02, 3.1428000e-02,
               9.8269000e-02, 3.0727100e-01, 9.6078300e-01, 3.0041990e+00,
               9.3936040e+00, 2.9372154e+01])
        
        >>> beta0 = gen_beta(T, num_grid=10)
        >>> S.scale_grid(beta0, T).round(6)
        array([  0.      ,   0.504744,   1.009488,   1.514231,   2.018975,
                 2.523719,   3.028463,  12.018462,  47.695298, 189.278915])

        >>> grid0 = np.array([alpha0, beta0])
        >>> S.scale_grid(grid0, T).round(6)
        array([[1.02800000e-03, 3.21400000e-03, 1.00510000e-02, 3.14280000e-02,
                9.82690000e-02, 3.07271000e-01, 9.60783000e-01, 3.00419900e+00,
                9.39360400e+00, 2.93721540e+01],
               [0.00000000e+00, 5.04744000e-01, 1.00948800e+00, 1.51423100e+00,
                2.01897500e+00, 2.52371900e+00, 3.02846300e+00, 1.20184620e+01,
                4.76952980e+01, 1.89278915e+02]])
        """
        return grid * therm / (const["Boltzmann constant in eV/K"][0] * T)

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
    >>> s = S.from_model(alpha_grid, beta_grid).data
    >>> _sum_rule(s.iloc[0, ::]).round(6)
    0.00105
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
    >>> s = S.from_model(alpha_grid, beta_grid).data
    >>> _normalization(s.iloc[0, ::]).round(6)
    1.0
    """
    beta = x.index.values
    S_asymm_values = x.values
    S = pd.Series((1 + np.exp(-beta)) * S_asymm_values, index=beta)
    return normalization_coeff(S)


def gen_beta(T, num_grid=400, mid_E=0.08,
             thermal_threshold=5.) -> np.ndarray:
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
    return np.concatenate((first_half, second_half))


def gen_alpha(T, atom_mass, num_grid=300, min_E=2.8e-3,
              thermal_threshold=5.) -> np.ndarray:
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
    return np.logspace(np.log10(min_alpha), np.log10(max_alpha), num=num_grid)
