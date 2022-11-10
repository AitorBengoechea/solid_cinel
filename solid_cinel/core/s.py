# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:46:42 2022
@author: Aitor Bengoechea
"""
from scipy.constants import physical_constants as const
from scipy.integrate import trapezoid
import numpy as np
import pandas as pd
import scipy as sp


class S():
    """
    Class containing all the methods and properties of a S(alpha, beta) matrix.
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
        self.normalization_check(self._data)
        self.sum_rule_check(self._data)
        self._data = df_

    @classmethod
    def from_FGM(cls, alpha_grid, beta_grid, w_t= 1.0):
        """
        Generate a S(alpha, beta) matrix for a given alpha and beta grid.
        .. math::
            S_t(\alpha,\,\beta)=\dfrac{1}{\sqrt{4\pi w_t\alpha}}\exp\left(-\dfrac{(w_t\alpha+\beta)^2}{4w_t\alpha}\right)\end{equation}

        Parameters
        ----------
        alpha_grid : 1D iterable
            Alpha grid.
        beta_grid : 1D iterable
            beta grid.
        w_t: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1
        """
        f = lambda i, j: np.exp(-(alpha_grid[i] * w_t - beta_grid[j]) ** 2 / (4 * w_t * alpha_grid[i])) / np.sqrt(4 * np.pi * w_t * alpha_grid[i])
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
        sum_rule = S.apply(_sum_rule, axis="columns") - S.index_values
        if (abs(sum_rule) > 1.0e-14):
            raise ValueError("Sum rule of S(alpha, beta) not satisfied")
        return

    @staticmethod
    def normalization_check(S):
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
        if (abs(normalization) > 1.0e-14):
            raise ValueError("Normalization of S(alpha, beta) not satisfied")
        return


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
    """
    beta = x.index.values
    S_values = x.values
    normalization_values = trapezoid((1 + np.exp(-beta)) * S_values, beta)
    return normalization_values


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
    """
    A = atom_mass / const["neutron mass in u"][0]
    AkT = A * const["Boltzmann constant in eV/K"][0] * T
    min_alpha = min_E / 4 / AkT
    max_alpha = 4 * thermal_threshold / AkT
    return np.logspace(np.log10(min_alpha), np.log10(max_alpha), num=num_grid)
