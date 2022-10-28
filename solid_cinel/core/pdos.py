# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:46:42 2022
@author: Aitor Bengoechea
"""

import pandas as pd
import numpy as np
import scipy as sp
def pdos():
    """
    
    Returns
    -------
    None.
    """
    kb = 8.6173303e-5  # eV/K

    def __init__(self, *args, **kwargs):
        self.rho = pd.Series(*args, **kwargs)

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, rho):
        rho_ = pd.Series(rho, dtype=float, name="rho in energy")

        if not len(rho_.shape) == 1:
            raise TypeError("Rho must have one dimension")

        if abs(1 - sp.integrate.trapezoid(rho_.values, x=rho_.index)) > 1.0e-6:
            dx = rho_.index[1] - rho_.index[0]
            rho_norm = self.normalization(rho_.values, dx)
            rho_.values = rho_norm
        self._rho = rho_

    @classmethod
    def from_data(cls, rho, interval_energy):
        rho_ = np.array(rho)
        index = np.arange(len(rho_)) * interval_energy
        return cls(rho, index=index)

    def change_grid(self, T=None):
        old_grid = self.rho.index
        rho_values = self.rho.values
        if T:
            grid = old_grid / (kb * T)
        return self.__class__(rho_values, index=grid)

    @staticmethod
    def normalization(y, dx, type="trapezoidal"):
        if type == "trapezoidal":
            y_norm = sp.integrate.trapezoid(y, dx=dx)
        elif type == "simpson":
            y_norm = sp.integrate.simpson(y, dx=dx)
        elif type == "romberg":
            y_norm = sp.integrate.romb(y, dx=dx)
        return y / y_norm
Footer