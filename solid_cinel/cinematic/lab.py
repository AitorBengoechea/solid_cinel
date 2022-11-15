# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:18:52 2022

@author: AB272525
"""

from scipy.constants import physical_constants as const
import scipy as sp
import numpy as np

class Neutron():
    def __init__(self, energy):
        """
        Initialize the class Neutron.

        Parameters
        ----------
        energy : 'float'
            Neutron energy in eV.
        """
        self.E = energy
        self.mass = const["neutron mass in u"][0]

    @property
    def wavelength(self) -> float:
        """
        Neutron wavelength in Angstrom

        Example
        -------
        >>> n = Neutron(2.301)
        >>> n.wavelength.round(6)
        0.188551
        """
        mass = self.mass * const["atomic mass unit-electron volt relationship"][0] / sp.constants.c ** 2
        walength = 2 * np.pi * const["reduced Planck constant in eV s"][0] / np.sqrt(2 * mass * self.E)
        return walength * 1.0e10
    