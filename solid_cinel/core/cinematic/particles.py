# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:15:29 2023

@author: Aitor Bengoechea
"""
from scipy.constants import physical_constants as const
from scipy.stats import maxwell
from solid_cinel.core.generic import sampling
import numpy as np

# Constants
amu_to_ev = const["atomic mass unit-electron volt relationship"][0]
c = const["speed of light in vacuum"][0]
kb = const["Boltzmann constant in eV/K"][0]

class Particle:
    """
    Class to store common properties of all the particles

    Attributes
    ----------
    m : "float"
        Mass of the particle in ev * s^2 / m^2
    """
    def __init__(self, m: float):
        """
        Initialize the class Particle

        Parameters
        ----------
        m : "float"
            Mass of the particle in Amu
        """
        self.m = m * amu_to_ev / c ** 2
        pass


class Neutron(Particle):
    """
    Class to store kinematics properties of the neutrons

    Attributes
    ----------
    E : "float"
        Energy of the neutron in eV

    Methods
    -------
    from_v -> "Neutron"
        Initialize the class Neutron from the velocity of the neutron

    Properties
    ----------
    v -> "float"
        Velocity of the neutron in m/s
    wavelength -> "float"
        Wavelength of the neutron in Angstrom
    d_min -> "float"
        Minimum distance for the LEAPR module of NJOY in Angstrom
    """
    def __init__(self, E: float):
        """
        Initialize the class Neutron
        Parameters
        ----------
        E : Energy of the neutron in eV
        """
        super().__init__(const["neutron mass in u"][0])
        self.E = E

    @classmethod
    def from_v(cls, v: float):
        """
        Initialize the class Neutron from the velocity of the neutron

        Parameters
        ----------
        v : float
            Velocity of the neutron in m/s

        Returns
        -------
        "Neutron"
            Neutron with the velocity v

        Example
        -------
        >>> v = 35534.004895483995
        >>> v_class = Neutron.from_v(v)
        >>> assert round(v, 2) == round(v_class.v, 2)
        """
        m_neutron = const["neutron mass in u"][0] * amu_to_ev / c ** 2
        return cls(v ** 2 * m_neutron / 2)

    @property
    def v(self) -> float:
        """
        Velocity of the neutron in m/s

        Returns
        -------
        "float"
            Velocity of the neutron in m/s

        Example
        -------
        >>> E = 6.6
        >>> v = 35534.004895483995
        >>> E_class = Neutron.from_v(v)
        >>> assert round(E, 2) == round(E_class.E, 2)
        """
        return np.sqrt(2 * self.E / self.m)

    @property
    def wavelength(self) -> float:
        """
        Wavelength of the neutron in Angstrom

        Returns
        -------
        "float"
            Wavelength of the neutron in Angstrom
        """
        walength = 2 * np.pi * const["reduced Planck constant in eV s"][0] / np.sqrt(2 * self.m * self.E)
        return walength * 1.0e10

    @property
    def d_min(self) -> float:
        """
        Minimum distance for the LEAPR module of NJOY in Angstrom

        Returns
        -------
        "float"
            Minimum distance for the LEAPR module of NJOY in Angstrom
        """
        return 0.5 * self.wavelength * 0.95


class Nucleus(Particle):
    """
    Class to store the cinematic properties nucleus in a crytal

    Attributes
    ----------
    m : "float"
        Mass of the nucleus in ev * s^2 / m^2
    sampling : "np.array"
        Array of random numbers between 0 and 1 to sample the Maxwell-Boltzmann
        velocity distribution

    Methods
    -------
    v : "float"
        Velocity of the nucleus in m/s according to the Maxwell-Boltzmann distribution
    """
    def __init__(self, M: float, samples: int = 1000):
        """
        Initialize the class Nucleus

        Parameters
        ----------
        M : "float"
            Mass of the nucleus in Amu
        samples : "int", optional
            Number of samples to generate random velocities of the nucleus
            according to Maxwell-Boltzmann distribution, by default 1000.
        """
        super().__init__(m=M)
        self.samples = samples
        pass

    def v(self, T: float, d: int = 1) -> np.array:
        """
        Random Velocity of the nucleus in m/s according to the Maxwell-Boltzmann
        distribution
        .. math::
            f(v) = \left(\frac{M}{2\pi k_BT}\right)^{3/2}exp\left(-\frac{M}{2k_B T}(v^\prime)^2\right)

        Parameters
        ----------
        T : "float"
            Temperature of the nucleus in K
        d : "int", optional
            Number of dimensions of the sampling, by default 1

        Returns
        -------
        "np.array", (samples,)
            Array of random velocities of the nucleus in m/s according to the
            Maxwell-Boltzmann velocity distribution
        """
        a = np.sqrt(kb * T / self.m)
        return maxwell(scale=a).ppf(sampling(d, self.samples))

    def get_E(self, v: np.array) -> np.array:
        """
        Kinetic energy of the nucleus in eV

        Parameters
        ----------
        v : "np.array", (N,)
            Array of velocities of the nucleus in m/s

        Returns
        -------
        "np.array", (N,)
            Array of kinetic energies of the nucleus in eV

        Example
        -------
        >>> Nucleus(238).get_E(np.array([10000, 20000]))
        array([123.33480887, 493.33923547])
        >>> Nucleus(1).get_E(np.array([10000, 20000]))
        array([0.51821348, 2.07285393])
        """
        return v ** 2 * self.m / 2
