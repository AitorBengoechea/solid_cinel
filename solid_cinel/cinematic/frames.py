# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:15:29 2023

@author: Aitor Bengoechea
"""
import numpy as np
import pandas as pd
from scipy.constants import physical_constants as const

from solid_cinel.cinematic.particles import Neutron, Nucleus, Particle
from solid_cinel.core.generic import sampling

# Constants
amu_to_ev = const["atomic mass unit-electron volt relationship"][0]
c = const["speed of light in vacuum"][0]
kb = const["Boltzmann constant in eV/K"][0]

# Test paratmeters
v_nucleus = np.array([267.27496695, 142.98091635])
mu = np.array([0.711655, -0.49416719])
muCm = np.array([0.91124234, -0.40826203])
phiCm = np.array([2.60926315, -2.12445998])


class Cm:
    """
    Class to store the cinematic properties of the centre of mass Frame. Since
    a collision in CM is assumed to be isotropic, all the angles in the CM frame
    are sampled randomly

    Attributes
    ----------
    samples : "int"
        Number of samples to generate random angles of the CM frame

    Properties
    ----------
    mu : "np.array", (samples,)
        Array of random cosines between the velocity of the neutron in the CM
        and the velocity of the nucleus in the CM in the collision plane
    phi : "np.array", (samples,)
        Angle between the velocity of the neutron in the CM and the velocity of
        the nucleus in the CM outside the collision plane
    """
    def __init__(self, samples: int = 1000) -> None:
        """
        Initialize the class Cm

        Parameters
        ----------
        samples : "int", optional
            Number of samples to generate random angles of the CM frame
        """
        self.samples = samples
        pass
    @property
    def mu(self) -> np.array:
        """
        Cosine of the angle between the velocity of the neutron in the CM and
        the velocity of the nucleus in the CM

        Returns
        -------
        "np.array", (samples,)
            Array of random cosines between the velocity of the neutron in the
            CM and the velocity of the nucleus in the CM in the collision plane
        """
        return sampling(1, self.samples) * 2 - 1
    @property
    def phi(self) -> np.array:
        """
        Angle between the velocity of the neutron in the CM and the velocity of
        the nucleus in the CM outside the collision plane
        Returns
        -------
        "np.array", (samples,)
            Array of random angles between the velocity of the neutron in the CM
            and the velocity of the nucleus in the CM outside the collision
            plane
        """
        return (sampling(1, self.samples) * 2 - 1) * np.pi


class Tr:
    """
    Class to store the cinematic properties of the Target at Rest frame.

    Attributes
    ----------
    mu : "np.array"
        Cosine of the velocity of the target nucleus and the velocity of the
        neutron in LAB frame.
    muCm : "np.array"
        Cosine of the velocity of the neutron in the CM and the velocity of the
        nucleus in the CM in the collision plane
    v_nucleus : "np.array"
        Velocity of the nucleus in the LAB frame according to the
        Maxwell-Boltzmann velocity distribution
    samples: "int"
        Number of samples to generate random velocities of the nucleus if the
        previous one is not provided.

    Methods
    -------
    v : "np.array", (samples,)
        Velocity of the neutron in the TR frame
    vprime : "np.array", (samples,)
        Velocity of the neutron after the colission in the TR frame

    Properties
    ----------
    muprime : "np.array", (samples,)
        Cosine of the angle between the velocities of the neutron before and
        after the collision in the TR frame
    """
    def __init__(self, M: float, T: float, mu: np.array = None,
                 muCm: np.array = None, v_nucleus: np.array = None,
                 samples: int = 1000):
        """
        Initialize the class Tr. If the arguments are not provided by the user,
        they are calculated randomly in other classes.

        Parameters
        ----------
        M : "float"
            Mass of the nucleus in Amu
        T : "float"
            Temperature of the nucleus in K
        mu : "np.array", optional
            Cosine of the velocity of the target nucleus and the velocity of the
            neutron in LAB frame. By default None, so its calculated randomly.
        muCm : "np.array", optional
            Cosine of the velocity of the neutron in the CM and the velocity of
            the nucleus in the CM in the collision . By default None, so it is
            randomly calculated in Cm class.
        v_nucleus : "np.array", optional
            Velocity of the nucleus in the LAB frame according to the
            Maxwell-Boltzmann velocity distribution. By default None, so it is
            randomly calculated in Nucleus class.
        samples : "int", optional
            Number of samples to generate random velocities of the nucleus if
            the previous arguments are not provided, by default 1000.
        """
        self.mu = mu if hasattr(mu, "__len__") else Cm(samples).mu
        self.muCm = muCm if hasattr(muCm, "__len__") else Cm(samples).mu
        self.mass_ratio = Particle(M).m / Particle(const["neutron mass in u"][0]).m
        self.v_nucleus = v_nucleus if hasattr(v_nucleus, "__len__") else Nucleus(M, samples=samples).v(T)
        self.samples = len(self.mu)  # For consistency
        pass

    def v(self, Eneutron: float) -> np.array:
        """
        Velocity of the neutron in the TR frame
        .. math::
            v = \sqrt{v_{neutron}^2 + v_{nucleus}^2 - 2v_{neutron}v_{nucleus}\mu}

        Parameters
        ----------
        Eneutron : "float"
            Energy of the neutron in the LAB frame in eV

        Returns
        -------
        "np.array", (samples,)
            Array of the velocities of the neutron in the TR frame for different
            nucleus velocities in the LAB frame

        Examples
        --------
        >>> Target_at_Rest = Tr(238, 300, v_nucleus = v_nucleus, mu = mu, muCm = muCm)
        >>> E = 6.6
        >>> Target_at_Rest.v(E).round(2)
        array([35344.3 , 35604.88])
        """
        v_neutron = Neutron(Eneutron).v
        return np.sqrt(v_neutron ** 2 + self.v_nucleus ** 2 - 2 * self.v_nucleus * v_neutron * self.mu)

    def vprime(self, Eneutron: float) -> np.array:
        """
        Velocity of the neutron in the Tr frame after the collision
        .. math::
            A = M / m
            v' = \sqrt{1 + 2\mu_{CM}A + A^2} * v_{TR} / (A + 1)

        Parameters
        ----------
        Eneutron : "float"
            Energy of the neutron in the LAB frame in eV

        Returns
        -------
        "np.array", (samples,)
            Array of the velocities of the neutron in the TR frame after the
            collision for different nucleus velocities and angles between the
            velocity of the neutron and the velocity of the nucleus in the LAB
            frame

        Examples
        --------
        >>> Target_at_Rest = Tr(238, 300, v_nucleus = v_nucleus, mu = mu, muCm = muCm)
        >>> E = 6.6
        >>> Target_at_Rest.vprime(E).round(2)
        array([35331.11, 35393.54])
        """
        v = self.v(Eneutron)
        vprime = np.sqrt(1 + 2 * self.muCm * self.mass_ratio + self.mass_ratio ** 2) * v
        vprime /= (self.mass_ratio + 1)
        return vprime

    @property
    def muprime(self) -> np.array:
        """
        Cosine of the angle between the velocities of the neutron before and
        after the collision in the TR
        .. math::
            A = M / m
            \mu' = (1 + \mu_{CM}A) / \sqrt{1 + 2\mu_{CM}A + A^2}

        Returns
        -------
        "np.array", (samples,)
            Array of angles between the velocities of the neutron before and
            after the collision in the TR frame for different nucleus velocities
            and angles between the velocity of the neutron and the velocity of
            the nucleus in the LAB frame.

        Examples
        --------
        >>> Target_at_Rest = Tr(238, 300, v_nucleus = v_nucleus, mu = mu, muCm = muCm)
        >>> Target_at_Rest.muprime.round(2)
        array([ 0.91, -0.4 ])
        """
        muprime = 1 + self.muCm * self.mass_ratio
        muprime /= np.sqrt(1 + 2 * self.muCm * self.mass_ratio + self.mass_ratio ** 2)
        return muprime


class Lab:
    """
    Class to store the cinematic properties of the LAB frame

    Attributes
    ----------
    M : "float"
        Mass of the nucleus in Amu
    T : "float"
        Temperature of the nucleus in K
    mu : "np.array"
        Cosine of the velocity of the target nucleus and the velocity of the
        neutron in LAB frame
    muCm : "np.array"
        Cosine of the velocity of the neutron in the CM and the velocity of the
        nucleus in the CM in the collision
    v_nucleus : "np.array"
        Velocity of the nucleus in the LAB frame according to the
        Maxwell-Boltzmann velocity distribution
    samples : "int"
        Number of samples to generate random velocities of the nucleus if
        the previous arguments are not provided

    Methods
    -------
    v -> np.array:
        Velocity of the neutron in the LAB frame
    vprime -> np.array:
        Velocity of the neutron in the LAB frame after the collision
    muprime -> np.array:
        Cosine of the angle between the velocities of the neutron before and
        after the collision in the LAB
    run -> pd.DataFrame:
        Run the simulation of the collision in the LAB frame
    """
    def __init__(self, M: float, T: float, mu: np.array = None,
                 muCm: np.array = None, phiCm: np.array = None,
                 v_nucleus: np.array = None, samples: int = 1000) -> None:
        """
        Class to simulate 2 body collisions in the LAB frame according to the
        SVT model

        Parameters
        ----------
        M : "float"
            Mass of the nucleus in Amu
        T : "float"
            Temperature of the nucleus in K
        mu : "np.array", optional
            Cosine of the velocity of the target nucleus and the velocity of the
            neutron in LAB frame. By default None, so its calculated randomly.
        muCm : "np.array", optional
            Cosine of the velocity of the neutron in the CM and the velocity of
            the nucleus in the CM in the collision . By default None, so it is
            randomly calculated in Cm class.
        phiCm : "np.array", optional
            Angle between the velocity of the neutron in the CM and the velocity
            of the nucleus in the CM outside the collision plane. By default
             None, so it is randomly calculated in Cm class.
        v_nucleus : "np.array", optional
            Velocity of the nucleus in the LAB frame according to the
            Maxwell-Boltzmann velocity distribution. By default None, so it is
            randomly calculated in Nucleus class.
        E_nucleus : "np.array", optional
            Energy of the nucleus in Ev
        samples : "int", optional
            Number of samples to generate random velocities of the nucleus if
            the previous arguments are not provided, by default 1000.
        """
        self.mu = mu if hasattr(mu, "__len__") else Cm(samples).mu
        self.muCm = muCm if hasattr(muCm, "__len__") else Cm(samples).mu
        self.phiCm = phiCm if hasattr(phiCm, "__len__") else Cm(samples).mu
        self.v_nucleus = v_nucleus if hasattr(v_nucleus, "__len__") else Nucleus(M, samples=samples).v(T)
        self.E_nucleus = Nucleus(M).get_E(self.v_nucleus)
        self.samples = len(self.mu)  # For consistency
        self.Tr = Tr(M, T, mu=self.mu, muCm=self.muCm, v_nucleus=self.v_nucleus, samples=self.samples)
        pass

    def mu_relative(self, Eneutron: float) -> np.array:
        """
        Relative cosine between the relative velocity of the neutron in the
        Target at Rest Frame and the velocity of the nucleus in the LAB frame.
        .. math::
            \mu_{r} = \frac{\mu v_{neutron} - v_{nucleus}}{\sqrt{v_{neutron}^2 + v_{nucleus}^2 - 2v_{neutron}v_{nucleus}\mu}}

        Parameters
        ----------
        Eneutron : "float"
            Energy of the neutron in the LAB frame in eV

        Returns
        -------
        "np.array", (samples,)
            Array of the relative cosine between the relative velocity in the TR
            frame and the velocity of the nucleus in the LAB frame.

        Examples
        --------
        >>> lab = Lab(238, 300, v_nucleus = v_nucleus, mu = mu, muCm = muCm, phiCm = phiCm)
        >>> E = 6.6
        >>> lab.mu_relative(E).round(2)
        array([ 0.71, -0.5 ])
        """
        v_neutron = Neutron(Eneutron).v
        mu_relative = self.mu * v_neutron - self.v_nucleus
        mu_relative /= np.sqrt(v_neutron ** 2 + self.v_nucleus ** 2 - 2 * v_neutron * self.v_nucleus * self.mu)
        return mu_relative

    def mu_Tr_v_nucl(self, Eneutron: float):
        """
        Cosine of the angle between the velocity of the neutron after the
        collision in the TR frame and the velocity of the nucleus in the LAB
        frame
        .. math::
            \mu'_{V} = \mu'_{CM}\mu_{r} + \sqrt{1 - \mu'_{CM}^2}\sqrt{1 - \mu_{r}^2}\cos(\phi_{CM})

        Parameters
        ----------
        Eneutron : "float"
            Energy of the neutron in the LAB frame in eV

        Returns
        -------
        "np.array", (samples,)
            Array of the cosine of the angle between the velocity of the neutron
            after the collision in the TR frame and the velocity of the nucleus
            in the LAB frame

        Examples
        --------
        >>> lab = Lab(238, 300, v_nucleus = v_nucleus, mu = mu, muCm = muCm, phiCm = phiCm)
        >>> E = 6.6
        >>> lab.mu_Tr_v_nucl(E).round(2)
        array([ 0.4 , -0.22])
        """
        mu_relative = self.mu_relative(Eneutron)
        Trmuprime = self.Tr.muprime
        return Trmuprime * mu_relative + np.sqrt(1 - Trmuprime ** 2) * np.sqrt(1 - mu_relative ** 2) * np.cos(self.phiCm)

    def vprime(self, Eneutron: float) -> np.array:
        """
        Velocity of the neutron after the collision in the LAB frame
        .. math::
            v' = \sqrt{(v^{\prime}_{TR})^2 + v_{nucleus}^2 + 2v_{nucleus}v^{\prime}_{TR}\mu'_{TR}}}

        Parameters
        ----------
        Eneutron : "float"
            Energy of the neutron in the LAB frame in eV

        Returns
        -------
        "np.array", (samples,)
            Array of the velocity of the neutron after the collision in the LAB
            frame

        Examples
        --------
        >>> lab = Lab(238, 300, v_nucleus = v_nucleus, mu = mu, muCm = muCm, phiCm = phiCm)
        >>> E = 6.6
        >>> lab.vprime(E).round(2)
        array([35437.77, 35362.94])
        """
        mu_Tr_v_nucl = self.mu_Tr_v_nucl(Eneutron)
        Tr_vprime = self.Tr.vprime(Eneutron)
        return np.sqrt(Tr_vprime ** 2 + 2 * Tr_vprime * self.v_nucleus * mu_Tr_v_nucl + self.v_nucleus ** 2)

    def muprime(self, Eneutron: float) -> np.array:
        """
        Cosine of the angle between the velocity of the neutron after and before
        the collision in the LAB frame
        .. math::
            \mu' = \frac{v_{TR}v^\prime_{TR}\mu'_{TR} + v_{TR}V\mu_r + v^\prime_{TR}V\mu'_{V} + V^2}{vv^\prime}

        Parameters
        ----------
        Eneutron : "float"
            Energy of the neutron in the LAB frame in eV

        Returns
        -------
        "np.array", (samples,)
            Array of the cosine of the angle between the velocity of the neutron
            after and before the collision in the LAB frame

        Examples
        --------
        >>> lab = Lab(238, 300, v_nucleus = v_nucleus, mu = mu, muCm = muCm, phiCm = phiCm)
        >>> E = 6.6
        >>> lab.muprime(E).round(2)
        array([ 0.91, -0.41])
        """
        # angles:
        mu_relative = self.mu_relative(Eneutron)
        mu_Tr_v_nucl = self.mu_Tr_v_nucl(Eneutron)
        Tr_muprime = self.Tr.muprime
        # velocities:
        Tr_v = self.Tr.v(Eneutron)
        Tr_v_prime = self.Tr.vprime(Eneutron)
        v_neutron = Neutron(Eneutron).v
        v_neutron_prime = self.vprime(Eneutron)
        # calculation:
        muprime = Tr_v * Tr_v_prime * Tr_muprime
        muprime += Tr_v * self.v_nucleus * mu_relative
        muprime += Tr_v_prime * self.v_nucleus * mu_Tr_v_nucl
        muprime += self.v_nucleus ** 2
        return muprime / (v_neutron * v_neutron_prime)

    def run(self, Eneutron: float, v: bool = True, E: bool = True,
            degree: bool = False) -> pd.DataFrame:
        """
        Run the calculation of the kinematics of the collision

        Parameters
        ----------
        Eneutron : "float"
            Energy of the neutron in the LAB frame in eV
        v : "bool", optional
            If True, the velocities are going to be showed in the DataFrame
        E : "bool", optional
            If True, the energies are going to be showed in the DataFrame
        degree : "bool", optional
            If True, the angles are going to be showed in degrees in the
            DataFrame

        Returns
        -------
        "pd.DataFrame"
            DataFrame with the kinematics of the collision

        Examples
        --------
        >>> lab = Lab(238, 300, v_nucleus = v_nucleus, mu = mu, muCm = muCm, phiCm = phiCm)
        >>> E = 6.6
        >>> lab.run(E).round(2)
             mu  muprime  v_nucleus  v_neutron_prime  E_nucleus  E_neutron_prime
        0  0.71     0.91     267.27         35437.77       0.09             6.56
        1 -0.49    -0.41     142.98         35362.94       0.03             6.54

        >>> lab.run(E, degree = True).round(2)
           theta  thetaprime  v_nucleus  v_neutron_prime  E_nucleus  E_neutron_prime
        0   0.78        0.42     267.27         35437.77       0.09             6.56
        1   2.09        1.99     142.98         35362.94       0.03             6.54

        >>> lab.run(E, v = False).round(2)
             mu  muprime  E_nucleus  E_neutron_prime
        0  0.71     0.91       0.09             6.56
        1 -0.49    -0.41       0.03             6.54

        >>> lab.run(E, E = False).round(2)
             mu  muprime  v_nucleus  v_neutron_prime
        0  0.71     0.91     267.27         35437.77
        1 -0.49    -0.41     142.98         35362.94
        """
        if degree:
            result = {"theta": np.arccos(self.mu),
                      "thetaprime": np.arccos(self.muprime(Eneutron))}
        else:
            result = {"mu": self.mu,
                      "muprime": self.muprime(Eneutron)}
        if v:
            result["v_nucleus"] = self.v_nucleus
            result["v_neutron_prime"] = self.vprime(Eneutron)
        if E:
            result["E_nucleus"] = self.E_nucleus
            result["E_neutron_prime"] = Neutron.from_v(self.vprime(Eneutron)).E

        return pd.DataFrame(result).dropna()
