import numpy as np
import pandas as pd
import os
from scipy.constants import physical_constants as const
from solid_cinel.core.generic import integrate, reshape_differential
from typing import Iterable
import warnings


# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]


class ScatFuncSD:
    """
    Single Differencial (angle or Outgoing energy) scattering function base class
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the ScatFuncSD class

        Parameters
        ----------
        args : Iterable
            The scattering function data
        kwargs : dict
            Optional arguments for the construction of the pd.Series
        """
        self.data = pd.Series(*args, **kwargs)

    @property
    def data(self) -> pd.Series:
        """
        The scattering function data

        Returns
        -------
        pd.Series
            The scattering function data
        """
        return self._data

    @data.setter
    def data(self, pdf: Iterable):
        """
        Set the scattering function data and check the normalization

        Parameters
        ----------
        pdf : pd.Series
            The scattering function data

        """
        pdf_ = pd.Series(pdf).sort_index()
        normalization = integrate(pdf_)
        if abs(normalization - 1) >= 0.1 and pdf_.name >= 0.005:
            raise ValueError("The scattering function is not normalized (normalization coeff < 0.9)")
        elif abs(normalization - 1) >= 0.01:
            warnings.warn("Normalizaton not satisfied with 1% accuracy")
        self._data = pdf_

    @classmethod
    def from_MD(cls, Ein: float, M: float, T: float, Eout: np.array):
        """
        Calculate the scattering function using Maxwellian velocity distribution
        and angular integration
        .. math::
            S(E, E^\prime) = \frac{1}{2}\sqrt{\frac{M}{m\pi k_BT}}\frac{\sqrt{E^\prime}}{E}\left(exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} - \sqrt{E^\prime}\right)^2 \right) - exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} + \sqrt{E^\prime}\right)^2 \right)\right)

        Parameters
        ----------
        Ein : float
            Incident energy of the neutron
        M : float
            Mass of the material in amu
        T : float
            Temperature of the material in K
        Eout : np.array
            Outgoing energy grid

        Returns
        -------
        ScatFuncSD
            The scattering function for the given temperature, incident energy
            and mass using Maxwellian velocity distribution and angular
            integration

        Examples
        --------
        # Generate Broadening test results:
        >>> Ein = 36.68723
        >>> Eout = np.linspace(Ein * 0.98 , Ein * 1.02, 1000)
        >>> M = 238.05077040419212
        >>> T = 300
        >>> pdf = ScatFuncSD.from_MD(Ein, M, T, Eout)
        >>> pdf.data.iloc[::100]
        Eout
        35.953485    8.937086e-15
        36.100381    1.841784e-09
        36.247277    2.425252e-05
        36.394173    2.074937e-02
        36.541069    1.172637e+00
        36.687964    4.449812e+00
        36.834860    1.152331e+00
        36.981756    2.069367e-02
        37.128652    2.618312e-05
        37.275548    2.371152e-09
        Name: 36.68723, dtype: float64
        """
        exp_negative = pd.Series(
            np.exp(- M / (m * kb * T) * (np.sqrt(Ein) - np.sqrt(Eout)) ** 2),
            index=pd.Index(Eout, name="Eout"), name=Ein)
        exp_positive = pd.Series(
            np.exp(- M / (m * kb * T) * (np.sqrt(Ein) + np.sqrt(Eout)) ** 2),
            index=pd.Index(Eout, name="Eout"), name=Ein)
        pdf = 1 / 2 * (exp_negative - exp_positive) * np.sqrt(Eout) / Ein
        pdf *= np.sqrt(M / (np.pi * m * kb * T))
        return cls(pdf)

    def convolve(self, xs: pd.Series, integral: bool = False) -> pd.Series:
        """
        Convolve the scattering function with a cross section

        Parameters
        ----------
        xs : pd.Series
            The cross section to convolve with the scattering function

        Returns
        -------
        pd.Series
            The convolved cross section

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("scatfunc.py", ""))
        >>> os.chdir("../../../data/xs/U238/")
        >>> xs_0K = pd.read_csv("u238.0.2", sep="    ", header=None, engine="python").set_index(0).drop([2], axis=1)
        >>> os.chdir(wd)
        >>> xs_0K = xs_0K[~xs_0K.index.duplicated(keep='first')]

        # Generate Scattering function:
        >>> Ein = 36.68723
        >>> Eout = np.linspace(Ein * 0.98 , Ein * 1.02, 1000)
        >>> M = 238.05077040419212
        >>> T = 300
        >>> scattering_function = ScatFuncSD.from_MD(Ein, M, T, Eout)

        # Convolve with 0K cross section:
        >>> scattering_function.convolve(xs_0K).iloc[::100]
        Eout
        35.953485    7.227742e-14
        36.100381    3.567348e-08
        36.247277    1.191135e-03
        36.394173    3.113927e+00
        36.541069    9.254287e+02
        36.687964    1.079050e+05
        36.834860    1.182981e+03
        36.981756    6.837390e+00
        37.128652    4.644441e-03
        37.275548    2.789231e-07
        Name: 36.68723, dtype: float64

        >>> round(scattering_function.convolve(xs_0K, integral=True), 2)
        7905.42
        """
        pdf = self.data
        xs_reshaped = reshape_differential(xs.index.values,
                                           xs.values,
                                           pdf.index.values)[::, 0]
        xs_convol = pdf * xs_reshaped
        if integral:
            return integrate(xs_convol)
        else:
            return xs_convol



class ScatFuncDD:
    """
    Double Differencial (angle, Outgoing energy) scattering function base class
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the ScatFuncSD class

        Parameters
        ----------
        args : Iterable
            The scattering function data
        kwargs : dict
            Optional arguments for the construction of the pd.Series
        """
        self.data = pd.DataFrame(*args, **kwargs)

    @property
    def data(self) -> pd.DataFrame:
        """
        The scattering function data

        Returns
        -------
        pd.Series
            The scattering function data
        """
        return self._data

    @data.setter
    def data(self, dd_pdf: Iterable):
        """
        Set the scattering function data and check the normalization

        Parameters
        ----------
        dd_pdf : pd.Series
            Double differential scattering function data

        """
        dd_pdf_ = pd.DataFrame(dd_pdf).sort_index(axis=0).sort_index(axis=1)
        dd_pdf_.index.name = "mu"
        dd_pdf_.columns.name = "Eout"
        normalization = integrate(dd_pdf_.apply(integrate))
        if abs(normalization - 1) >= 0.1 and dd_pdf_.name >= 0.005:
            raise ValueError("The scattering function is not normalized (normalization coeff < 0.9)")
        elif abs(normalization - 1) >= 0.01:
            warnings.warn("Normalizaton not satisfied with 1% accuracy")
        self._data = dd_pdf_

    @classmethod
    def from_Sab(cls, Ein: float, M: float, T: float, Eout: np.array,
                 mu: np.array):
        return

class ScatFunc(ScatFuncSD, ScatFuncDD):
    """
    Scattering function base class
    """
    def __init__(self, *args, **kwargs):
        pass






