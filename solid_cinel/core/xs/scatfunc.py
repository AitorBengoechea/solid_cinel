"""
Python for working with Diferential XS.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
from scipy.constants import physical_constants as const
from solid_cinel.core.scattering_function import TransferFunc, DynamicStruc
from solid_cinel.core.scattering_function.alpha import get_alpha
from solid_cinel.core.generic import integrate, reshift, interpolation
import os
from typing import Iterable

# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]

# Avoid numba fast math:
nb.config.FASTMATH_DEFAULT = False

class Xs0K:
    def __init__(self, M:float, *args, **kwargs):
        self.M = M
        self.data = pd.Series(*args, **kwargs)

    @property
    def data(self) -> pd.Series:
        return self._data

    @data.setter
    def data(self, xs: pd.Series):
        # Sort the Series by index
        xs_ = pd.Series(xs, name="0").sort_index()
        xs_.index.name = "Ein"

        # Drop duplicates, keeping the first occurrence
        self._data = xs_.drop_duplicates(keep='first')

    @property
    def values(self) -> np.ndarray:
        return self.data.values
    @property
    def EinGrid(self) -> np.ndarray:
        return self.data.index.values

    @staticmethod
    def read_xs(filename: str, header: [int, list] = None,
                usecols: [int, list] = [0, 1], index_col: int = 0,
                engine: str = "python") -> pd.Series:
        """
        Read the xs data from a file

        Parameters
        ----------
        filename: str
            The filename of the xs data
        header: int, list, optional
            The header of the file. The default is None, so no header is used.
        usecols: int, list, optional
            The columns to use. The default is [0, 1], so the first two columns
            are used.
        index_col: int, optional
            The index column. The default is 0, so the first column is used as
            index.
        engine: str, optional
            The engine to use. The default is "python".

        Returns
        -------
        pd.Series
            The xs data
        """
        # Read the data from the file into a pandas DataFrame
        xsData = pd.read_csv(filename, sep='\s+', header=header, index_col=index_col,
                             usecols=usecols, engine=engine).squeeze("columns")

        xsData.index.name = "E"
        # Ensure not duplicated index and if they are duplicated, take the first
        xsData = xsData.reset_index().drop_duplicates(subset='E', keep='first')

        return xsData.set_index('E').squeeze("columns")

    @classmethod
    def from_file(cls, filename: str, M: float, **kwargs):
        return cls(M, cls.read_xs(filename, **kwargs))

    def interpolate(self, EinSmall: [float, np.array], inplace: bool = False,
                    values: bool = False) -> ["Xs0K", pd.Series, np.ndarray]:
        """

        Parameters
        ----------
        EinSmall :
        inplace :
        values :

        Returns
        -------
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("scatfunc.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs0K = Xs0K.from_file("u238.0.2", M)
        >>> os.chdir(wd)

        >>> xs0K.interpolate(1., values=False)
        1.0    9.269424
        dtype: float64

        >>> xs0K.interpolate([1.0, 2.0], values=False)
        1.0    9.269424
        2.0    9.085342
        dtype: float64
        """
        if values and inplace:
            raise ValueError("Values and inplace cannot be True at the same time")
        EinSmall = np.array(EinSmall) if hasattr(EinSmall, "__len__") else np.array([EinSmall])
        xs0Kinterp = interpolation(self.data, EinSmall, values=values)
        return self.update(xs0Kinterp) if inplace else xs0Kinterp

    def update(self, dataNew: pd.Series) -> "Xs0K":
            self.data = dataNew
            return self



class ScatFunc:
    """
    Class for the Scattering function of inelastic scattering
    """
    def __init__(self, xs0K: Xs0K, Ein: float, T: float, *args, **kwargs):
        """
        Class for the Scattering function of inelastic scattering

        Parameters
        ----------
        xs : pd.Series or pd.DataFrame, (N,) or (M, N)
            0K xs data for the given material in barns. If the cross
            section is a matrix, the scattering function is convolved directly
            with xs. If the cross section is a vector, the scattering function
            is convolved with the cross section for each outgoing energy or
            with the Exs introduced by the user.
        Exs : np.ndarray, optional, (N,) or (M, N)
            Displazed Energy grid of the cross section. If not provided, the
            energy grid of the scattering function is used.
        theta : np.ndarray, (M,)
            The neutron outgoing angle grid in degrees (0, 180]
        kwargs : dict
            Extra parameters for the selected algorithm
        """
        # Atributes of the scattering function (Change in these parameters will
        # change the scattering function):
        self.xs0K = xs0K
        self.Ein = Ein
        self.T = T
        # The dxs data:
        self.data = pd.Series(*args, **kwargs)

    @property
    def data(self) -> pd.Series:
        """
        Diferential xs data.

        Returns
        -------
        pd.Series
            Diferential xs data
        """
        return self._data

    @data.setter
    def data(self, pdf: Iterable):
        """
        Set the scattering function data and check the normalization.

        Parameters
        ----------
        pdf : pd.Series
            The scattering function data

        """
        pdf_ = pd.Series(pdf).sort_index()
        pdf_.index.name = "Eout"
        self._data = pdf_.drop_duplicates(keep='first')

    def update(self, dataNew: pd.Series) -> "Xs0K":
            self.data = dataNew
            return self
    @property
    def integral(self) -> float:
        """
        The integral value of the Scattering function

        Returns
        -------
        float
            The integral value of the Diferential xs

        Examples
        --------
        # 0K xs data for U238:
        
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("scatfunc.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs0K = Xs0K.read_xs("u238.0.2")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)

        # SIGMA1 algorithm:
        >>> float(round(ScatFunc.from_sigma1(xs0K, Ein, M, T, Eout).integral, 2))
        9.09

        # DOPUSH algorithm:
        >>> theta = np.arange(0, 180, 1)[1::]
        >>> float(round(ScatFunc.from_alpha0(xs0K, Ein, M, T, Eout, theta, model="fgm").integral, 2))
        9.09
        """
        return integrate(self.data)

    @property
    def prob(self) -> dict:
        """
        Get the upscattering and downscattering probabilities for the selected
        Ein, T and M.

        Returns
        -------
        dict
            Dictionary with the upscattering and downscattering probabilities

        Examples
        --------
        # 0K xs data for U238:
        
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("scatfunc.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs0K = Xs0K.read_xs("u238.0.2")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> dxs = ScatFunc.from_sigma1(xs0K, Ein, M, T, Eout)
        >>> float(round(dxs.prob["upscattering"], 6))
        0.505184
        >>> float(round(dxs.prob["downscattering"], 6))
        0.490636
        >>> float(round(dxs.prob["Ein=Eout"], 6))
        0.004179
        """
        # Get the integral value:
        integral = self.integral

        # Get the outgoig energy grid:
        Eout = self.data.index.values

        # Get the upscattering and downscattering probabilities:
        up = integrate(self.data.loc[Eout > self.Ein]) / integral
        down = integrate(self.data.loc[Eout < self.Ein]) / integral

        return {"upscattering": up,  "downscattering": down, "Ein=Eout": 1.0 - up - down}

    @property
    def pdf(self) -> pd.Series:
        """
        Get the probability density function of the Scattering function

        Returns
        -------
        pd.Series
            The probability density function of the Scattering function

        Examples
        --------
        # 0K xs data for U238:
        
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("scatfunc.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs0K = Xs0K.read_xs("u238.0.2")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> dxs = ScatFunc.from_sigma1(xs0K, Ein, M, T, Eout)

        # SIGMA1 algorithm:
        >>> dxs.pdf.iloc[::100]
        Eout
        1.80000     0.000005
        1.84004     0.001091
        1.88008     0.063339
        1.92012     1.102673
        1.96016     5.974293
        2.00020    10.438638
        2.04024     6.084029
        2.08028     1.221555
        2.12032     0.087118
        2.16036     0.002272
        dtype: float64
        """
        return self.data / self.integral


    @classmethod
    def from_sigma1(cls, xs0K: pd.Series, Ein: float, M: float, T: float, Eout: np.ndarray):
        """
        Generate the Scattering function for inelastic scattering from sigma1
        ..math::
            \frac{d\sigma_T(E)}{dE^\prime} = \frac{1}{2}\sqrt{\frac{M}{m\pi k_BT}}\frac{\sqrt{E^\prime}}{E}\sigma_0(E^\prime)\left(exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} - \sqrt{E^\prime}\right)^2 \right) - exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} + \sqrt{E^\prime}\right)^2 \right)\right)

        Parameters
        ----------
        xs0K : pd.Series, (Z,)
            0K xs data for the given material in barns
        Ein : float
        The incident energy of the neutron in eV
        M : float
            Mass of the material in amu
        T : float
            Temperature of the material in K
        Eout : np.ndarray, (N,)
            The neutron outgoing energy grid in eV

        Returns
        -------
        ScatFunc
            Scattering function for inelastic scattering

        Examples
        --------
        # 0K xs data for U238:

        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("scatfunc.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs0K = Xs0K.read_xs("u238.0.2")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)

        # SIGMA1 algorithm:
        >>> ScatFunc.from_sigma1(xs0K, Ein, M, T, Eout).data.iloc[::100]
        Eout
        1.80000     0.000049
        1.84004     0.009909
        1.88008     0.575486
        1.92012    10.018740
        1.96016    54.281606
        2.00020    94.844029
        2.04024    55.278649
        2.08028    11.098885
        2.12032     0.791546
        2.16036     0.020645
        dtype: float64
        """
        # Intiliaze the xs0K data:
        xs0K = Xs0K(M, xs0K)

        # Get the transfer function:
        transferFunc = TransferFunc.from_sigma1(Ein, M, T, Eout)

        # Get the Scattering function:
        dxs = transferFunc.data * xs0K.interpolate(transferFunc.Eout, values=True)

        return cls(xs0K, Ein, T,  dxs)

    @classmethod
    def from_alpha(cls, xs0K: pd.Series, alpha: float, Ein: float, M: float,
                   T: float, Eout: np.ndarray, *args, model: str = "fgm", **kwargs):
        """
        Generate the Scattering function for inelastic scattering from the
        S(alpha, beta) tables distribution

        Parameters
        ----------
        xs0K : pd.Series, (Z,)
            0K xs data for the given material in barns
        Ein : float
            The incident energy of the neutron in eV
        M : float
            Mass of the material in amu
        T : float
            Temperature of the material in K
        Eout : np.ndarray, (N,)
            The neutron outgoing energy grid in eV
        model : str
            The model used to calculate the S(alpha, beta) distribution. The available models are:
                - "fgm": Free Gas Model (default)
                - "sct": Short Collision Time
                - "pdos": Phonon Density of States

        Parameters for sct
        ------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.

        Parameters for pdos
        -------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        threshold : 'float', optional
            Minimun value to take into account in the creation of tauN
            functions. For T>200 is convenient to set into 1.0e-14 to speed up
            the calculations. The default is 0.0.
        decimal: 'float'
            Decimal precision for the calculation of the expansion order.
            The default is 1.0e-6.
        order_max: 'int'
            Maximun expansion order. The default is 5000.

        Returns
        -------
        ScatFunc
            Scattering function for inelastic scattering

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("scatfunc.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs0K = Xs0K.read_xs("u238.0.2")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> alpha = Ein / (kb * T) / M

        # alpha0 algorithm:
        >>> ScatFunc.from_alpha(xs0K, alpha, Ein, M, T, Eout, model="fgm").data.iloc[::100]
        Eout
        1.80000     0.000299
        1.84004     0.034346
        1.88008     1.303107
        1.92012    16.344437
        1.96016    67.757179
        2.00020    92.829654
        2.04024    42.029652
        2.08028     6.289029
        2.12032     0.311053
        2.16036     0.005082
        dtype: float64

        # Using the Short Collision Time model:
        >>> from solid_cinel.core.material import Pdos
        >>> from solid_cinel.tests.materials.UO2.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> ScatFunc.from_alpha(xs0K, alpha, Ein, M, T, Eout, pdos, model="sct").data.iloc[::100]
        Eout
        1.80000     0.000312
        1.84004     0.035241
        1.88008     1.320349
        1.92012    16.416150
        1.96016    67.719720
        2.00020    92.676130
        2.04024    42.007690
        2.08028     6.317055
        2.12032     0.315203
        2.16036     0.005216
        dtype: float64
        """
        # Intiliaze the xs0K data:
        xs0K = Xs0K(M, xs0K)

        # Get the transfer function:
        transferFunc = TransferFunc.from_alpha(alpha, Ein, M, T, Eout, *args,
                                               model=model, **kwargs)

        # Get the recoil energy:
        recoil = alpha * kb * T

        # Get the Scattering function:
        dxs = transferFunc.data * xs0K.interpolate(transferFunc.Eout + recoil, values=True)

        return cls(xs0K, Ein, T, dxs)

    @classmethod
    def from_alpha0(cls, xs0K: pd.Series, Ein: float, M: float, T: float, Eout: np.ndarray,
                   theta: np.ndarray, *args, model: str = "fgm", **kwargs):
        """
        Generate the Scattering function for inelastic scattering from the most
        similar distribution of the S(alpha, -beta) tables and sigma1 algorithm
        ..math::
            \frac{d\sigma_T(E)}{dE^\prime} = \frac{\sigma(E^\prime + R)}{2 * k_B * T}\sqrt{\frac{E^\prime}{E}} S(\alpha(\theta, E^\prime, E, M, T), \beta( E^\prime, E, T))

        Parameters for fgm, sct and pdos models
        ----------------------------------------
        xs0K : pd.Series, (Z,)
            0K xs data for the given material in barns
        Ein : float
        The incident energy of the neutron in eV
        M : float
            Mass of the material in amu
        T : float
            Temperature of the material in K
        Eout : np.ndarray, (N,)
            The neutron outgoing energy grid in eV
        theta : np.ndarray, (M,)
            The neutron outgoing angle grid in degrees (0, 180]
        model : str
            The model used to calculate the S(alpha, beta) distribution. The available models are:
                - "fgm": Free Gas Model (default)
                - "sct": Short Collision Time
                - "pdos": Phonon Density of States

        Parameters for sct
        ------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.

        Parameters for pdos
        -------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        threshold : 'float', optional
            Minimun value to take into account in the creation of tauN
            functions. For T>200 is convenient to set into 1.0e-14 to speed up
            the calculations. The default is 0.0.
        decimal: 'float'
            Decimal precision for the calculation of the expansion order.
            The default is 1.0e-6.
        order_max: 'int'
            Maximun expansion order. The default is 5000.

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("scatfunc.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs0K = Xs0K.read_xs("u238.0.2")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> theta = np.arange(1, 180, 1)
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)

        # alpha0 algorithm:
        >>> ScatFunc.from_alpha0(xs0K, Ein, M, T, Eout, theta, model="fgm").data.iloc[::100]
        Eout
        1.80000     0.000287
        1.84004     0.033460
        1.88008     1.285044
        1.92012    16.258397
        1.96016    67.751989
        2.00020    92.982545
        2.04024    42.024971
        2.08028     6.255488
        2.12032     0.306710
        2.16036     0.004950
        dtype: float64

        # Using the Short Collision Time model:
        >>> from solid_cinel.core.material import Pdos
        >>> from solid_cinel.tests.materials.UO2.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> ScatFunc.from_alpha0(xs0K, Ein, M, T, Eout, theta, pdos, model="sct").data.iloc[::100]
        Eout
        1.80000     0.000299
        1.84004     0.034323
        1.88008     1.301847
        1.92012    16.328835
        1.96016    67.714845
        2.00020    92.831027
        2.04024    42.003188
        2.08028     6.283015
        2.12032     0.310753
        2.16036     0.005079
        dtype: float64

        # Using the Phonon Density of States model:
        >>> ScatFunc.from_alpha0(xs0K, Ein, M, T, Eout, theta, pdos, model="pdos").data.iloc[::100]
        Eout
        1.80000     0.005152
        1.84004     0.112088
        1.88008     1.741582
        1.92012    15.942947
        1.96016    66.211564
        2.00020    95.747483
        2.04024    41.058942
        2.08028     6.141493
        2.12032     0.418065
        2.16036     0.016800
        dtype: float64
        """
        # Calculate alpha0 values from the scattering function:
        alpha0 = DynamicStruc.from_model(Ein, M, T, Eout, theta, *args,
                                         model=model, **kwargs).alpha0

        # Get dxs based on the alpha0:
        return cls.from_alpha(xs0K, alpha0, Ein, M, T, Eout, *args, model=model, **kwargs)

    @classmethod
    def from_theta(cls, xs0K: pd.Series, Ein: float, M: float, T: float, Eout: np.ndarray,
                   theta: float, *args, model: str = "fgm", **kwargs):
        """
        Generate the Scattering function for inelastic scattering from the
        Scattering function angle distribution

        Parameters
        ----------
        xs0K : pd.Series, (Z,)
            0K xs data for the given material in barns
        Ein : float
            The incident energy of the neutron in eV
        M : float
            The mass of the target material in amu
        T : float
            Temperature of the material in K
        Eout : np.array
            The neutron outgoing energy grid in eV
        theta : float
            The angle of the scattering in degrees
        model : str
            The model used to generate the S(alpha, beta) table. The available
            models are:
                - "pdos": Phonon expansion model
                - "fgm" : Free Gas Model (Default)
                - "sct" : Short Collision Time model

        Parameters for SCT model
        ------------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        ws: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.
        twt: 'float', optional
            twt for the effective temperature. For solid is 1.

        Parameters for PDOS model
        -------------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        nphonon: 'int', optional
            Phonon expansion order. The default is None and the order is
            calculated using the get_expansionOrder function.
        decimal: 'float', optional
            Decimal precision for the calculation of the expansion order.
            The default is 1.0e-6.
        order_max: 'int', optional
            Maximun expansion order. The default is 5000.
        threshold: 'float', optional
            Minimun value to take into account in the creation of tauN
            functions

        Returns
        -------
        TransferFunc
            Double differential scattering scattering function

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("scatfunc.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs0K = Xs0K.read_xs("u238.0.2")
        >>> os.chdir(wd)

        >>> Ein = 7.2
        >>> Eout = np.array([7.10, 7.15, 7.2, 7.25, 7.3157])
        >>> T = 1000
        >>> theta = 15

        # Using the Free Gas Model:
        >>> ScatFunc.from_theta(xs0K, Ein, M, T, Eout, theta, model="fgm").data.round(6)
        Eout
        7.1000      0.000277
        7.1500      8.291518
        7.2000    212.369147
        7.2500      5.032810
        7.3157      0.000001
        Name: 15, dtype: float64
        """
        # Intiliaze the xs0K data:
        xs0K = Xs0K(M, xs0K)

        # Get the transfer function:
        transferFunc = TransferFunc.from_theta(Ein, M, T, Eout, theta, *args,
                                               model=model, **kwargs)

        # Get the recoil energy:
        recoil = get_alpha(Ein, T, M, Eout, np.cos(np.deg2rad(theta)))
        recoil *= kb * T

        # Get the Scattering function:
        dxs = transferFunc.data * xs0K.interpolate(transferFunc.Eout + recoil, values=True)

        return cls(xs0K, Ein, T, dxs)

    @classmethod
    def from_sab(cls, xs0K: pd.Series, Ein: float, M: float, T: float, Eout: np.ndarray,
                 theta: np.ndarray, *args, model: str = "fgm", recoil: bool = True,
                 **kwargs):
        """
        Generate the Scattering function for inelastic scattering from the
        Dynamic Structure Factor.

        Parameters
        ----------
        xs0K : pd.Series, (Z,)
            0K xs data for the given material in barns
        Ein : float
            The incident energy of the neutron in eV
        M : float
            The mass of the target material in amu
        T : float
            Temperature of the material in K
        Eout : np.ndarray
            The neutron outgoing energy grid in eV
        theta : np.ndarray
            Grid of angle of the scattering angle
        model: str
            The model used to generate the S(alpha, beta) table. The available
            models are:
                - "pdos": Phonon expansion model
                - "fgm" : Free Gas Model (Default)
                - "sct" : Short Collision Time model

        Parameters for SCT model
        ------------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        ws: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.
        twt: 'float', optional
            twt for the effective temperature. For solid is 1.

        Parameters for PDOS model
        -------------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        nphonon: 'int', optional
            Phonon expansion order. The default is None and the order is
            calculated using the get_expansionOrder function.
        decimal: 'float', optional
            Decimal precision for the calculation of the expansion order.
            The default is 1.0e-6.
        order_max: 'int', optional
            Maximun expansion order. The default is 5000.
        threshold: 'float', optional
            Minimun value to take into account in the creation of tauN
            functions

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("scatfunc.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs0K = Xs0K.read_xs("u238.0.2")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> theta = np.arange(1, 180, 1)
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)

        # Using the Free Gas Model(NO RECOIL):
        >>> dxs = ScatFunc.from_sab(xs0K, Ein, M, T, Eout, theta, model="fgm", recoil=False)
        >>> dxs.data.iloc[::100]
        Eout
        1.80000     0.768794
        1.84004     3.231998
        1.88008    10.451361
        1.92012    26.580672
        1.96016    54.522950
        2.00020    91.812241
        2.04024    34.506930
        2.08028    10.974592
        2.12032     2.920481
        2.16036     0.643604
        dtype: float64


        # Using the Free Gas Model(RECOIL):
        >>> dxs = ScatFunc.from_sab(xs0K, Ein, M, T, Eout, theta, model="fgm")
        >>> dxs.data.iloc[::100]
        Eout
        1.80000     0.768342
        1.84004     3.230182
        1.88008    10.445725
        1.92012    26.567992
        1.96016    54.502029
        2.00020    91.788547
        2.04024    34.492890
        2.08028    10.968947
        2.12032     2.918748
        2.16036     0.643167
        dtype: float64


        # Using the Short Collision Time model:
        >>> from solid_cinel.core.material import Pdos
        >>> from solid_cinel.tests.materials.UO2.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> dxs = ScatFunc.from_sab(xs0K, Ein, M, T, Eout, theta, pdos, model="sct")
        >>> dxs.data.iloc[::100]
        Eout
        1.80000     0.775840
        1.84004     3.248590
        1.88008    10.473227
        1.92012    26.580418
        1.96016    54.452587
        2.00020    91.640126
        2.04024    34.517269
        2.08028    11.008760
        2.12032     2.939884
        2.16036     0.650626
        dtype: float64

        # Using the Phonon Density of States model:
        >>> dxs = ScatFunc.from_sab(xs0K, Ein, M, T, Eout, theta, pdos, model="pdos")
        >>> dxs.data.iloc[::100]
        Eout
        1.80000     1.170555
        1.84004     3.940598
        1.88008    11.186093
        1.92012    26.701503
        1.96016    53.725774
        2.00020    80.771003
        2.04024    34.120291
        2.08028    10.823388
        2.12032     2.896455
        2.16036     0.651860
        dtype: float64
        """
        # Intiliaze the xs0K data:
        xs0K = Xs0K(M, xs0K)

        # Calculate the Dynamic Structure Factor:
        dynStruc = DynamicStruc.from_model(Ein, M, T, Eout, theta, *args,
                                           model=model, **kwargs)

        # Get the recoil energy if needed:
        if recoil:
            xs0Kinterp = xs0K.interpolate(Eout + dynStruc.recoil(), values=True)
            dxs = (dynStruc.data * xs0Kinterp).apply(integrate)
        else:
            transferFunc = dynStruc.to_ScatFunc
            dxs = transferFunc.data * xs0K.interpolate(transferFunc.Eout, values=True)

        return cls(xs0K, Ein, T, dxs)

    def shift(self, dx: [float, np.ndarray, pd.DataFrame]):
        """
        Shift the Scattering function in the given axis and interpolate to get
        the values of the original axis.

        Parameters
        ----------
        dx : float or np.ndarray or pd.Series or pd.DataFrame
            The shift value in the given axis. If a pd.DataFrame is given, the shift value is calculated according to
            the index or the columns of the pd.DataFrame (next argument to select).
        axis : str, optional
            The axis to shift the Double Scattering function. The default is "Eout".

        Returns
        -------
        DDxs
            The shifted Double Scattering function values in the original axis

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("scatfunc.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs0K = Xs0K.read_xs("u238.0.2")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> dxs = ScatFunc.from_sigma1(xs0K, Ein, M, T, Eout)
        >>> dxs.data.iloc[::200].round(6)
        Eout
        1.80000     0.000049
        1.88008     0.575486
        1.96016    54.281606
        2.04024    55.278649
        2.12032     0.791546
        dtype: float64

        # Shift the DDXS with float:
        >>> recoil = kb * T / M
        >>> dxs.shift(recoil).data.iloc[::200].round(6)
        Eout
        1.80000     0.000000
        1.88008     0.557797
        1.96016    53.733203
        2.04024    55.817580
        2.12032     0.814398
        dtype: float64

        # Shift the DDXS in the Eout axis:
        >>> recoil = Eout * kb * T / M
        >>> dxs.shift(recoil).data.iloc[::200].round(6)
        Eout
        1.800000     0.000000
        1.880480     0.544490
        1.960561    53.265537
        2.040641    56.321152
        2.120721     0.838164
        dtype: float64
        """
        # copy data to avoid changing the original data:
        dxs = self.data.copy()

        # Check the dx:
        dx_ = check_dx(self.data, dx, 0)

        # Shift the data:
        if isinstance(dx, float) or isinstance(dx, int):
            dxs = reshift(dxs, dx_)
        else:
            dxs.loc[dx_.index] = reshift(dxs.loc[dx_.index], dx_)

        return self.update(dxs)


def check_dx(data: [pd.DataFrame, pd.Series],
             dx: [float, np.ndarray, pd.DataFrame],
             axis: [str, int]) -> [float, pd.Series, pd.DataFrame]:
    """
    Check the dx value to shift the Double Scattering function and return the value
    in the correct format for the shift aplicattion.

    Parameters
    ----------
    data : pd.DataFrame, pd.Series
        Double or Single Scattering function data to shift
    dx : float, np.ndarray, pd.DataFrame
        Value to shift the data
    axis : str, int
        Axis to shift the data. Available options are "Eout", "mu" or 0, 1
        respectively.

    Returns
    -------
    float, pd.Series, pd.DataFrame
        Value to shift the data in the correct format

    Examples
    --------
    >>> data = pd.DataFrame([[1, 1, 1]] * 3, index=[-1, 0, 1], columns=[1, 2, 3])
    >>> check_dx(data, 0.1, "Eout")
    0.1

    >>> check_dx(data, pd.Series([1, 1], index=[[0, -1]]), "mu")
     0    1
    -1    1
    dtype: int64

    >>> check_dx(data, [1, 1, 1], "mu")
    -1    1
     0    1
     1    1
    dtype: int64

    >>> check_dx(data, [1, 1, 1], 1)
    1    1
    2    1
    3    1
    dtype: int64

    >>> check_dx(data, [[1, 1, 1]] * 3, 0)
        1	2	3
    -1	1	1	1
    0	1	1	1
    1	1	1	1
    """
    if isinstance(dx, float) or isinstance(dx, int) or isinstance(dx, pd.Series) or isinstance(dx, pd.DataFrame):
        return dx
    elif len(np.array(dx).shape) == 1:
        axis_ = 1 if axis == "Eout" else 0 if axis == "mu" else axis
        return pd.Series(dx, index=data.index if axis_ == 0 else data.columns)
    else:
        return pd.DataFrame(dx, index=data.index, columns=data.columns)