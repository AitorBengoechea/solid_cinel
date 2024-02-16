"""
Python for working with Diferential XS.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
from scipy.constants import physical_constants as const
from solid_cinel.core.scattering_function import ScatFunc
from solid_cinel.core.scattering_function.alpha import get_gressier_recoil
from solid_cinel.core.generic import integrate, reshift
import os

from typing import Iterable

# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]

# Avoid numba fast math:
nb.config.FASTMATH_DEFAULT = False

# Example variables:
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
3.550530E+00 3.349917E+00 2.768379E+00 2.177488E+00
1.856123E+00 1.622775E+00 1.445254E+00 1.300794E+00
1.180078E+00 1.075748E+00 9.928057E-01 9.238564E-01
8.577708E-01 8.073819E-01 7.634820E-01 7.172257E-01
6.728183E-01 6.251482E-01 5.496737E-01 4.992486E-01
3.945195E-01 2.206960E-01 1.452214E-01 1.246671E-01
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


class Dxs:
    """
    Class for the differential cross section for elastic scattering
    """
    def __init__(self, Ein: float, T: float, M: float, algorithm: str, *args, **kwargs):
        """
        Class for the Double differential cross section for elastic scattering

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
        algorithm : 'str'
            The algorithm to use for the double differencial doppler broadened
            elastic cross section. The available algorithms are:
                - "sigma1": sigma1 algorithm from NJOY2016 manual
                - "sab": S(alpha, -beta) tables for ddxs
                - "dopush": From the chosen S(alpha, -beta) model, the
                            distribution more similar to sigma1 is chosen and a
                            recoil energy
                - "courcelle": Fourier double-Laplace transform of a 4-point
        kwargs : dict
            Extra parameters for the selected algorithm
        """
        # Atributes of the scattering function (Change in these parameters will
        # change the scattering function):
        self.Ein = Ein
        self.T = T
        self.M = M
        self.algorithm = algorithm
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
        self._data = pdf_


    @classmethod
    def from_sigma1(cls, xs_0K: pd.Series, Ein: float, M: float, T: float, Eout: np.ndarray):
        """
        Generate the Differential xs for elastic scattering from sigma1
        ..math::
            \frac{d\sigma_T(E)}{dE^\prime} = \frac{1}{2}\sqrt{\frac{M}{m\pi k_BT}}\frac{\sqrt{E^\prime}}{E}\sigma_0(E^\prime)\left(exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} - \sqrt{E^\prime}\right)^2 \right) - exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} + \sqrt{E^\prime}\right)^2 \right)\right)

        Parameters
        ----------
        xs_0K : pd.Series, (Z,)
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
        Dxs
            Differential cross section for elastic scattering

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("dxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212

        # SIGMA1 algorithm:
        >>> Dxs.from_sigma1(xs_0K, Ein, M, T, Eout).data.iloc[::100]
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
        scatfunction = ScatFunc.from_sigma1(Ein, M, T, Eout)
        return cls(Ein, T, M, "sigma1", scatfunction.convolve(xs_0K))

    @classmethod
    def from_recoil(cls, xs_0K: pd.Series, Ein: float, M: float, T: float,
                    Eout: np.ndarray, *args, **kwargs):
        """
        Generate the Differential xs for elastic scattering from the most similar distribution of the S(alpha, -beta)
        tables and sigma1 algorithm
        ..math::
            \frac{d\sigma_T(E)}{dE^\prime} = \frac{\sigma(E^\prime + R)}{2 * k_B * T}\sqrt{\frac{E^\prime}{E}} S(\alpha(\theta, E^\prime, E, M, T), \beta( E^\prime, E, T))

        Parameters for fgm, sct and pdos models
        ----------------------------------------
        xs_0K : pd.Series, (Z,)
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
        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("dxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212

        # DOPUSH algorithm:
        >>> Dxs.from_recoil(xs_0K, Ein, M, T, Eout, model="fgm").data.iloc[::100]
        Eout
        1.80000     0.000127
        1.84004     0.020078
        1.88008     0.977778
        1.92012    14.653507
        1.96016    67.580932
        2.00020    95.914649
        2.04024    41.890458
        2.08028     5.630233
        2.12032     0.232873
        2.16036     0.002964
        dtype: float64
        """
        scatfunction = ScatFunc.from_recoil(Ein, M, T, Eout, *args, **kwargs)
        Exs = Eout + get_gressier_recoil(Ein, T, M)
        return cls(Ein, T, M, "dopush", scatfunction.convolve(xs_0K, Exs=Exs))

    @property
    def integral(self) -> float:
        """
        The integral value of the Diferential xs

        Returns
        -------
        float
            The integral value of the Diferential xs

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("dxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212

        # SIGMA1 algorithm:
        >>> round(Dxs.from_sigma1(xs_0K, Ein, M, T, Eout).integral, 2)
        9.09

        # DOPUSH algorithm:
        >>> theta = np.arange(0, 180, 1)[1::]
        >>> round(Dxs.from_recoil(xs_0K, Ein, M, T, Eout, model="fgm").integral, 2)
        9.09
        """
        return integrate(self.data)

    @property
    def prob(self) -> dict:
        """
        Get the upscattering and downscattering probabilities for the selected Ein, T, M

        Returns
        -------
        dict
            Dictionary with the upscattering and downscattering probabilities

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("dxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> dxs = Dxs.from_sigma1(xs_0K, Ein, M, T, Eout)
        >>> round(dxs.prob["upscattering"], 6)
        0.505184
        >>> round(dxs.prob["downscattering"], 6)
        0.490636
        >>> round(dxs.prob["Ein=Eout"], 6)
        0.004179
        """
        integral = self.integral
        Eout = self.data.index.values
        up = integrate(self.data.loc[Eout > self.Ein]) / integral
        down = integrate(self.data.loc[Eout < self.Ein]) / integral
        return {"upscattering": up,  "downscattering": down, "Ein=Eout": 1.0 - up - down}

    @property
    def pdf(self) -> pd.Series:
        """
        Get the probability density function of the Differential XS

        Returns
        -------
        pd.Series
            The probability density function of the Differential XS

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("dxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> dxs = Dxs.from_sigma1(xs_0K, Ein, M, T, Eout)

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

    def shift(self, dx: [float, np.ndarray, pd.DataFrame]):
        """
        Shift the Double Differential XS in the given axis and interpolate to get the values of the original axis

        Parameters
        ----------
        dx : float or np.ndarray or pd.Series or pd.DataFrame
            The shift value in the given axis. If a pd.DataFrame is given, the shift value is calculated according to
            the index or the columns of the pd.DataFrame (next argument to select).
        axis : str, optional
            The axis to shift the Double Differential XS. The default is "Eout".

        Returns
        -------
        DDxs
            The shifted Double Differential XS values in the original axis

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("dxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> dxs = Dxs.from_sigma1(xs_0K, Ein, M, T, Eout)
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
        1.80000     0.000000
        1.88008     0.542661
        1.96016    53.207633
        2.04024    56.378145
        2.12032     0.840643
        dtype: float64
        """
        # copy data to avoid changing the original data:
        dxs = self.data
        # Check the dx:
        dx_ = check_dx(self.data, dx, 0)
        if isinstance(dx, float) or isinstance(dx, int):
            dxs = reshift(dxs, dx_)
        else:
            dxs.loc[dx_.index] = reshift(dxs.loc[dx_.index], dx_)
        return Dxs(self.Ein, self.T, self.M, self.algorithm, dxs)


def check_dx(data: [pd.DataFrame, pd.Series],
             dx: [float, np.ndarray, pd.DataFrame],
             axis: [str, int]) -> [float, pd.Series, pd.DataFrame]:
    """
    Check the dx value to shift the Double Differential XS and return the value
    in the correct format for the shift aplicattion.

    Parameters
    ----------
    data : pd.DataFrame, pd.Series
        Double or Single Differential XS data to shift
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