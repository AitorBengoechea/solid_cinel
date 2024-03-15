"""
Python for working with Diferential XS.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
from scipy.constants import physical_constants as const
from solid_cinel.core.scattering_function import ScatFunc, Beta
from solid_cinel.core.scattering_function.alpha import get_gressierRecoil
from solid_cinel.core.generic import integrate, reshift, interpolation
import os
from typing import Iterable

# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]

# Avoid numba fast math:
nb.config.FASTMATH_DEFAULT = False


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
    def from_sigma1(cls, xs0K: pd.Series, Ein: float, M: float, T: float, Eout: np.ndarray):
        """
        Generate the Differential xs for elastic scattering from sigma1
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
        Dxs
            Differential cross section for elastic scattering

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("dxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212

        # SIGMA1 algorithm:
        >>> Dxs.from_sigma1(xs0K, Ein, M, T, Eout).data.iloc[::100]
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
        return cls(Ein, T, M, "sigma1", scatfunction.convolve(xs0K))

    @classmethod
    def from_alpha0(cls, xs0K: pd.Series, Ein: float, M: float, T: float,
                    Eout: np.ndarray, *args, **kwargs):
        """
        Generate the Differential xs for elastic scattering from the most similar distribution of the S(alpha, -beta)
        tables and sigma1 algorithm
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
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212

        # alpha0 algorithm:
        >>> Dxs.from_alpha0(xs0K, Ein, M, T, Eout, model="fgm").data.iloc[::100]
        Eout
        1.80000     0.000127
        1.84004     0.020090
        1.88008     0.977898
        1.92012    14.653507
        1.96016    67.580929
        2.00020    95.914649
        2.04024    41.890462
        2.08028     5.630375
        2.12032     0.232929
        2.16036     0.002964
        dtype: float64
        """
        scatfunction = ScatFunc.from_alpha0(Ein, M, T, Eout, *args, **kwargs)
        recoil = get_gressierRecoil(Ein, T, M)
        if Ein + recoil <= 0:
            raise ValueError("The incident energy is lower than the recoil energy")
        else:
            EoutShift = Eout + recoil
        return cls(Ein, T, M, "alpha0", scatfunction.convolve(xs0K, Exs=EoutShift))

    @staticmethod
    def get_alpha0(xs0K: pd.Series, Ein: np.ndarray, M: float, T: float, *args,
                   **kwargs) -> pd.DataFrame:
        """
        Get the Dxs function for the 0K cross section

        Parameters
        ----------
        xs0K : pd.Series, (Z,)
            0K xs data for the given material in barns
        Ein: np.ndarray
            The incident energy grid in eV
        M: float
            The mass of the target material in amu
        T: float
            Temperature of the material in K

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
        pd.DataFrame
            The alpha0 scattering function

        Examples
        --------
        >>> from solid_cinel.core.material.vibration import Pdos
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238

        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("dxs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        >>> Ein = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> index = pd.Index(Ein, name="Ein")
        >>> T = 300
        >>> M = 238.05077040419212
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> Dxs.get_alpha0(xs0K, Ein, M, T, model="fgm").iloc[::, 1000::1000].round(6)
        beta    -2.662634  -1.114591   0.433452   1.981495   9.902464
        Ein
        6.7554  28.734670  20.108117   7.358406   1.078944        0.0
        6.9050   4.650556   6.747923   3.490777   0.636176        0.0
        7.0439   3.129387   4.835731   2.650856   0.514044        0.0
        7.2000   2.566863   3.985399   2.229881   0.449221        0.0
        7.3157   2.361360   3.634402   2.044991   0.420547        0.0
        7.4480   2.216768   3.362471   1.896636   0.397744        0.0
        >>> Dxs.get_alpha0(xs0K, Ein, M, T, pdos, model="sct").iloc[::, 1000::1000].round(6)
        beta    -2.668826  -1.120783   0.427260   1.975303   9.739857
        Ein
        6.7554  28.821334  19.787290   7.292477   1.076292        0.0
        6.9050   4.638501   6.628216   3.456548   0.633850        0.0
        7.0439   3.117428   4.748840   2.624669   0.511817        0.0
        7.2000   2.554504   3.913638   2.208039   0.446993        0.0
        7.3157   2.348490   3.569032   2.025162   0.418284        0.0
        7.4480   2.203177   3.302152   1.878482   0.395426        0.0
        >>> Dxs.get_alpha0(xs0K, Ein, M, T, pdos, model="pdos").iloc[::, 1000::1000].round(6)
        beta    -2.998559  -1.450516   0.097527   1.645570   4.033176
        Ein
        6.7554  28.922999  22.199127  10.213472   1.734711   0.018831
        6.9050   3.532269   6.685214   4.562669   0.982888   0.013663
        7.0439   2.338940   4.737526   3.419771   0.784096   0.012083
        7.2000   1.920413   3.901155   2.858675   0.679510   0.011346
        7.3157   1.773443   3.563735   2.614582   0.633089   0.011108
        7.4480   1.674328   3.306635   2.419227   0.596052   0.011008
        """
        scatfunc = ScatFunc.get_alpha0(Ein, M, T, *args, **kwargs)
        EinGrid = Ein + (scatfunc.columns.values * kb * T)[:, np.newaxis]
        EinGrid += get_gressierRecoil(Ein, T, M)
        return scatfunc * interpolation(xs0K, EinGrid.T, parallel=True)

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
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212

        # SIGMA1 algorithm:
        >>> round(Dxs.from_sigma1(xs0K, Ein, M, T, Eout).integral, 2)
        9.09

        # DOPUSH algorithm:
        >>> theta = np.arange(0, 180, 1)[1::]
        >>> round(Dxs.from_alpha0(xs0K, Ein, M, T, Eout, model="fgm").integral, 2)
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
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> dxs = Dxs.from_sigma1(xs0K, Ein, M, T, Eout)
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
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> dxs = Dxs.from_sigma1(xs0K, Ein, M, T, Eout)

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
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> dxs = Dxs.from_sigma1(xs0K, Ein, M, T, Eout)
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