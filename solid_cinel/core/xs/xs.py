"""
Python for working with Angle integrated scattering xs at different temperature.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from scipy.constants import physical_constants as const
from typing import Iterable, Union
from solid_cinel.core.xs.dxs import Dxs
from solid_cinel.core.material.vibration.pdos import Pdos
from solid_cinel.core.generic import integrate, interpolation
from solid_cinel.core.scattering_function.alpha import Alpha, get_alphaRecoil
import warnings
import os
import dask
import dask.bag as db

# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]

# Avoid numba fast math:
nb.config.FASTMATH_DEFAULT = False

# Get the number of available processors
num_processors = os.cpu_count()

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


class Xs:
    """
    Class for the differential cross section for elastic scattering
    """
    def __init__(self, M: float, temperatures: Union[float, Iterable[float]],
                 *args, xs0Kcomplete: pd.Series = None, **kwargs):
        """
        Class for working with Angle integrated scattering xs at different
        temperature.

        Parameters
        ----------
        T : float
            Temperature in K
        *args : tuple
            The scattering xs data
        **kwargs : dict
            The scattering function data

        Returns
        -------
        None.

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        >>> M = 238.05077040419212
        >>> T = 0
        >>> Xs(M, T, xs0K).data.iloc[0:10000:1000]
        T                   0
        Ein
        0.00001      9.420892
        11.23650     9.239644
        34.70286     1.146785
        58.18538     9.794358
        80.66597     1.639337
        97.56808     4.895060
        116.82090  457.760100
        145.67660   39.688900
        165.85470   11.753670
        200.79510   15.734840
        """
        self.M = M
        temperatures = self.check_T(temperatures)
        data_frame = pd.DataFrame(*args, **kwargs)
        data_frame.columns = pd.Index(temperatures, name="T")
        self.data = data_frame.loc[:, ~data_frame.columns.duplicated()]
        self.get_xs0Kcomp(xs0Kcomplete)

    @property
    def data(self) -> pd.DataFrame:
        """
        Diferential xs data.

        Returns
        -------
        pd.DataFrame
            Diferential xs data
        """
        return self._data

    @data.setter
    def data(self, xs: Iterable):
        """
        Set the scattering function data and check the normalization.

        Parameters
        ----------
        pdf : pd.Series
            The scattering function data

        """
        xs_ = pd.DataFrame(xs).sort_index(axis=0).sort_index(axis=1)
        xs_.index.name = "Ein"
        xs_.columns.name = "T"
        self._data = xs_

    def update_data(self, xsT: pd.DataFrame, inplace: bool):
        """
        Update the data with the new results

        Parameters
        ----------
        xsT: pd.DataFrame
            Cross section results for the new temperatures
        inplace: bool
            If True, the data is stored in the class attribute, otherwise it
            is returned as a new object

        Returns
        -------
        None, Xs
            New object with the updated data or None if inplace is True, so
            the data is stored in the class attribute and modified in place.
        """
        dataNew = pd.concat([self.data, xsT], axis=1).sort_index(axis=1)
        if inplace:
            self.data = dataNew
        else:
            return Xs(self.M, dataNew.columns, dataNew)

    @staticmethod
    def check_T(temperatures: Union[float, Iterable[float]]):
        """
        Check the temperature input

        Parameters
        ----------
        temperatures: Union[float, Iterable[float]]
            The temperature in K

        Returns
        -------
        Union[float, Iterable[float]]
            The temperature in K
        """
        if isinstance(temperatures, (int, float)):
            temperatures = [temperatures]
        elif not all(isinstance(t, (int, float)) for t in temperatures):
            raise TypeError("All temperatures must be int or float")
        return temperatures
    def check_algorithm(self, algorithm: str, model: str = None) -> callable:
        """
        Check the algorithm input

        Parameters
        ----------
        algorithm: str
            The algorithm to use

        Returns
        -------
        callable
            The algorithm to use for the calculation
        """
        if algorithm == "sigma1":
            func = self._calc_sigma1Ein
        elif algorithm == "alpha0":
            if model == "pdos":
                func = self._calc_alpha0TClm
            else:
                func = self._calc_alpha0Sct
        else:
            raise ValueError("Invalid algorithm")
        return func

    def get_Tcalc(self, temperatures: Union[float, Iterable[float]]) -> pd.Index:
        """
        Get the new temperatures to calculate.

        Parameters
        ----------
        T: Union[float, Iterable[float]]
            The temperature in K

        Returns
        -------
        pd.Index
            The new temperatures to calculate

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        >>> M = 238.05077040419212
        >>> T = 300
        >>> Xs(M, 0, xs0K).get_Tcalc(T)
        Index([300], dtype='int64')
        """
        Tnew = pd.Index(self.check_T(temperatures))
        return Tnew.difference(self.data.columns)

    def get_Tinterp(self, temperatures: Union[float, Iterable[float]]) -> pd.Index:
        """
        Get from the new temperatures, the temperatures already in the object

        Parameters
        ----------
        T: Union[float, Iterable[float]]
            The temperature in K

        Returns
        -------
        pd.Index
            The new temperatures to calculate

        Returns
        -------
        pd.Index
            The temperatures in the object

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        >>> M = 238.05077040419212
        >>> T = [0, 300]
        >>> Xs(M, 0, xs0K).get_Tinterp(T)
        Index([0], dtype='int64')
        """
        Tnew = pd.Index(self.check_T(temperatures))
        return self.data.columns.intersection(Tnew)


    def get_xs0Kcomp(self, xs0Kcomplete: [pd.Series, None]) -> pd.Series:
        """
        Get the 0K scattering function with all the data

        Parameters
        ----------
        xs0Kcomplete: pd.Series
            The 0K scattering function with all the data

        Returns
        -------
        pd.Series
            The 0K scattering function with all the data
        """
        if xs0Kcomplete is None:
            if 0 in self.data.columns:
                self.xs0Kcomplete = self.data.loc[::, 0]
            else:
                raise ValueError("0K data not found")
        else:
            self.xs0Kcomplete = xs0Kcomplete

    def get_EinTcomb(self, Tnew: Iterable, EinGrid: Iterable = None) -> list:
        """
        Get the incident energy and temperature combinations

        Parameters
        ----------
        Tnew: Iterable
            The new temperatures to calculate
        EinGrid: Iterable, None
            The incident energy grid in eV. If not provided, it will be taken
            from the class attribute.

        Returns
        -------
        list
            The incident energy and temperature combinations

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        >>> M = 238.05077040419212
        >>> xs = Xs(M, 0, xs0K.iloc[0:10000:5000], xs0Kcomplete=xs0K)
        >>> Tnew = [300, 100]
        >>> xs.get_EinTcomb(Tnew)
        [(300, 1e-05), (300, 97.56808), (100, 1e-05), (100, 97.56808)]

        >>> xs.get_EinTcomb(Tnew, np.array([1, 2, 3]))
        [(300, 1), (300, 2), (300, 3), (100, 1), (100, 2), (100, 3)]

        >>> xs.get_EinTcomb(Tnew, np.array([[1, 2], [3, 4]]))
        [(300, 1), (300, 2), (100, 3), (100, 4)]
        """
        if hasattr(EinGrid, "__len__"):
            if len(EinGrid.shape) == 2:
                N, M = EinGrid.shape[0], EinGrid.shape[1]
                return [(Tnew[i], EinGrid[i, j]) for i in range(N) for j in range(M)]
            else:
                return [(T, Ein) for T in Tnew for Ein in EinGrid]
        else:
            return [(T, Ein) for T in Tnew for Ein in self.data.index]

    @staticmethod
    def _calc_sigma1Ein(T: float, Ein: float, xs0K: pd.Series, M: float) -> float:
        """
        Calculate the elastic scattering cross section at temperature T and
        incident energy Ein

        Parameters
        ----------
        T: float
            The temperature in K
        Ein: float
            The incident energy in eV
        xs0K: pd.Series
            The 0K scattering function
        M: float
            The mass of the nucleus in amu


        Returns
        -------
        pd.Series
            The elastic scattering cross section in barns

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        >>> M = 238.05077040419212
        >>> T = 300
        >>> Ein = 2.0
        >>> round(Xs._calc_sigma1Ein(T, Ein, xs0K, M), 6)
        9.086237
        """
        Eout = default_Eout(Ein)
        return Dxs.from_sigma1(xs0K, Ein, M, T, Eout).integral

    @staticmethod
    def _calc_alpha0Sct(T: float, Ein: float, xs0K: pd.Series, M: float,
                        *args, **kwargs) -> float:
        """
        Calculate the elastic scattering cross section at temperature T and
        incident energy Ein using alpha0 model

        Parameters
        ----------
        T: float
            The temperature in K
        Ein: float
            The incident energy in eV
        xs0K: pd.Series
            The 0K scattering function
        M: float
            The mass of the nucleus in amu
        model : str
            The model used to calculate the S(alpha, beta) distribution. The available models are:
                - "fgm": Free Gas Model (default)
                - "sct": Short Collision Time

        Parameters for sct
        ------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.

        Returns
        -------
        float
            The elastic scattering cross section in barns for the given
            temperature and incident energy.

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate Broadening test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> M = 238.05077040419212
        >>> Xs._calc_alpha0Sct(T, Ein, xs0K, M, model="fgm")
        9.085407458237226

        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> Xs._calc_alpha0Sct(T, Ein, xs0K, M, pdos, model="sct")
        9.081349151760891
        """
        Eout = default_Eout(Ein)
        return Dxs.from_recoil(xs0K, Ein, M, T, Eout, *args, **kwargs).integral

    @staticmethod
    def _calc_alpha0TClm(T: float, EinGrid: np.ndarray, xs0K: pd.Series, M: float,
                         *args, **kwargs):
        """
        Calculate the elastic scattering cross section at temperature T and
        incident energy Ein using alpha0 model

        Parameters
        ----------
        xs0K: pd.Series
            The 0K scattering function
        EinGrid: np.ndarray
            The incident energy grid in eV
        M: float
            The mass of the nucleus in amu
        T: float
            The temperature in K
        pdos: Pdos
            Pdos object

        Returns
        -------
        pd.Series
            The elastic scattering cross section in barns for the given
            temperature and incident energy using alpha0 model

        Examples
        --------
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        >>> M = 238.05077040419212
        >>> EinGrid = np.array([2.0, 6.67])
        >>> xsSmall = interpolation(xs0K, EinGrid)
        >>> xs = Xs(M, 0, xsSmall, xs0Kcomplete=xs0K)
        >>> T = 300
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> Xs._calc_alpha0TClm(T, EinGrid, xs0K, M, model="fgm").round(6)
        array([  9.085279, 457.021623])
        >>> Xs._calc_alpha0TClm(T, EinGrid, xs0K, M, pdos, model="sct").round(6)
        array([  9.02379 , 449.805325])
        >>> Xs._calc_alpha0TClm(T, EinGrid, xs0K, M, pdos, model="pdos").round(6)
        array([  9.084969, 461.718705])
        """
        model = kwargs.get("model", "fgm")
        dxs = Dxs.get_alpha0(xs0K, EinGrid, M, T, *args, **kwargs)
        dxsIntegral = dxs.apply(integrate, axis=1).values
        if model == "pdos":
            alpha = Alpha(dxs.index.get_level_values("alpha").to_numpy())
            return dxsIntegral / alpha.get_expansPorcen(args[0], T)
        else:
            return dxsIntegral
    def _compute(self, Tnew: Iterable, *args, EinGrid: Iterable = None,
                      algorithm: str = "sigma1", **kwargs) -> np.ndarray:
        """
        Calculate the elastic scattering cross section at new temperatures using
        dask and the selected algorithm.

        Parameters
        ----------
        Tnew: Iterable
            The new temperatures to calculate
        EinGrid: Iterable, None
            The incident energy grid in eV. If not provided, it will be taken
            from the class attribute.
        algorithm: str, optional
            The algorith to use for the calculation. The options are:
            - "sigma1": Calculate the elastic scattering cross section with the
                        sigma1 method from Njoy
            - "alpha0": Calculate the elastic scattering cross section with the
                        alpha0

        Returns
        -------
        pd.DataFrame
            The elastic scattering cross section in barns for the new
            temperatures. Each value is the incident introduced by the user or
            the default.

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        >>> M = 238.05077040419212
        >>> EinGrid = np.array([2.0, 6.67])
        >>> xsSmall = interpolation(xs0K, EinGrid)
        >>> xs = Xs(M, 0, xsSmall, xs0Kcomplete=xs0K)
        >>> Tnew = [300, 100]
        >>> pd.DataFrame(xs._compute(Tnew), index=Tnew, columns=EinGrid)
                 2.00        6.67
        300  9.086237  455.670534
        100  9.086957  664.556512

        # Check the alpha0 algorithm with the fgm model + EinGrid < Recoil
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)

        # situation (NaN in the results)
        >>> pd.DataFrame(xs._compute(Tnew, algorithm="alpha0", model="fgm"), index=Tnew, columns=EinGrid)
                 2.00        6.67
        300  9.085935  457.051619
        100  9.086803  665.494663

        >>> pd.DataFrame(xs._compute(Tnew, pdos, algorithm="alpha0", model="pdos"), index=Tnew, columns=EinGrid)
                 2.00        6.67
        300  9.084969  461.718705
        100  9.085596  649.526642
        """
        model = kwargs.get("model", "fgm")
        func = self.check_algorithm(algorithm, model)
        args = (self.xs0Kcomplete, self.M) + args
        NTnew = len(Tnew)
        if model == "pdos":
            Ein = EinGrid if hasattr(EinGrid, "__len__") else self.data.index.values
            if len(Ein.shape) == 1:
                results = [func(T, Ein, *args, model="pdos") for T in Tnew]
            else:
                results = [func(Tnew[i], Ein[i], *args, model="pdos") for i in range(NTnew)]
        else:
            bag = db.from_sequence(self.get_EinTcomb(Tnew, EinGrid))\
                    .map(lambda x: func(*x, *args, **kwargs))
            with dask.config.set(num_workers=os.cpu_count()):
                results = bag.compute()
        return np.array(results).reshape(NTnew, -1)

    def calc_T(self, T:float, *args, algorithm: str = "sigma1",
               inplace: bool = False, **kwargs):
        """
        Calculate the elastic scattering cross section at temperature T using
        the selected algorithm.

        Parameters
        ----------
        T : float
            Temperature in K
        algorithm : str, optional
            The algorith to use for the calculation. The options are:
            - "sigma1": Calculate the elastic scattering cross section with the
                        sigma1 method from Njoy
        inplace : bool, optional
            If True, the data is stored in the class attribute, otherwise it
            is returned

        Returns
        -------
        None, Xs
            New object with the updated data or None if inplace is True, so
            the data is stored in the class attribute and modified in place.

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        >>> M = 238.05077040419212
        >>> T = [300, 100]
        >>> xs = Xs(M, 0, xs0K.iloc[100:5000:500], xs0Kcomplete=xs0K)
        >>> xs.calc_T(T).data
        T                  0            100          300
        Ein
        0.065625      9.411657     9.414734     9.419595
        6.717251    172.623200   282.096835   323.919192
        11.367190     9.198383     9.198416     9.199371
        20.912000  1893.389000  3257.315536  2639.268058
        35.640580     0.974924     1.042582     1.184656
        44.877660    14.089820    14.090012    14.090359
        63.498800     5.773424     5.770605     5.764487
        66.436310    85.621850    90.534332   114.828045
        80.731840    39.201520    40.746929    29.811032
        89.051940     9.208071     9.213771     9.226450

        >>> xs.calc_T(T, algorithm="alpha0", model="fgm").data
        T                  0            100          300
        Ein
        0.065625      9.411657     9.412449     9.412080
        6.717251    172.623200   282.663018   324.891765
        11.367190     9.198383     9.200360     9.200623
        20.912000  1893.389000  3261.798758  2646.577942
        35.640580     0.974924     1.041481     1.180870
        44.877660    14.089820    14.090070    14.092491
        63.498800     5.773424     5.770789     5.765396
        66.436310    85.621850    90.540165   114.481715
        80.731840    39.201520    40.785557    29.838786
        89.051940     9.208071     9.213726     9.226565


        >>> EinGrid = np.array([0.065625, 2.0, 4.0, 5.0, 6.67, 7.0])
        >>> xsSmall = interpolation(xs0K, EinGrid)
        >>> xs = Xs(M, 0, xsSmall, xs0Kcomplete=xs0K)
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> xs.calc_T(T, pdos, algorithm="alpha0", model="pdos").data
        T                 0           100         300
        Ein
        0.065625     9.411657    9.403287    9.408170
        2.000000     9.085342    9.085596    9.084969
        4.000000     8.481975    8.482253    8.481713
        5.000000     7.805580    7.805930    7.804704
        6.670000  1269.792131  649.526642  461.718705
        7.000000    19.825115   19.941105   20.060625

        """
        Tnew = self.get_Tcalc(T)
        if Tnew.empty:
            warnings.warn("All the temperatures are already calculated")
            return self
        kwargs["algorithm"] = algorithm
        xsT = pd.DataFrame(self._compute(Tnew, *args, **kwargs).T,
                           index=self.data.index, columns=Tnew)
        return self.update_data(xsT, inplace)

    @classmethod
    def from_sigma1(cls, T: float, M: float, xs0Kshort: pd.Series,
                    xs0Kcomplete: pd.Series = None, inplace: bool = False):
        """
        Calculate the elastic scattering cross section for a nucleus with mass
        M at temperature T > 0.

        Parameters
        ----------
        T: float
            The temperature in K
        M: float
            The mass of the nucleus in amu
        xs0Kshort: pd.Series
            The 0K scattering function with the incident energy grid to use. It
            is recommended to use a short grid to avoid the doppler broadening
            of all the Ein.
        xs0Kcomplete: pd.Series, optional
            The 0K scattering function with all the data. If not provided, it
            will be taken from the class attribute.

        Returns
        -------
        Xs
            The elastic scattering cross section in barns

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        >>> M = 238.05077040419212
        >>> T = 300
        >>> xs0Kshort = xs0K.iloc[0:10000:1000]
        >>> Xs.from_sigma1(T, M, xs0Kshort, xs0Kcomplete=xs0K).data
        T                 0           300
        Ein
        0.00001      9.420892   36.117710
        11.23650     9.239644    9.240652
        34.70286     1.146785    1.147109
        58.18538     9.794358    9.793941
        80.66597     1.639337   23.581216
        97.56808     4.895060    4.891338
        116.82090  457.760100  931.532221
        145.67660   39.688900   13.281879
        165.85470   11.753670   12.634276
        200.79510   15.734840   15.733460
        """
        # Initialize the class
        xs = cls(M, 0, xs0Kshort, xs0Kcomplete=xs0Kcomplete)
        return xs.calc_T(T, algorithm="sigma1", inplace=inplace)

    @classmethod
    def from_alpha0(cls, T: float, M: float, xs0Kshort: pd.Series, *args,
                    xs0Kcomplete: pd.Series = None, inplace: bool = False,
                    **kwargs):
        """
        Calculate the elastic scattering cross section for a nucleus with mass
        M at temperature T > 0 using the alpha0 model.

        Parameters
        ----------
        T: float
            The temperature in K
        M: float
            The mass of the nucleus in amu
        xs0Kshort: pd.Series
            The 0K scattering function with the incident energy grid to use. It
            is recommended to use a short grid to avoid the doppler broadening
            of all the Ein.
        xs0Kcomplete: pd.Series, optional
            The 0K scattering function with all the data. If not provided, it
            will be taken from the class attribute.
        inplace: bool, optional
            If True, the data is stored in the class attribute, otherwise it
            is returned
        kwargs: dict
            The parameters for the alpha0 model

        Returns
        -------
        Xs
            The elastic scattering cross section in barns

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        >>> M = 238.05077040419212
        >>> T = [300, 100]
        >>> EinGrid = np.array([0.065625, 2.0, 4.0, 5.0, 6.67, 7.0])
        >>> xsSmall = interpolation(xs0K, EinGrid)
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> Xs.from_alpha0(T, M, xsSmall, model="fgm", xs0Kcomplete=xs0K).data
        T                 0           100         300
        Ein
        0.065625     9.411657    9.412449    9.412080
        2.000000     9.085342    9.086803    9.085935
        4.000000     8.481975    8.484040    8.482812
        5.000000     7.805580    7.807330    7.805865
        6.670000  1269.792131  665.494663  457.051619
        7.000000    19.825115   19.902142   20.048596

        >>> Xs.from_alpha0(T, M, xsSmall, pdos, model="pdos", xs0Kcomplete=xs0K).data
        T                 0           100         300
        Ein
        0.065625     9.411657    9.403287    9.408170
        2.000000     9.085342    9.085596    9.084969
        4.000000     8.481975    8.482253    8.481713
        5.000000     7.805580    7.805930    7.804704
        6.670000  1269.792131  649.526642  461.718705
        7.000000    19.825115   19.941105   20.060625
        """
        # Initialize the class
        xs = cls(M, 0, xs0Kshort, xs0Kcomplete=xs0Kcomplete)
        # Get cls attributes using the available information
        return xs.calc_T(T, *args, algorithm="alpha0", inplace=inplace, **kwargs)

    def interp_Ein(self, Ein: [float, np.ndarray]) -> pd.DataFrame:
        """
        Interpolate Xs objet to a new Ein

        Parameters
        ----------
        Ein: float, np.ndarray
            New Ein grid

        Returns
        -------
        pd.DataFrame
            New Ein grid for all the temperatures in the object

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        >>> M = 238.05077040419212
        >>> T = [300, 100]
        >>> EinGrid = np.array([0.065625, 2.0, 4.0, 5.0, 6.67, 7.0])
        >>> xsSmall = interpolation(xs0K, EinGrid)
        >>> xs = Xs.from_sigma1(T, M, xsSmall, xs0Kcomplete=xs0K)
        >>> xs.interp_Ein([3.0])
        T         0         100       300
        3.0  8.783659  8.784881  8.784565
        >>> xs.interp_Ein([3.0, 4.5])
        T         0         100       300
        3.0  8.783659  8.784881  8.784565
        4.5  8.143778  8.144254  8.144288
        """
        return self.data.apply(lambda x: interpolation(x, Ein))

    @staticmethod
    def get_4PCFEin(Ein: float, Eout: np.ndarray, mu: np.ndarray,
                    M: float) -> np.ndarray:
        """
        Get the incident energy matrix for the arno model.

        Parameters
        ----------
        Ein: float
            The incident energy of the neutron in eV
        Eout: np.ndarray, (Z,)
            The neutron outgoing energy grid in eV
        mu: np.ndarray, (M,)
            The neutron outgoing angle grid in degrees (0, 180]
        M: float
            Mass of the material in amu

        Returns
        -------
        EinArno: np.ndarray, (M, Z)
            Incident energy matrix for the arno model

        Examples
        --------
        >>> Ein = 2.0
        >>> Eout = np.linspace(2.0 * 0.9, 2.0 * 1.1, 5)
        >>> mu = np.array([-1.0, -0.5, 0.0, 0.5, 0.9])
        >>> M = 238.05077040419212
        >>> Xs.get_4PCFEin(Ein, Eout, mu, M)
        Eout       1.8       1.9       2.0       2.1       2.2
        mu
        -1.0  1.916519  1.966736  2.016949  2.067159  2.117367
        -0.5  1.912284  1.962499  2.012712  2.062923  2.113132
         0.0  1.908051  1.958263  2.008474  2.058686  2.108898
         0.5  1.903825  1.954028  2.004237  2.054452  2.104671
         0.9  1.900524  1.950660  2.000847  2.051083  2.101362
        """
        @nb.jit(nopython=True, nogil=False, cache=True)
        def calc_4PCFEin(Ein: float, Eout: np.ndarray, mu: np.ndarray,
                         M: float) -> np.ndarray:
            EinArno = np.empty((len(mu), len(Eout)))
            for i in prange(len(mu)):
                EinArno[i, :] = EinArnoRow(Ein, Eout, mu[i], M)
            return EinArno
        return pd.DataFrame(calc_4PCFEin(Ein, Eout, mu, M),
                            index=pd.Index(mu, name="mu"),
                            columns=pd.Index(Eout, name="Eout"))

    def get_4PCFxs(self, Ein, T, Eout, theta, *args, **kwargs) -> pd.DataFrame:
        """
        Get the angle-integrated xs matrix for 4PCF model

        Parameters
        ----------
        Ein: float
            The incident energy of the neutron in eV
        T: float
            Temperature of the material in K
        Eout: np.array, (N,)
            The neutron outgoing energy grid in eV
        theta: np.array, (M,)
            The neutron outgoing angle grid in degrees (0, 180]

        Returns
        -------
        pd.DataFrame
            Angle-integrated xs matrix for 4PCF model

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        >>> M = 238.05077040419212
        >>> xs = Xs(M, 0, xs0K)
        >>> T = 300
        >>> Ein = 2.0
        >>> Eout = np.linspace(2.0 * 0.9, 2.0 * 1.1, 5)
        >>> theta = np.array([180, 120, 90, 60, 30])
        >>> xs.get_4PCFxs(Ein, T, Eout, theta, algorithm="sigma1")
        Eout                1.8       1.9       2.0       2.1       2.2
        mu
        -1.000000e+00  9.102355  9.092121  9.081758  9.071139  9.060521
        -5.000000e-01  9.104914  9.094623  9.084231  9.073561  9.062890
         6.123234e-17  9.105581  9.095328  9.084990  9.074371  9.063732
         5.000000e-01  9.106114  9.095876  9.085560  9.074971  9.064341
         8.660254e-01  9.106574  9.096350  9.086046  9.075473  9.064836

        >>> xs.get_4PCFxs(Ein, T, Eout, theta, algorithm="alpha0", model="fgm")
        Eout                1.8       1.9       2.0       2.1       2.2
        mu
        -1.000000e+00  9.102355  9.092121  9.081758  9.071139  9.060521
        -5.000000e-01  9.104981  9.094763  9.084445  9.073848  9.063247
         6.123234e-17  9.105232  9.094997  9.084678  9.074079  9.063461
         5.000000e-01  9.105792  9.095559  9.085250  9.074668  9.064044
         8.660254e-01  9.106263  9.096043  9.085744  9.075176  9.064543

        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> xs.get_4PCFxs(Ein, T, Eout, theta, pdos, algorithm="alpha0", model="pdos")
        Eout                1.8       1.9       2.0       2.1       2.2
        mu
        -1.000000e+00  9.102355  9.092121  9.081758  9.071139  9.060521
        -5.000000e-01  9.104000  9.093744  9.083411  9.072788  9.062147
         6.123234e-17  9.103939  9.093695  9.083370  9.072767  9.062137
         5.000000e-01  9.104621  9.094390  9.084082  9.073504  9.062880
         8.660254e-01  9.105234  9.095019  9.084725  9.074162  9.063533
        """
        mu = np.sort(np.cos(np.deg2rad(theta)))
        # Get the incident energy grid:
        Ein_4PCF = self.get_4PCFEin(Ein, Eout, mu, self.M)
        Eout_ = pd.Index(Eout, name="Eout")

        # Get the temperatures:
        Tarno = T * (1 + mu) / 2
        Ein_4PCF = Ein_4PCF.set_axis(Tarno, axis=0)
        Tcalc, Tinterp = self.get_Tcalc(Tarno), self.get_Tinterp(Tarno)

        # Interpolation:
        if Tinterp.empty:
            xsInterp = None
        else:
            xsInterpComplete = self.interp_Ein(np.unique(Ein_4PCF.loc[Tinterp]))
            xsInterpValues = {T: xsInterpComplete.loc[Ein_4PCF.loc[T], T]
                              for T in Tinterp}
            xsInterp = pd.DataFrame(xsInterpValues).T.set_axis(Eout_, axis=1)
        # Calculation
        if Tcalc.empty:
            xsCalc = None
        else:
            xsCalcValues = self._compute(Tcalc, *args,
                                         EinGrid=Ein_4PCF.loc[Tcalc].values,
                                         **kwargs)
            xsCalc = pd.DataFrame(xsCalcValues, index=Tcalc, columns=Eout_)
        return pd.concat([xsInterp, xsCalc]).set_axis(pd.Index(mu, name="mu"),
                                                      axis=0)


@nb.jit(nopython=True, nogil=False, cache=True)
def default_Eout(Ein: float) -> np.ndarray:
    """
    Generate the default Eout grid for the convolution. The grid is tested with
    NJOY values to ensure a relative difference smaller than 0.4%

    Parameters
    ----------
    Ein : float
        Incident energy in eV

    Returns
    -------
    Eout : ndarray
        Outgoing energy grid in eV

    Examples
    --------
    Test the default Eout with NJOY values:
    # 0K xs data for U238:
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("xs.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
    >>> os.chdir(wd)

    # Generate Broadening test results:
    >>> T = 1000
    >>> Ein = 2.0
    >>> Eout = default_Eout(Ein)
    >>> M = 238.05077040419212
    >>> round(Dxs.from_sigma1(xs0K, Ein, M, T, Eout).integral, 2)
    9.09
    """
    EoutSmall = np.linspace(0,
                             0.99 * Ein,
                             2000)
    EoutMid = np.linspace(0.99 * Ein,
                              Ein * 1.01,
                              3000)
    if Ein * 2 < 5.0:
        EoutGreat = np.logspace(np.log10(Ein * 1.01),
                                 np.log10(5.0),
                                 2000)
    else:
        EoutGreat = np.logspace(np.log10(Ein * 1.01),
                                 np.log10(2 * Ein),
                                 2000)
    return np.sort(np.concatenate((EoutGreat, EoutSmall, EoutMid)))


@nb.jit(nopython=True, nogil=False, cache=True)
def EinArnoRow(Ein: float, Eout: np.ndarray, mu: float,
               M: float) -> np.ndarray:
    """
    Get the incident energy row for the arno model.

    Parameters
    ----------
    Ein: float
        The incident energy of the neutron in eV
    Eout: np.ndarray, (Z,)
        The neutron outgoing energy grid in eV
    mu: float
        The cosine of the neutron outgoing angle in degrees radians (0, 180]
    M: float
        Mass of the material in amu

    Returns
    -------
    EinArnoRow: np.ndarray, (Z,)
        Incident energy row for the arno model
    """
    muEinArno = (Eout + Ein) / 2 - Ein * mu * m / M
    muEinArno += 0.5 * get_alphaRecoil(Eout, Ein, M, mu) / (1 - mu)
    return muEinArno
