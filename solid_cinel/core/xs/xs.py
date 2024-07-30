"""
Python for working with Angle integrated scattering xs at different temperature.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
from numba import vectorize
from scipy.constants import physical_constants as const
from typing import Iterable, Union
from solid_cinel.core.scattering_function.scatfunc import ScatFunc
from solid_cinel.core.xs.dxs import Dxs
from solid_cinel.core.material.pdos import Pdos
from solid_cinel.core.generic import interpolation
from solid_cinel.core.scattering_function.alpha import get_alphaRecoil
import warnings
import os
import dask.bag as db

# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]

# Avoid numba fast math:
nb.config.FASTMATH_DEFAULT = False


class Xs:
    """
    Class for the differential cross section for elastic scattering
    """
    def __init__(self, M: float, temperatures: Union[float, Iterable[float]],
                 xsSmall: pd.Series, xs0Kcomplete: pd.Series = None):
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
        >>> xs0K = Xs.read_xs("u238.0.2")
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
        # Set the mass of the nucleus in amu
        self.M = M

        # Set the temperature
        temperatures = self.check_InputValues(temperatures)

        # Force xsSmall to be a DataFrame with temperatures as columns
        df = pd.DataFrame(xsSmall)
        df.columns = pd.Index(temperatures, name="T")
        self.data = df

        # Set the 0K scattering function with all the data
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
        # Set the data style
        xs_ = pd.DataFrame(xs).sort_index(axis=0).sort_index(axis=1)
        xs_.index.name = "Ein"
        xs_.columns.name = "T"

        # save the data
        self._data = xs_.loc[:, ~xs_.columns.duplicated()]

    def update_data(self, xsT: pd.DataFrame, inplace: bool, axis: int = 1):
        """
        Update the data with the new results

        Parameters
        ----------
        xsT: pd.DataFrame
            Cross section results for the new temperatures
        inplace: bool
            If True, the data is stored in the class attribute, otherwise it
            is returned as a new object
        axis: int
            The axis to concatenate the data

        Returns
        -------
        None, Xs
            New object with the updated data or None if inplace is True, so
            the data is stored in the class attribute and modified in place.
        """
        if isinstance(xsT, pd.Series):
            dataNew = self.data.copy()
            if xsT.name == "Ein":
                dataNew = dataNew.append(xsT)
            else:
                dataNew[xsT.name] = xsT
        else:
            dataNew = pd.concat([self.data, xsT], axis=axis)

        # Update the data in the object or return a new object
        if inplace:
            self.data = dataNew.sort_index(axis=axis)
            return self
        else:
            return Xs(self.M, dataNew.columns, dataNew, self.xs0Kcomplete)

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
    def from_xs0K(cls, filename: str, M: float, EinSmall: [float, np.array] = None,
                  **kwargs):
        """
        Create a new object from the 0K xs data

        Parameters
        ----------
        filename : str
            The filename of the 0K xs data
        M : float
            The mass of the nucleus in amu
        EinSmall : float, np.array, optional
            The incident energy grid in eV. If not provided, it will be taken
            from the class attribute.
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
        Xs
            The Xs object based on the 0K xs data
        """
        # Read the 0K xs data
        xs0K = cls.read_xs(filename, **kwargs)

        # Interpolate the data if needed for handling smaller incident energies grid
        if EinSmall is not None:
            EinSmall_ = np.array(cls.check_InputValues(EinSmall))
            xs0Ksmall = interpolation(xs0K, EinSmall_)
        else:
            xs0Ksmall = xs0K.copy()

        return cls(M, 0, xs0Ksmall, xs0K)

    def get_xs0Kcomp(self, xs0Kcomplete: [pd.Series, None]):
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

    @staticmethod
    def check_InputValues(input: [float, Iterable[float]]) -> Iterable:
        """
        Check the incident energy input

        Parameters
        ----------
        Ein: Union[float, Iterable[float]]
            The incident energy in eV

        Returns
        -------
        Union[float, Iterable[float]]
            The incident energy in eV
        """
        if hasattr(input, "__len__"):
            return input
        elif isinstance(input, (int, float)):
            return [input]
        elif not all(isinstance(e, (int, float)) for e in input):
            raise TypeError("All input values must be int or float")

    def get_EinCalc(self, Ein: [float, Iterable[float]]) -> pd.Index:
        """
        Check the incident energy input

        Parameters
        ----------
        Ein: Union[float, Iterable[float]]
            The incident energy in eV

        Returns
        -------
        Union[float, Iterable[float]]
            The incident energy in eV

        Examples
        --------
        # Define the parameters for the test:
        >>> M = 238.05077040419212
        >>> Ein = [1.0, 3.0]

        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs = Xs.from_xs0K("u238.0.2", M, 1.0)
        >>> os.chdir(wd)

        # Check the incident energy
        >>> assert xs.get_EinCalc(Ein) == pd.Index([3.0])
        """
        EinNew = pd.Index(self.check_InputValues(Ein))
        return EinNew.difference(self.data.index)

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
        >>> M = 238.05077040419212
        >>> xs = Xs.from_xs0K("u238.0.2", M)
        >>> os.chdir(wd)


        >>> T = 300
        >>> assert xs.get_Tcalc(T) == pd.Index([T])
        """
        Tnew = pd.Index(self.check_InputValues(temperatures))
        return Tnew.difference(self.data.columns)

    def get_EinInterp(self, Ein: [float, Iterable[float]]) -> pd.Index:
        """
        Get from the new incident energies, the energies already in the object

        Parameters
        ----------
        Ein: Union[float, Iterable[float]]
            The incident energy in eV

        Returns
        -------
        Union[float, Iterable[float]]
            The incident energy in eV

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> xs = Xs.from_xs0K("u238.0.2", M)
        >>> os.chdir(wd)

        >>> M = 238.05077040419212
        >>> Ein = [1.0, 3.0]
        >>> assert xs.get_EinInterp(Ein) == pd.Index([1.0])
        """
        EinNew = pd.Index(self.check_InputValues(Ein))
        return self.data.index.intersection(EinNew)

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
        >>> M = 238.05077040419212
        >>> xs = Xs.from_xs0K("u238.0.2", M)
        >>> os.chdir(wd)

        >>> M = 238.05077040419212
        >>> T = [0, 300]
        >>> assert xs.get_Tinterp(T) == pd.Index([0])
        """
        Tnew = pd.Index(self.check_InputValues(temperatures))
        return self.data.columns.intersection(Tnew)

    def get_output(self, data: [list, np.ndarray], T: Iterable = None,
                   Ein: Iterable = None) -> [pd.Series, pd.DataFrame]:
        """
        Get the output data
        Parameters
        ----------
        data: list, np.ndarray
            The output data
        T: Iterable, None
            The new temperatures calculated temperatures
        Ein: Iterable, None
            The new incident energies calculated

        Returns
        -------
        pd.Series, pd.DataFrame
            The output data in the corresponding format
        """
        # Check the temperature and incident energy
        T_ = self.data.columns if T is None else pd.Index(
            self.check_InputValues(T), name="T")
        Ein_ = self.data.index if Ein is None else pd.Index(
            self.check_InputValues(Ein), name="Ein")
        return pd.DataFrame(data, index=Ein_, columns=T_)

    def get_EinTcomb(self, Tnew: Iterable, EinGrid: Iterable) -> list:
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
        >>> M = 238.05077040419212
        >>> xs = Xs.from_xs0K("u238.0.2", M)
        >>> os.chdir(wd)

        >>> M = 238.05077040419212
        >>> Tnew = [300, 100]
        >>> pd.Series(xs.get_EinTcomb(Tnew, np.array([1, 2, 3])))
        0    (300, 1)
        1    (300, 2)
        2    (300, 3)
        3    (100, 1)
        4    (100, 2)
        5    (100, 3)
        dtype: object
        >>> pd.Series(xs.get_EinTcomb(Tnew, np.array([[1, 2], [3, 4]])))
        0    (300, 1)
        1    (300, 2)
        2    (100, 3)
        3    (100, 4)
        dtype: object
        """
        if len(EinGrid.shape) == 2:
            N, M = EinGrid.shape[0], EinGrid.shape[1]
            return [(Tnew[i], EinGrid[i, j]) for i in range(N) for j in range(M)]
        else:
            return [(T, Ein) for T in Tnew for Ein in EinGrid]

    def _compute_sigma1(self, Tnew: Iterable, EinGrid: [Iterable, None]) -> list:
        """
        Calculate the elastic scattering cross section at new temperatures using
        sigma1 method from Njoy

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
            The elastic scattering cross section in barns for the new
        """
        # Get the temperature and the incident energy combinations: (T, Ein)
        EinTcomb = self.get_EinTcomb(Tnew, EinGrid)

        # Create a Dask bag from the list of (T, Ein) combinations.
        bag = db.from_sequence(EinTcomb, npartitions=os.cpu_count())

        # Calculate xs using SIGMA1 to each element in the bag using the `map` function.
        bag = bag.map(lambda x: Dxs.from_sigma1(self.xs0Kcomplete, x[1], self.M, x[0], default_Eout(x[1])).integral)

        # Trigger the parallel computation and collect the results into a list.
        return bag.compute()

    @staticmethod
    def _calc_alpha0(T: float, EinGrid: np.ndarray, xs0K: pd.Series, M: float,
                     *args, theta: np.ndarray = np.arange(1, 181, 40),
                     **kwargs) -> np.ndarray:
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
        theta: np.ndarray
            The scattering angle in degrees. The default is np.arange(0, 181, 1).
        model: str, optional
            The model to use for the calculation. The options are:
            - "fgm": Use the free gas model (default)
            - "sct": Use Short Collision Time model
            - "pdos": Use the phonon expansion model

        Extra parameters for sct
        -------------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.

        Extra parameters for pdos
        --------------------------
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
        pd.Series
            The elastic scattering cross section in barns for the given
            temperature and incident energy using alpha0 model

        Examples
        --------
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> EinGrid = np.array([2.0, 6.67])
        >>> xs = Xs.from_xs0K("u238.0.2", M, EinGrid)
        >>> os.chdir(wd)

        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> T = 300
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> Xs._calc_alpha0(T, EinGrid, xs.xs0Kcomplete, M, model="fgm").round(6)
        array([  9.085972, 466.29523 ])

        >>> Xs._calc_alpha0(T, EinGrid, xs.xs0Kcomplete, M, pdos, model="sct").round(6)
        array([  9.024954, 458.934423])

        >>> Xs._calc_alpha0(T, EinGrid, xs.xs0Kcomplete, M, pdos, model="pdos").round(6)
        array([  8.602954, 469.614362])
        """
        # Generate the resutls for each incident energy
        dxsIntegral = []

        # Calculate the integral for each incident energy
        for Ein in EinGrid:
            # Generate the outgoing energy grid
            Eout = default_Eout(Ein)

            # Calculate the alpha0
            alpha0 = ScatFunc.from_model(Ein, M, T, Eout, theta, *args, **kwargs).alpha0

            # Calculate integral of the differential cross section for the outgoing energy
            dxsIntegral.append(Dxs.from_alpha(xs0K, alpha0, Ein, M, T, Eout, *args, **kwargs).integral)

        # Return the results in the corresponding format
        return np.array(dxsIntegral)

    def _compute_alpha0(self, Tnew: Iterable, EinGrid: Iterable, *args,
                        **kwargs) -> list:
        """
        Calculate the elastic scattering cross section at new temperatures using
        alpha0 model

        Parameters
        ----------
        Tnew: Iterable
            The new temperatures to calculate
        EinGrid: Iterable, None
            The incident energy grid in eV. If not provided, it will be taken
            from the class attribute.
        T: float
            The temperature in K
        model: str, optional
            The model to use for the calculation. The options are:
            - "fgm": Use the free gas model (default)
            - "sct": Use Short Collision Time model
            - "pdos": Use the phonon expansion model

        Extra parameters for sct
        -------------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.

        Extra parameters for pdos
        --------------------------
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
        list
            The elastic scattering cross section in barns for the new
            temperatures. Each value is the incident introduced by the user or
            the default.
        """
        # Update the arguments
        args = (self.xs0Kcomplete, self.M) + args

        # Calculation:
        if len(EinGrid.shape) == 1:
            return [Xs._calc_alpha0(T, EinGrid, *args, **kwargs)
                    for T in Tnew]
        else:
            return [Xs._calc_alpha0(Tnew[i], EinGrid[i], *args, **kwargs)
                    for i in range(len(Tnew))]

    def _compute(self, Tnew: Iterable, EinGrid: Iterable, *args,
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
        >>> M = 238.05077040419212
        >>> EinGrid = np.array([2.0, 6.67])
        >>> xs = Xs.from_xs0K("u238.0.2", M, EinGrid)
        >>> os.chdir(wd)

        >>> Tnew = [300, 100]
        >>> pd.DataFrame(xs._compute(Tnew, EinGrid, algorithm="sigma1"), index=Tnew, columns=EinGrid)
                 2.00        6.67
        300  9.086237  455.670534
        100  9.086957  664.556512

        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> pd.DataFrame(xs._compute(Tnew, EinGrid, algorithm="alpha0", model="fgm"), index=Tnew, columns=EinGrid)
                 2.00        6.67
        300  9.085972  466.295230
        100  9.086873  677.372091

        >>> pd.DataFrame(xs._compute(Tnew, EinGrid, pdos, algorithm="alpha0", model="sct"), index=Tnew, columns=EinGrid)
                 2.00        6.67
        300  9.024954  458.934423
        100  8.627295  617.394144

        >>> pd.DataFrame(xs._compute(Tnew, EinGrid, pdos, algorithm="alpha0", model="pdos"), index=Tnew, columns=EinGrid)
                 2.00        6.67
        300  8.602954  469.614362
        100  5.835051  646.772904
        """
        # Calculate the cross section for the new temperatures
        if algorithm == "sigma1":
            results = self._compute_sigma1(Tnew, EinGrid)
        elif algorithm == "alpha0":
            results = self._compute_alpha0(Tnew, EinGrid, *args, **kwargs)
        else:
            raise ValueError("invalid algorithm")

        # Return the results in the corresponding format
        return np.array(results).reshape(len(Tnew), -1)

    def calc_T(self, T: float, *args, inplace: bool = False, **kwargs):
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
            - "alpha0": Calculate the elastic scattering cross section with the
                        alpha0
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
        >>> M = 238.05077040419212
        >>> EinGrid = np.array([0.065625, 2.0, 4.0, 5.0, 6.67, 7.0])
        >>> xs = Xs.from_xs0K("u238.0.2", M, EinGrid)
        >>> os.chdir(wd)

        # Sigma1:
        >>> T = [300, 100]
        >>> xs.calc_T(T, algorithm="sigma1").data
        T                 0           100         300
        Ein
        0.065625     9.411657    9.414734    9.419595
        2.000000     9.085342    9.086957    9.086237
        4.000000     8.481975    8.482804    8.482893
        5.000000     7.805580    7.805703    7.805682
        6.670000  1269.792131  664.556512  455.670534
        7.000000    19.825115   19.893739   20.039076

        # Alpha0 with FGM:
        >>> xs.calc_T(T, algorithm="alpha0", model="fgm").data
        T                 0           100         300
        Ein
        0.065625     9.411657    9.412320    9.411831
        2.000000     9.085342    9.086873    9.085972
        4.000000     8.481975    8.484021    8.482886
        5.000000     7.805580    7.807253    7.805968
        6.670000  1269.792131  677.372091  466.295230
        7.000000    19.825115   19.897556   20.036626

        # Alpha0 with SCT:
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> xs.calc_T(T, pdos, algorithm="alpha0", model="sct").data
        T                 0           100         300
        Ein
        0.065625     9.411657    9.210666    9.393819
        2.000000     9.085342    8.627295    9.024954
        4.000000     8.481975    8.105054    8.420278
        5.000000     7.805580    7.488588    7.748244
        6.670000  1269.792131  617.394144  458.934423
        7.000000    19.825115   19.298819   19.915131

        # Alpha0 with PDOS:
        >>> xs.calc_T(T, pdos, algorithm="alpha0", model="pdos").data
        T                 0           100         300
        Ein
        0.065625     9.411657    0.033061    0.224653
        2.000000     9.085342    5.835051    8.602954
        4.000000     8.481975    7.538804    8.460237
        5.000000     7.805580    7.316357    7.801128
        6.670000  1269.792131  646.772904  469.614362
        7.000000    19.825115   19.541023   20.052674
        """
        # Get the new temperatures to calculate
        Tnew = self.get_Tcalc(T)

        # Check if the temperatures are already calculated
        if Tnew.empty:
            warnings.warn("All the temperatures are already calculated")
            return self

        # Calculate the cross section for the new temperatures
        xsTValues = self._compute(Tnew, self.data.index, *args, **kwargs).T

        # Get the output data in the corresponding format
        xsT = self.get_output(xsTValues, T=Tnew)

        return self.update_data(xsT, inplace)

    def calc_Ein(self, Ein: [float, np.ndarray], *args, inplace: bool = False,
                 **kwargs):
        """
        Calculate the elastic scattering cross section at incident energies
        using the selected algorithm.

        Parameters
        ----------
        Ein: Union[float, Iterable[float]]
            The incident energy in eV
        algorithm : str, optional
            The algorith to use for the calculation. The options are:
            - "sigma1": Calculate the elastic scattering cross section with the
                        sigma1 method from Njoy
            - "alpha0": Calculate the elastic scattering cross section with the
                        alpha0
        inplace : bool, optional
            If True, the data is stored in the class attribute, otherwise it
            is returned

        Returns
        -------
        Xs
            New object with the updated data or None if inplace is True, so
            the data is stored in the class attribute and modified in place.

        Examples
        --------
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> M = 238.05077040419212
        >>> EinGrid = np.array([0.065625, 2.0, 4.0, 5.0, 6.67, 7.0])
        >>> xs = Xs.from_xs0K("u238.0.2", M, EinGrid)
        >>> os.chdir(wd)

        # Add several temperatures:
        >>> xs = xs.calc_T([300, 100])

        # Sigma1:
        >>> xs.calc_Ein(1.0, algorithm="sigma1").data
        T                 0           100         300
        Ein
        0.065625     9.411657    9.414734    9.419595
        1.000000     9.254035    9.271830    9.270573
        2.000000     9.085342    9.086957    9.086237
        4.000000     8.481975    8.482804    8.482893
        5.000000     7.805580    7.805703    7.805682
        6.670000  1269.792131  664.556512  455.670534
        7.000000    19.825115   19.893739   20.039076


        # Several incident energies with alpha0 model FGM:
        >>> Ein = [1.0, 2.0]
        >>> xs.calc_Ein(Ein, algorithm="alpha0", model="fgm").data
        T                 0           100         300
        Ein
        0.065625     9.411657    9.414734    9.419595
        1.000000     9.254035    9.271237    9.270165
        2.000000     9.085342    9.086957    9.086237
        4.000000     8.481975    8.482804    8.482893
        5.000000     7.805580    7.805703    7.805682
        6.670000  1269.792131  664.556512  455.670534
        7.000000    19.825115   19.893739   20.039076

        # With alpha0 model SCT:
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> xs.calc_Ein(Ein, pdos, algorithm="alpha0", model="sct").data
        T                 0           100         300
        Ein
        0.065625     9.411657    9.414734    9.419595
        1.000000     9.254035    8.817159    9.218082
        2.000000     9.085342    9.086957    9.086237
        4.000000     8.481975    8.482804    8.482893
        5.000000     7.805580    7.805703    7.805682
        6.670000  1269.792131  664.556512  455.670534
        7.000000    19.825115   19.893739   20.039076

        # With alpha0 model pdos:
        >>> xs.calc_Ein(Ein, pdos, algorithm="alpha0", model="pdos").data
        T                 0           100         300
        Ein
        0.065625     9.411657    9.414734    9.419595
        1.000000     9.254035    3.188988    6.986018
        2.000000     9.085342    9.086957    9.086237
        4.000000     8.481975    8.482804    8.482893
        5.000000     7.805580    7.805703    7.805682
        6.670000  1269.792131  664.556512  455.670534
        7.000000    19.825115   19.893739   20.039076
        """
        # Get the new incident energies to calculate
        EinGrid = self.get_EinCalc(Ein)
        if EinGrid.size == 0:
            warnings.warn("All the incident energies are already calculated")
            return self

        # Drop the 0K data
        temp = self.data.columns.drop([0])

        # Calculate the cross section for the new incident energies
        xsEin = self.interp_Ein(EinGrid, T=0)

        # Interpolate the 0K data to the new incident energies:
        if temp.empty:
            return Xs(self.M, 0, xsEin, xs0Kcomplete=self.xs0Kcomplete)

        # Compute the calculation
        xsEinValuesCalc = self._compute(temp, EinGrid,  *args, **kwargs).T

        # Get the output data in the appropriate format
        xsEinCalc = self.get_output(xsEinValuesCalc, Ein=EinGrid, T=temp)

        # concat the data
        xsCalc = pd.concat([xsEin, xsEinCalc], axis=1)

        return self.update_data(xsCalc, inplace, axis=0)

    def interp_Ein(self, Ein: [float, np.ndarray], T: [float, Iterable] = None,
                   kind: str = "slinear", bounds_error: bool = True) -> [pd.Series, pd.DataFrame]:
        """
        Interpolate Xs objet to a new Ein. If T is provided, the interpolation
        is done for the selected temperatures.

        Parameters
        ----------
        Ein: float, np.ndarray
            New Ein grid
        T: float, Iterable, optional
            The temperatures to interpolate. If not provided, all the
            temperatures are interpolated.
        kind: str, optional
            The kind of interpolation to use. The options are:
            - "linear": Linear interpolation
            - "slinear": Spline interpolation of first degree
            - "quadratic": Quadratic interpolation
            - "cubic": Cubic interpolation

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
        >>> M = 238.05077040419212
        >>> EinGrid = np.array([0.065625, 2.0, 4.0, 5.0, 6.67, 7.0])
        >>> xs = Xs.from_xs0K("u238.0.2", M, EinGrid)
        >>> os.chdir(wd)

        # Add several temperatures:
        >>> xs = xs.calc_T([300, 100])

        # Sigma1:
        >>> xs.interp_Ein([3.0, 4.5])
        T         0         100       300
        Ein
        3.0  8.783659  8.784881  8.784565
        4.5  8.143778  8.144254  8.144288

        >>> xs.interp_Ein([3.0], T=0)
        T           0
        Ein
        3.0  8.783659

        >>> xs.interp_Ein([3.0, 4.5], T=100)
        T         100
        Ein
        3.0  8.784881
        4.5  8.144254

        >>> xs.interp_Ein([3.0])
        T         0         100       300
        Ein
        3.0  8.783659  8.784881  8.784565
        """
        # Get the new incident energies to calculate
        Ein = np.unique(Ein)

        # Select the temperatures to interpolate:
        Tinterp = self.data.columns if T is None else self.get_Tinterp(T)
        if Tinterp.empty:
            raise ValueError("The temperatures are not in the object")

        # Get the xs data to interpolate:
        xsInterpData = self.data.loc[::, Tinterp]

        # Interpolate the data
        xsInterp = xsInterpData.apply(lambda xsT: interpolation(xsT, Ein, kind=kind, bounds_error=bounds_error))

        # Get the output data in the appropriate format
        return self.get_output(xsInterp, T=T, Ein=Ein)

    def get_4PCFxs(self, Ein: float, T: float, Eout: np.ndarray,
                   theta: np.ndarray, *args, **kwargs) -> pd.DataFrame:
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
        >>> M = 238.05077040419212
        >>> xs = Xs.from_xs0K("u238.0.2", M)
        >>> os.chdir(wd)

        >>> T = 300
        >>> Ein = 2.0
        >>> Eout = np.linspace(2.0 * 0.9, 2.0 * 1.1, 5)
        >>> theta = np.array([180, 120, 90, 60, 30])
        >>> index = pd.Index(theta, name="theta")
        >>> xs.get_4PCFxs(Ein, T, Eout, theta, algorithm="sigma1").set_axis(index, axis=0)
        Eout        1.8       1.9       2.0       2.1       2.2
        theta
        180    9.102355  9.092121  9.081758  9.071139  9.060521
        120    9.104914  9.094623  9.084231  9.073561  9.062890
        90     9.105581  9.095328  9.084990  9.074371  9.063732
        60     9.106114  9.095876  9.085560  9.074971  9.064341
        30     9.106574  9.096350  9.086046  9.075473  9.064836

        >>> xs.get_4PCFxs(Ein, T, Eout, theta, algorithm="alpha0", model="fgm").set_axis(index, axis=0)
        Eout        1.8       1.9       2.0       2.1       2.2
        theta
        180    9.102355  9.092121  9.081758  9.071139  9.060521
        120    9.105038  9.094818  9.084497  9.073897  9.063295
        90     9.105296  9.095061  9.084744  9.074144  9.063526
        60     9.105840  9.095607  9.085300  9.074717  9.064094
        30     9.106301  9.096081  9.085785  9.075215  9.064583

        >>> mu = np.cos(np.deg2rad(theta))
        >>> T4PCF = T * (1 + mu) / 2
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(T4PCF[1::], rho_in_energy_U238, interv_in_energy_U238)
        >>> xs.get_4PCFxs(Ein, T, Eout, theta, pdos, algorithm="alpha0", model="sct").set_axis(index, axis=0)
        Eout        1.8       1.9       2.0       2.1       2.2
        theta
        180    9.102355  9.092121  9.081758  9.071139  9.060521
        120    8.421971  8.414644  8.407319  8.399813  8.392374
        90     8.868318  8.857919  8.847491  8.836837  8.826207
        60     8.996254  8.985686  8.975071  8.964208  8.953327
        30     9.036288  9.025780  9.015211  9.004388  8.993517

        >>> xs.get_4PCFxs(Ein, T, Eout, theta, pdos, algorithm="alpha0", model="pdos").set_axis(index, axis=0)
        Eout        1.8       1.9       2.0       2.1       2.2
        theta
        180    9.102355  9.092121  9.081758  9.071139  9.060521
        120    4.877076  4.976241  5.082816  5.177156  5.265810
        90     6.893454  6.979544  7.058797  7.136817  7.202696
        60     7.985702  8.045719  8.094413  8.147071  8.185316
        30     8.426736  8.469297  8.500190  8.536056  8.558696
        """
        # Get the cosine of the angle
        mu = np.sort(np.cos(np.deg2rad(theta)))

        # Get the temperature grid for the 4PCF model:
        T4PCF = T * (1 + mu) / 2

        # Get the incident energy grid:
        Ein4PCF = EinMat4PCF(Ein, Eout, mu[::, np.newaxis], self.M)

        # Get the output data in the appropriate format
        Eout, mu = pd.Index(Eout, name="Eout"), pd.Index(mu, name="mu")

        # Get the temperatures:
        Tcalc, Tinterp = self.get_Tcalc(T4PCF), self.get_Tinterp(T4PCF)

        # Interpolation:
        if Tinterp.empty:
            xsInterp = None
        else:
            kind = kwargs.pop("kind", "slinear")
            bounds = kwargs.pop("bounds_error", True)
            xsInterpValues = self.interp_Ein(Ein4PCF[np.isin(T4PCF, Tinterp)], T=Tinterp, kind=kind, bounds_error=bounds)
            # Reshape the data row wise if there are several interpolated temperatures:
            if len(Tinterp) > 1:
                xsInterpValues = {T: xsInterpValues.loc[Ein4PCF[T4PCF == T], T] for T in Tinterp}
            xsInterp = pd.DataFrame(xsInterpValues).T.set_axis(Eout, axis=1)

        # Calculation
        if Tcalc.empty:
            xsCalc = None
        else:
            xsCalcValues = self._compute(Tcalc, Ein4PCF[np.isin(T4PCF, Tcalc)], *args, **kwargs)
            xsCalc = pd.DataFrame(xsCalcValues, index=Tcalc, columns=Eout)
        return pd.concat([xsInterp, xsCalc]).set_axis(mu, axis=0)


@nb.jit(nopython=True, cache=True)
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
    >>> xs0K = Xs.read_xs("u238.0.2")
    >>> os.chdir(wd)

    # Generate Broadening test results:
    >>> T = 1000
    >>> Ein = 2.0
    >>> Eout = default_Eout(Ein)
    >>> M = 238.05077040419212
    >>> float(round(Dxs.from_sigma1(xs0K, Ein, M, T, Eout).integral, 2))
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
    return np.unique(np.concatenate((EoutGreat, EoutSmall, EoutMid)))


@vectorize(["float64(float64, float64, float64, float64)"],
           target="parallel", cache=True)
def EinMat4PCF(Ein: float, Eout: np.ndarray, mu: np.ndarray,
                    M: float) -> float:
    """
    Get the incident energy matrix for 4PCF model.

    Parameters
    ----------
    Ein: float
        The incident energy of the neutron in eV
    Eout: float
        The neutron outgoing energy grid in eV
    mu: float
        The cosine of the neutron outgoing angle
    M: float
        Mass of the material in amu

    Returns
    -------
    Ein4PCF: float
        Incident energy matrix for 4PCF model

    Examples
    --------
    >>> Ein = 2.0
    >>> Eout = np.linspace(2.0 * 0.9, 2.0 * 1.1, 5)
    >>> mu = np.array([-1.0, -0.5, 0.0, 0.5, 0.9])
    >>> M = 238.05077040419212
    >>> values = EinMat4PCF(Ein, Eout, mu[::, np.newaxis], M)
    >>> pd.DataFrame(values, index=pd.Index(mu, name="mu"), columns=pd.Index(Eout, name="Eout"))
        Eout       1.8       1.9       2.0       2.1       2.2
        mu
        -1.0  1.916519  1.966736  2.016949  2.067159  2.117367
        -0.5  1.912284  1.962499  2.012712  2.062923  2.113132
         0.0  1.908051  1.958263  2.008474  2.058686  2.108898
         0.5  1.903825  1.954028  2.004237  2.054452  2.104671
         0.9  1.900524  1.950660  2.000847  2.051083  2.101362
    """
    EinArno = (Eout + Ein + get_alphaRecoil(Eout, Ein, M, mu)) / 2
    return EinArno
