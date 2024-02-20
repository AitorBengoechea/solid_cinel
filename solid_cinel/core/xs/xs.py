"""
Python for working with Angle integrated scattering xs at different temperature.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
from scipy.constants import physical_constants as const
from typing import Iterable, Union
from solid_cinel.core.xs.dxs import Dxs
import os
import dask

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
                 *args, **kwargs):
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
        self._M = M
        temperatures = self.check_T(temperatures)
        data_frame = pd.DataFrame(*args, **kwargs)
        data_frame.columns = pd.Index(temperatures, name="T")
        self.data = data_frame

    @property
    def M(self):
        """
        Property that returns the value of 'M' if it exists, or raises an
        AttributeError if it doesn't.

        Returns:
        The value of 'M'.
        """
        if hasattr(self, '_M'):
            return self._M
        else:
            raise AttributeError(
                "'{}' object has no attribute 'M'".format(type(self).__name__))

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

    def get_xs0K(self) -> pd.Series:
        """
        Get the 0K scattering function

        Returns
        -------

        """
        if 0 in self.data.columns:
            return self.data.loc[::, 0]
        else:
            raise ValueError("0K data not found")

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


    @staticmethod
    def _calc_sigma1(xs0K: pd.Series, EinGrid: Iterable, M: float, T: float):
        """
        Calculate the elastic scattering cross section at temperature T and
        energy E

        Parameters
        ----------
        xs0K: pd.Series
            The 0K scattering function
        EinGrid: Iterable
            The incident energy grid in eV
        M: float
            The mass of the nucleus in amu
        T: float
            The temperature in K

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
        >>> EinGrid = [1, 2, 3]
        >>> Xs._calc_sigma1(xs0K, EinGrid, M, T)
        Ein
        1    9.270573
        2    9.086237
        3    8.843855
        dtype: float64
        """
        xs_T = [Dxs.from_sigma1(xs0K, Ein, M, T, default_Eout(Ein)).integral
                for Ein in EinGrid]
        return pd.Series(xs_T, index=pd.Index(EinGrid, name="Ein"))


    @classmethod
    def from_sigma1(cls, T: float, M: float, xs0Kshort: pd.Series,
                    xs0Kcomplete: pd.Series = None):
        """
        Calculate the elastic scattering cross section for a nucleus with mass
        M at temperature T.

        Parameters
        ----------
        xs0K: pd.Series
            The 0K scattering function with the incident energy as index
        M: float
            The mass of the nucleus in amu
        T: float
            The temperature in K

        Parameters for calc_sigma1:
        xs0K : pd.Series, optional
            The 0K scattering function with all the data. If not provided, it
            will be taken from the class attribute. These option is available to
            avoid the doppler broadening of all the Ein.

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
        xs = cls(M, 0, xs0Kshort)
        xs.calc_sigma1(T, xs0Kcomplete=xs0Kcomplete, inplace=True)
        return xs

    def calc_sigma1(self, T: float, M: float = None, xs0Kcomplete: pd.Series = None,
                    inplace: bool = False) -> [None, pd.Series]:
        """
        Calculate the elastic scattering cross section at temperature T and
        energy E

        Parameters
        ----------
        T : float
            Temperature in K
        M : float, optional
            The mass of the nucleus in amu
        xs0K : pd.Series, optional
            The 0K scattering function
        inplace : bool, optional
            If True, the data is stored in the class attribute, otherwise it
            is returned

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
        >>> xs = Xs(M, 0, xs0K.iloc[0:10000:1000])
        >>> xs.calc_sigma1(T, xs0Kcomplete=xs0K, inplace=True)
        >>> xs.data
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
        xs0Kcomplete = self.get_xs0K() if xs0Kcomplete is None else xs0Kcomplete
        M = self.M if M is None else M
        Tnew = self.check_T(T)
        # Set the number of threads to the number of available processors
        with dask.config.set(num_workers=num_processors):
            # Create a delayed computation for each temperature
            delayed_computations = [
                dask.delayed(self._calc_sigma1)(xs0Kcomplete, self.data.index, M, T) for T in Tnew
            ]

            # Compute all at once
            results = dask.compute(*delayed_computations)
        # Create DataFrame from results
        xs_T = pd.DataFrame({T: result for T, result in zip(Tnew, results)})
        if inplace:
            self.data = pd.concat([self.data, xs_T], axis=1).sort_index(axis=1)
        else:
            return xs_T

    def calc_T(self, T:float, algorith: str = "sigma1",
               xs0Kcomplete: pd.Series = None, inplace: bool=False):
        """
        Calculate the elastic scattering cross section at temperature T and
        energy E

        Parameters
        ----------
        T : float
            Temperature in K
        inplace : bool, optional
            If True, the data is stored in the class attribute, otherwise it
            is returned

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
        >>> T = [300, 100]
        >>> xs = Xs(M, 0, xs0K.iloc[0:10000:1000])
        >>> xs.calc_T(T, xs0Kcomplete=xs0K, inplace=True)
        >>> xs.data
        T                 0            100         300
        Ein
        0.00001      9.420892    21.618999   36.117710
        11.23650     9.239644     9.239677    9.240652
        34.70286     1.146785     1.146882    1.147109
        58.18538     9.794358     9.794220    9.793941
        80.66597     1.639337    21.909253   23.581216
        97.56808     4.895060     4.893586    4.891338
        116.82090  457.760100  1236.911682  931.532221
        145.67660   39.688900    14.746250   13.281879
        165.85470   11.753670    11.887786   12.634276
        200.79510   15.734840    15.734518   15.733460
        """
        if algorith == "sigma1":
            self.calc_sigma1(T, xs0Kcomplete=xs0Kcomplete, inplace=inplace)


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