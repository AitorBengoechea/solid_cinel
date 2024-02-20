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
from solid_cinel.core.xs.ddxs import DDxs
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
        self.get_xs0Kcomplete(xs0Kcomplete)

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

    def get_Tnew(self, temperatures: Union[float, Iterable[float]]) -> pd.Index:
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
        >>> T = 0
        >>> xs = Xs(M, T, xs0K)
        >>> xs.get_Tnew(300)
        Int64Index([300], dtype='int64')
        """
        Tnew = pd.Index(self.check_T(temperatures))
        return Tnew.difference(self.data.columns)

    def get_xs0K(self) -> pd.Series:
        """
        Get the 0K scattering function

        Returns
        -------
        pd.Series
            The 0K scattering function with the data in the class attribute
            data
        """


    def get_xs0Kcomplete(self, xs0Kcomplete: [pd.Series, None]) -> pd.Series:
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
    def _calc_sigma1(xs0K: pd.Series, EinGrid: Iterable, M: float,
                     T: float) -> pd.Series:
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

    def compute_sigma1results(self, xs0Kcomplete: pd.Series, M: float,
                              Tnew: Iterable) -> list:
        """
        Calculate the elastic scattering cross section at new temperatures using
        dask and Sigma1 algorithm from Njoy

        Parameters
        ----------
        xs0Kcomplete: pd.Series
            The 0K scattering function with all the data
        M: float
            The mass of the nucleus in amu
        Tnew: Iterable
            The new temperatures to calculate

        Returns
        -------
        list
            The elastic scattering cross section in barns
        """
        with dask.config.set(num_workers=os.cpu_count()):
            delayed_computations = [
                dask.delayed(self._calc_sigma1)(xs0Kcomplete, self.data.index,
                                                M, T) for T in Tnew]
            return dask.compute(*delayed_computations)

    def update_data(self, xs_T: pd.DataFrame, inplace: bool) -> [None, pd.DataFrame]:
        """
        Update the data with the new results

        Parameters
        ----------
        xs_T: pd.DataFrame
            The new results
        inplace: bool
            If True, the data is stored in the class attribute, otherwise it
            is returned as a new object

        Returns
        -------
        pd.DataFrame
            The new data
        """
        dataNew = pd.concat([self.data, xs_T], axis=1).sort_index(axis=1)
        if inplace:
            self.data = dataNew
        else:
            return Xs(self.M, dataNew.columns, dataNew)

    def calc_sigma1(self, T: float, inplace: bool = False) -> [None, pd.Series]:
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
        >>> T = 300
        >>> xs = Xs(M, 0, xs0K.iloc[0:10000:1000], xs0Kcomplete=xs0K)
        >>> xs.calc_sigma1(T, inplace=True)
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

        >>> xs = Xs(M, 0, xs0K.iloc[0:10000:1000], xs0Kcomplete=xs0K)
        >>> xs.calc_sigma1(T).data
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
        xs0Kcomplete, M = self.xs0Kcomplete, self.M
        Tnew = self.get_Tnew(T)
        if Tnew.empty:
            return self
        results = self.compute_sigma1results(xs0Kcomplete, M, Tnew)
        xs_T = pd.DataFrame({T: result for T, result in zip(Tnew, results)})
        return self.update_data(xs_T, inplace)


    @classmethod
    def from_sigma1(cls, T: float, M: float, xs0Kshort: pd.Series,
                    xs0Kcomplete: pd.Series = None, inplace: bool = False):
        """
        Calculate the elastic scattering cross section for a nucleus with mass
        M at temperature T.

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
        xs = cls(M, 0, xs0Kshort)
        # Get cls attributes using the available information
        xs.get_xs0Kcomplete(xs0Kcomplete)
        return xs.calc_sigma1(T, inplace=inplace)

    def calc_T(self, T:float, algorithm: str = "sigma1", inplace: bool = False):
        """
        Calculate the elastic scattering cross section at temperature T and
        energy E

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
        >>> xs = Xs(M, 0, xs0K.iloc[0:10000:1000], xs0Kcomplete=xs0K)
        >>> xs.calc_T(T).data
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
        if algorithm == "sigma1":
            return self.calc_sigma1(T, inplace=inplace)
        else:
            raise ValueError("Invalid algorithm")


    def get_dxs(self, Ein: float, T: float, Eout: np.ndarray, *args,
                algorithm: str = "sigma1", **kwargs) -> Dxs:
        """
        Calculate the differential cross section for elastic scattering
        at temperature T and energy E using the algorithm specified

        Parameters
        ----------
        Ein: float
            Incident energy in eV
        T: float
            Temperature in K
        Eout: np.ndarray
            Outgoing energy grid in eV
        algorithm: str, optional
            The algorithm to use for the calculation. The options are:
            - "sigma1": Calculate the elastic scattering cross section with the
                        sigma1 method from Njoy
            - "alpha": Calculate the elastic scattering cross section with the
                       alpha zero method.

        Extra Parameters for alpha
        ---------------------------
        model: str, optional
            The model to use for the calculation. The options are:
                - "pdos": Phonon expansion model
                - "fgm" : Free Gas Model (Default)
                - "sct" : Short Collision Time model

        Extra Parameters for SCT (algorithm = "alpha")
        ----------------------------------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object

        Extra Parameters for pdos (algorithm = "alpha")
        ----------------------------------------------
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
        Dxs
            The differential cross section for elastic scattering
            at temperature T and energy E.

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
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> xs = Xs(M, 0, xs0K.iloc[0:10000:1000], xs0Kcomplete=xs0K)

        # DOPUSH algorithm:
        >>> xs.get_dxs(Ein, T, Eout).data.iloc[::100]
        sigma1
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

        >>> xs.get_dxs(Ein, T, Eout, algorithm="alpha").data.iloc[::100]
        alpha
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
        print(algorithm)
        if algorithm == "sigma1":
            return Dxs.from_sigma1(self.xs0Kcomplete, Ein, self.M, T, Eout)
        elif algorithm == "alpha":
            return Dxs.from_recoil(self.xs0Kcomplete, Ein, self.M, T, Eout,
                                   *args, **kwargs)
        else:
            raise ValueError("Invalid algorithm")
    def get_ddxs(self, Ein: float, T: float, Eout: np.ndarray,
                 theta: np.ndarray, *args, algorithm: str = "sigma1",
                 **kwargs) -> DDxs:
        """
        Calculate the double differential cross section for elastic scattering
        at temperature T and energy E using the algorithm specified and the 4PCF
        method in ddxs.py

        Parameters
        ----------
        Ein: float
            Incident energy in eV
        T: float
            Temperature in K
        Eout: np.ndarray
            Outgoing energy grid in eV
        theta:  np.ndarray
            Scattering angle grid in grades
        algorithm: str, optional
            The algorithm to use for the calculation. The options are:
            - "sigma1": Calculate the elastic scattering cross section with the
                        sigma1 method from Njoy
            - "alpha": Calculate the elastic scattering cross section with the
                       alpha zero method.
        model: str, optional
            The model to use for the calculation. The options are:
                - "pdos": Phonon expansion model
                - "fgm" : Free Gas Model (Default)
                - "sct" : Short Collision Time model

        Extra Parameters for sct
        ------------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object

        Extra Parameters for pdos
        -------------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object
        threshold : 'float', optional
            Minimun value to take into account in the creation of tauN functions. For T>200 is convenient to set into
            1.0e-14 to speed up the calculations. The default is 0.0.
        decimal: 'float'
            Decimal precision for the calculation of the expansion order.
            The default is 1.0e-6.
        order_max: 'int'
            Maximun expansion order. The default is 5000.

        Returns
        -------
        DDxs
            The double differential cross section for elastic scattering
            at temperature T and energy E.

        Examples
        --------
        # 0K xs data for U238:
        # 0K xs data for U238:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> Ein = 2.0
        >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 1000)
        >>> M = 238.05077040419212
        >>> theta = np.arange(0, 180, 10)[1::]
        >>> xs = Xs(M, 0, xs0K.iloc[0:10000:1000], xs0Kcomplete=xs0K)

        # Coercelle with sigma1 algorithm:
        >>> xs.get_ddxs(Ein, T, Eout, theta).data.iloc[::, ::200].round(6)
        Eout            1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -9.848078e-01  1.799454  12.011826  23.795201  15.058832  3.254168
        -9.396926e-01  1.676368  11.820165  24.045974  15.218228  3.207630
        -8.660254e-01  1.481046  11.484979  24.467361  15.486062  3.125737
        -7.660444e-01  1.229144  10.983197  25.064005  15.865282  3.002054
        -6.427876e-01  0.943760  10.284397  25.841170  16.359232  2.827887
        -5.000000e-01  0.654925   9.354774  26.802832  16.970446  2.593109
        -3.420201e-01  0.396320   8.165850  27.947664  17.698083  2.288171
        -1.736482e-01  0.197800   6.711538  29.260922  18.532824  1.908309
         6.123234e-17  0.074460   5.037219  30.697832  19.446272  1.461210
         1.736482e-01  0.018204   3.278979  32.148848  20.368969  0.978386
         3.420201e-01  0.002218   1.689219  33.366368  21.143932  0.525271
         5.000000e-01  0.000081   0.578041  33.810815  21.428972  0.191553
         6.427876e-01  0.000000   0.090864  32.347398  20.504444  0.033458
         7.660444e-01  0.000000   0.002704  26.829842  17.009134  0.001208
         8.660254e-01  0.000000   0.000001  14.852992   9.417271  0.000001
         9.396926e-01  0.000000   0.000000   1.824940   1.157149  0.000000
         9.848078e-01  0.000000   0.000000   0.000005   0.000003  0.000000

        # Coercelle with FGM algorithm:
        >>> xs.get_ddxs(Ein, T, Eout, theta, algorithm="alpha").data.iloc[::, ::200].round(6)
        Eout            1.80000    1.88008    1.96016    2.04024   2.12032
        mu
        -9.848078e-01  1.799454  12.011827  23.795201  15.058833  3.254168
        -9.396926e-01  1.676368  11.820171  24.045985  15.218236  3.207631
        -8.660254e-01  1.481047  11.484990  24.467384  15.486077  3.125740
        -7.660444e-01  1.229145  10.983212  25.064040  15.865304  3.002058
        -6.427876e-01  0.943762  10.284418  25.841231  16.359271  2.827893
        -5.000000e-01  0.654927   9.354802  26.802921  16.970502  2.593117
        -3.420201e-01  0.396321   8.165883  27.947781  17.698162  2.288181
        -1.736482e-01  0.197802   6.711574  29.261084  18.532927  1.908320
         6.123234e-17  0.074460   5.037252  30.698047  19.446408  1.461220
         1.736482e-01  0.018204   3.279005  32.149110  20.369142  0.978394
         3.420201e-01  0.002218   1.689235  33.366688  21.144134  0.525276
         5.000000e-01  0.000081   0.578047  33.811182  21.429205  0.191556
         6.427876e-01  0.000000   0.090865  32.347784  20.504689  0.033458
         7.660444e-01  0.000000   0.002704  26.830185  17.009357  0.001208
         8.660254e-01  0.000000   0.000001  14.853195   9.417400  0.000001
         9.396926e-01  0.000000   0.000000   1.824966   1.157166  0.000000
         9.848078e-01  0.000000   0.000000   0.000005   0.000003  0.000000
        """
        if algorithm == "sigma1":
            return DDxs.from_4PCF(self.xs0Kcomplete, Ein, self.M, T, Eout, theta, *args, **kwargs)
        elif algorithm == "alpha":
            return DDxs.from_4PCF_recoil(self.xs0Kcomplete, Ein, self.M, T, Eout, theta, *args, **kwargs)
        else:
            raise ValueError("Invalid algorithm")


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