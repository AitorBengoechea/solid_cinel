"""
Python file for working with Phonon Density Of States.

@author: AB272525
"""
from solid_cinel.core.generic import integrate
from solid_cinel.core.scattering_function.beta import Beta
from solid_cinel.core.material.vibration.tau import get_tauNfunc, get_tauNbeta
from scipy.interpolate import RectBivariateSpline
import pandas as pd
import numpy as np
from typing import Iterable, Union


# Examples variables:
rho_in_energy_str = '''
    0 .0066 .0264 .0594 .1055 .1649 .2374 .3232 .4221
    .5342 .6595 .7980 .9497 1.1146 1.2927 1.4839 1.6884
    2.0169 2.4373 2.9366 3.6133 4.6775 7.1346 7.3650
    7.5156 7.6733 7.8309 8.0740 8.4419 9.0595 9.6773
    7.3645 6.2674 5.1965 4.7958 4.8024 4.6841 4.4673
    4.1914 3.8169 3.3439 2.7855 3.2782 5.3082 8.5930
    12.3377 8.4616 5.6695 4.1585 2.6081 0.0
'''
rho_in_energy = np.fromstring(rho_in_energy_str, dtype=np.float64, sep=' ')
interv_in_energy = 0.0008


class Tpdos:
    """
    Object containing the method and properties of the phonon density of states
    for a certain temperature.
    """
    def __init__(self, T, *args, **kwargs):
        """
        Initialize of the pdos object.
        Parameters
        ----------
        T: 'float'
            Temperature in K.
        args: 'variables'
            variables for the creation of the pandas Series.
        kwargs: 'dict'
            Dictionary to create the pandas series of rho.
        Returns
        -------
        'TPdos'
            Object containing the method and properties of rho in beta.

        Examples
        --------
        Object initialization:
        >>> T = 20
        >>> p = Tpdos.from_dE(T, rho_in_energy, interv_in_energy)
        """
        self.T = T
        self.data = pd.Series(*args, **kwargs)

    @property
    def data(self) -> pd.Series:
        """Pandas Series containing the rho values in energy (index)."""
        return self._data

    @data.setter
    def data(self, rhoData: Union[pd.Series, Iterable]) -> pd.Series:
        """
        Data setter for rho to ensure the following properties of the data:
            - Shape of the data: 1 dimension
            - Energy index monotoally increasing
            - Rho values normalization

        Parameters
        ----------
        rho : pd.Series
            rho values in energy.

        Returns
        -------
        "pd.Series"
            Rho normalize.

        Raises
        ------
        TypeError
            Rho is not 1 dimension pd.Series
        SyntaxError
            Energy grid is not monotonically increasing.

        Examples
        --------
        Object initialization:
        >>> p = Epdos.from_dE(rho_in_energy, interv_in_energy)

        Test the results:
        >>> assert integrate(p.data) == 1.0
        """
        data_ = pd.Series(rhoData, name="rho")

        if data_.index.name != "beta":
            raise SyntaxError("Energy index must be beta")

        if not len(data_.shape) == 1:
            raise TypeError("Rho must have one dimension")

        if not data_.index.is_monotonic_increasing:
            raise SyntaxError("beta grid is not monotonically increasing")

        self._data = data_ / integrate(data_)

    @property
    def beta(self) -> Beta:
        """
        Initialize the Beta class with the information of S(alpha, -beta).
        matrix
        """
        return Beta(self.data.index.values)

    @classmethod
    def from_dE(cls, T: float, rho: Iterable, intervalE: float):
        """
        Extract rho in energy from the introduced data and create a pdos object
        for a certain temperature.

        Parameters
        ----------
        T: 'float'
            Temperature in K.
        rho: '1D iterable'
            rho values.
        intervalE: 'float'
            Energy interval in eV.

        Returns
        -------
        "TPdos"
            Rho normalize object.

        Examples
        --------
        Object initialization:
        >>> p = Tpdos.from_dE(800, rho_in_energy, interv_in_energy)
        >>> p.data.iloc[0:10]
        beta
        0.000000    0.000000
        0.011605    0.002837
        0.023209    0.011349
        0.034814    0.025536
        0.046418    0.045354
        0.058023    0.070890
        0.069627    0.102058
        0.081232    0.138943
        0.092836    0.181460
        0.104441    0.229651
        Name: rho, dtype: float64
        """
        rho_ = np.array(rho)
        grid = np.arange(len(rho_)) * intervalE
        return cls(T, rho_, index=Beta.from_dE(grid, T).to_index)

    @classmethod
    def from_file(cls, T: float, file: str, header=None, index_col=None,
                  usecols=None, engine="python"):
        """
        Extract rho in energy from the introduced file.

        Parameters
        ----------
        T: 'float'
            Temperature in K.
        file: 'str'
            File path.
        header: 'int', optional
            Header of the file. The default is None.
        index_col: 'int', optional
            Index column of the file. The default is None.
        usecols: 'list', optional
            Columns to use. The default is None.
        engine: 'str', optional
            Engine to read the file. The default is "python".

        Returns
        -------
        "Epdos"
            Rho normalize object.
        """
        df = pd.read_csv(file, delim_whitespace=True, header=header,
                         index_col=index_col,
                         usecols=usecols, engine=engine).iloc[::, 0]
        df.index.name = "dE"
        return cls(T, df)

    def from_dE_file(T: float, file: str, header=None, index_col=None,
                     usecols=None, engine="python"):
        """
        Extract rho in energy from the introduced file and create a Tpdos object
        based on the temperature.

        Parameters
        ----------
        T: 'float'
            Temperature in K.
        file: 'str'
            File path.
        header: 'int', optional
            Header of the file. The default is None.
        index_col: 'int', optional
            Index column of the file. The default is None.
        usecols: 'list', optional
            Columns to use. The default is None.
        engine: 'str', optional
            Engine to read the file. The default is "python".

        Returns
        -------
        "Tpdos"
            Rho normalize object.

        Examples
        --------
        Object initialization:
        >>> import os
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("pdos.py", ""))
        >>> os.chdir("../../../data/pdos/")

        >>> T = 300
        >>> file = "interp.300"
        >>> Tpdos.from_dE_file(T, file, usecols=[0, 1], index_col=0).data.iloc[0:5]
        beta
        0.000000    0.000000
        0.015473    0.001083
        0.030945    0.002295
        0.046418    0.005459
        0.061891    0.008876
        Name: rho, dtype: float64

        >>> os.chdir(wd)
        """
        return Epdos.from_file(file, header, index_col, usecols, engine).beta_grid(T)

    @property
    def P(self, threshold=1.0e-6) -> pd.Series:
        """
        Calculate P function for LEAPR formalism with PDOS.
        .. math::
            P(\beta^\prime)=\dfrac{\rho(\beta^\prime)}{2\beta^\prime\sinh(\beta^\prime/2)}

        Parameters
        ----------
        T : 'float'
            Temperature in K.
        threshold : 'float', optional
            Value to chech the initial DOS. The default is 1.e-6.

        Returns
        -------
        "pd.Series"
            P function.

        Raises
        ------
        ValueError
            Initial point of input DOS is not zero.

        Example
        -------
        Object initialization:
        >>> T = 300
        >>> pdos = Tpdos.from_dE(T, rho_in_energy, interv_in_energy)
        >>> pdos.P.iloc[0:6].round(6)
        beta
        0.000000    1.111089
        0.030945    1.111045
        0.061891    1.110912
        0.092836    1.110690
        0.123782    1.109328
        0.154727    1.109309
        Name: P, dtype: float64
        """
        rhoValues, rhoBeta = self.data.values, self.data.index
        if abs(rhoBeta[0]) > threshold:
            raise ValueError("Initial point of input DOS is not zero")
        P_values = np.zeros(len(rhoValues))

        # rho_in_beta is assumed to vary as beta^2 in the nearby of 0
        P_values[0] += rhoValues[1] / rhoBeta[1] ** 2

        # Rest of P values calculation:
        P_values[1:] = 0.5 * rhoValues[1:] / rhoBeta[1:] / np.sinh(0.5 * rhoBeta[1:])
        return pd.Series(P_values, index=rhoBeta, name="P")

    @property
    def Teff(self) -> float:
        """
        Calculate the effective temperature for a certain pdos information.
        .. math::
            T_{eff} = w_t * T + \frac{1}{2k_B}\int_{0}^{\infty}\varepsilon \rho(\varepsilon)\coth\left(\frac{\varepsilon}{2k_BT}\right)d\varepsilon = \left(w_t+\int_{0}^{\infty}\beta^2P(\beta)\cosh(\beta/2)d\beta\right)T

        Parameters
        ----------
        T : 'float'
            Temperature in K.
        twt : 'float', optional
            Translational weight, for solid is zero. The default is 0.

        Returns
        -------
        "float"
            Effective temperature for certain pdos.

        Example
        -------
        Object initialization:
        >>> Tpdos.from_dE(20, rho_in_energy, interv_in_energy).Teff.round(4)
        149.1699
        >>> Tpdos.from_dE(80, rho_in_energy, interv_in_energy).Teff.round(4)
        159.1632
        """
        P = self.P
        P *= self.beta.data ** 2 * np.cosh(0.5 * self.beta.data)
        return integrate(P) * self.T

    @property
    def DebyeWallerCoeff(self) -> float:
        """
        Calculate Debye Waller Coefficient in LEAPR formalism for a certain
        pdos information
        .. math::
            \lambda_s &=\int_{-\infty}^{\infty}P_s(\beta)\exp(-\dfrac{\beta}{2})d\beta \\
                      &=\int_{-\infty}^{0}P_s(\beta)\exp(-\dfrac{\beta}{2})d\beta+\int_{0}^{\infty}P_s(\beta)\exp(-\dfrac{\beta}{2})d\beta \\
                      &=-\int_{\infty}^{0}P_s(-\beta^\prime)\exp(\dfrac{\beta^\prime} {2})d\beta^\prime+\int_{0}^{\infty}P_s(\beta)\exp(-\dfrac{\beta}{2})d\beta \quad\textrm{($\beta^\prime=-\beta$)} \\
                      &=\int_{0}^{\infty}P_s(\beta)\exp(\dfrac{\beta}{2})d\beta+\int_{0}^{\infty}P_s(\beta)\exp(-\dfrac{\beta}{2})d\beta \\
                      &=\int_{0}^{\infty}P_s(\beta)\left(\exp(\dfrac{\beta}{2})+\exp(-\dfrac{\beta}{2})\right)d\beta \\
                      &=2\int_{0}^{\infty}P_s(\beta)\cosh(\dfrac{\beta}{2})d\beta \\
                      &=2\int_{0}^{\beta_{\textrm{max}}}P_s(\beta)\cosh(\dfrac{\beta}{2})d\beta

        Parameters
        ----------
        T : 'float'
            Temperature in K.

        Returns
        -------
        "float"
            Debye Waller Coefficient.

        Examples
        --------
        Object initialization:
        >>> Tpdos.from_dE(20, rho_in_energy, interv_in_energy).DebyeWallerCoeff.round(6)
        0.077454

        >>> Tpdos.from_dE(80, rho_in_energy, interv_in_energy).DebyeWallerCoeff.round(6)
        0.379937
        """
        P = self.P
        P *= 2 * np.cosh(0.5 * self.beta.data)
        return integrate(P)

    @property
    def tau1(self) -> pd.Series:
        """
        Get the Tau(-beta) function for 1 phonon expansion in LEAPR formalism.
        .. math::
            \mathcal{T}_1(-\beta) = \exp(\beta)\mathcal{T}_1(\beta) =\dfrac{P(\beta)\exp(\beta/2)}{\lambda}

        Parameters
        ----------
        T : 'float'
            Temperature in K.

        Returns
        -------
        "pd.Series"
            Tau(-beta) function for the 1 phonon.

        Raises
        ------
        ValueError
            Tau function doesnt satisfy normalization condition.

        Examples
        --------
        Object initialization:
        >>> Tpdos.from_dE(20, rho_in_energy, interv_in_energy).tau1.iloc[:10]
        beta
        0.000000    0.004250
        0.464181    0.005313
        0.928361    0.006524
        1.392542    0.007875
        1.856723    0.009344
        2.320904    0.010932
        2.785084    0.012606
        3.249265    0.014359
        3.713446    0.016167
        4.177627    0.018020
        Name: 1, dtype: float64
        """
        P, beta = self.P, self.beta.data
        tau1 = P * np.exp(0.5 * beta) / self.DebyeWallerCoeff
        if integrate(tau1 * (1 + np.exp(-beta))) < 1.e-5:
            raise ValueError("Tau function for 1 phonon expansion doesnt satisfy the normalization condition")
        tau1.name = 1
        tau1.index.name = "beta"
        return tau1

    def tauN(self, nphonon: int, threshold: float, check: bool = True,
             values: bool = False) -> [np.ndarray, pd.DataFrame]:
        """
        Get the Tau(-beta) function for n phonon expansion in LEAPR formalism
        for a certain temperature.

        Parameters
        ----------
        nphonon: 'int'
            Number of phonons.
        threshold: 'float'
            Threshold to check the tauN normalization.
        check: 'bool', optional
            Check the normalization of the tauN functions. The default is True.
        values: 'bool', optional
            Return the tauN values. The default is False.

        Returns
        -------
        "pd.DataFrame", (len(rho) * nphonon, nphonon)
            Tau(-beta) function for n phonon.

        Examples
        --------
        Object initialization:
        >>> T = 800
        >>> p = Tpdos.from_dE(T, rho_in_energy, interv_in_energy)
        >>> threshold = 0.0
        >>> tauN = p.tauN(5, threshold)
        >>> tauN.iloc[::, :100:20].round(6)
           0.000000  0.232090  0.464181  0.696271  0.928361
        1  0.862582  1.322890  0.341423  0.000000  0.000000
        2  1.068786  0.835423  0.650492  0.397400  0.067640
        3  0.721827  0.778009  0.645243  0.431710  0.257169
        4  0.649349  0.669368  0.608380  0.476611  0.305529
        5  0.572522  0.608795  0.572271  0.475181  0.348585
        """
        tau1Values, beta = self.tau1.values, self.beta.data
        tauN = get_tauNfunc(tau1Values, beta, nphonon, threshold)
        if values:
            return tauN
        else:
            tauN = pd.DataFrame(tauN, columns=get_tauNbeta(beta, tauN.shape[1]))
            tauN.index += 1
            if check:
                # tau1 is not included in the check:
                integrals_value = tauN.apply(integrate, axis=1).iloc[1::]
                if (integrals_value < 1.e-5).any():
                    raise ValueError(
                        "Tau function doesnt satisfy the normalization condition")
            return tauN

    @property
    def dE_grid(self):
        """
        Transform from a specific temperature phonon spectrum to a general
        phonon spectrum in energy.

        Returns
        -------
        "Epdos"
            Rho in energy.

        Examples
        --------
        Object initialization:
        >>> T = 300
        >>> p = Tpdos.from_dE(T, rho_in_energy, interv_in_energy)
        >>> p.data.iloc[0:5]
        beta
        0.000000    0.000000
        0.030945    0.001064
        0.061891    0.004256
        0.092836    0.009576
        0.123782    0.017008
        Name: rho, dtype: float64

        Test the results:
        >>> p.dE_grid.data.iloc[0:5]
        dE
        0.0000    0.000000
        0.0008    0.041157
        0.0016    0.164629
        0.0024    0.370415
        0.0032    0.657892
        Name: rho, dtype: float64
        """
        dE_index = pd.Index(self.beta.get_dE(self.T), name="dE")
        return Epdos(self.data.values.copy(), index=dE_index)


class Epdos:
    """
    Object containing the method and properties of the phonon density of states
    in energy.

    Properties
    ----------
    rho : 'pd.Series'
        Pandas Series containing the rho values in energy (index).

    Methods
    -------
    from_dE: Pdos
        Create a pdos object from a rho in energy
    beta_grid: Pdos
        Change the energy grid of rho for a beta grid
    plot: None
        Plot the pdos data
    P: pd.Series
        Calculate P function for LEAPR formalism with PDOS
    Teff: float
        Calculate the effective temperature
    DebyeWaller: float
        Calculate the Debye-Waller factor for LEAPR formalism with PDOS
    get_tau: pd.DataFrame
        Calculate the tauN functions
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize of the pdos object.

        Parameters
        ----------
        *args : variables
            variables for the creation of the pandas Series.
        **kwargs : 'dict'
            Dictionary to create the pandas series of rho.

        Returns
        -------
        'Pdos'
            Object containing the method and properties of rho in energy.

        """
        self.data = pd.Series(*args, **kwargs)

    @property
    def data(self) -> pd.Series:
        """Pandas Series containing the rho values in energy (index)."""
        return self._data

    @data.setter
    def data(self, rhoData: Union[pd.Series, Iterable]) -> pd.Series:
        """
        Data setter for rho to ensure the following properties of the data:
            - Shape of the data: 1 dimension
            - Energy index monotoally increasing
            - Rho values normalization

        Parameters
        ----------
        rho : pd.Series
            rho values in energy.

        Returns
        -------
        "pd.Series"
            Rho normalize.

        Raises
        ------
        TypeError
            Rho is not 1 dimension pd.Series
        SyntaxError
            Energy grid is not monotonically increasing.

        Examples
        --------
        Object initialization:
        >>> p = Epdos.from_dE(rho_in_energy, interv_in_energy)

        Test the results:
        >>> assert integrate(p.data) == 1.0
        """
        data_ = pd.Series(rhoData, name="rho")

        if data_.index.name != "dE" and data_.index.name != "E":
            raise SyntaxError("Energy index must be dE")

        if not len(data_.shape) == 1:
            raise TypeError("Rho must have one dimension")

        if not data_.index.is_monotonic_increasing:
            raise SyntaxError("dE grid is not monotonically increasing")

        self._data = data_ / integrate(data_)

    @classmethod
    def from_dE(cls, rho: Iterable, intervalE: float):
        """
        Extract rho in energy from the introduced data.

        Parameters
        ----------
        rho : 1D iterable
            rho values.
        intervalE : 'float'
            Energy interval in eV.

        Returns
        -------
        "Pdos"
            Rho normalize object.

        Examples
        --------
        Object initialization:
        >>> p = Epdos.from_dE(rho_in_energy, interv_in_energy)

        Test the results:
        >>> p.data.iloc[0:10]
        dE
        0.0000    0.000000
        0.0008    0.041157
        0.0016    0.164629
        0.0024    0.370415
        0.0032    0.657892
        0.0040    1.028308
        0.0048    1.480414
        0.0056    2.015458
        0.0064    2.632193
        0.0072    3.331243
        Name: rho, dtype: float64
        """
        rho_ = np.array(rho)
        index = pd.Index(np.arange(len(rho_)) * intervalE, name="dE")
        return cls(rho_, index=index)

    @classmethod
    def from_file(cls, file: str, header=None, index_col=None,
                  usecols=None, engine="python"):
        """
        Extract rho in energy from the introduced file.

        Parameters
        ----------
        file: 'str'
            File path.
        header: 'int', optional
            Header of the file. The default is None.
        index_col: 'int', optional
            Index column of the file. The default is None.
        usecols: 'list', optional
            Columns to use. The default is None.
        engine: 'str', optional
            Engine to read the file. The default is "python".

        Returns
        -------
        "Epdos"
            Rho normalize object.

        Examples
        --------
        Object initialization:
        >>> import os
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("pdos.py", ""))
        >>> os.chdir("../../../data/pdos/")
        >>> file = "interp.300"
        >>> Epdos.from_file(file, usecols=[0, 1], index_col=0).data.iloc[0:5]
        dE
        0.0000    0.000000
        0.0004    0.041879
        0.0008    0.088790
        0.0012    0.211161
        0.0016    0.343320
        Name: rho, dtype: float64
        >>> os.chdir(wd)
        """
        df = pd.read_csv(file, delim_whitespace=True, header=header,
                         index_col=index_col,
                         usecols=usecols, engine=engine).iloc[::, 0]
        df.index.name = "dE"
        return cls(df)

    def beta_grid(self, T: float):
        """
        Change the energy grid of rho. Two options available:
            - Tranform energy grid in beta grid by introducing T

        Parameters
        ----------
        T : 'float'
            Temperature to change energy grid to beta grid. The default is
            None.

        Returns
        -------
        "Tpdos"
            pdos object with the new grid.

        Examples
        --------
        Object initialization:
        >>> p = Epdos.from_dE(rho_in_energy, interv_in_energy)
        >>> p.data.iloc[0:5]
        dE
        0.0000    0.000000
        0.0008    0.041157
        0.0016    0.164629
        0.0024    0.370415
        0.0032    0.657892
        Name: rho, dtype: float64

        Test the results:
        >>> T = 300
        >>> p.beta_grid(T).data.iloc[0:5]
        beta
        0.000000    0.000000
        0.030945    0.001064
        0.061891    0.004256
        0.092836    0.009576
        0.123782    0.017008
        Name: rho, dtype: float64
        """
        rho = self.data.copy()
        rho.index = Beta.from_dE(rho.index.values, T).to_index
        return Tpdos(T, rho)

    def Teff(self, T: float) -> float:
        """
        Calculate the effective temperature for a certain pdos information.

        Parameters
        ----------
        T : 'float'
            Temperature in K.

        Returns
        -------
        "float"
            Effective temperature for certain pdos.

        Examples
        --------
        Object initialization:
        >>> T = 20
        >>> pdos = Epdos.from_dE(rho_in_energy, interv_in_energy)
        >>> pdos.Teff(T).round(4)
        149.1699
        """
        return self.beta_grid(T).Teff

    def DebyeWallerCoeff(self, T: float) -> float:
        """
        Calculate Debye Waller Coefficient in LEAPR formalism for a certain
        pdos information.

        Parameters
        ----------
        T : 'float'
            Temperature in K.

        Returns
        -------
        "float"
            Debye Waller Coefficient.

        Examples
        --------
        Object initialization:
        >>> Epdos.from_dE(rho_in_energy, interv_in_energy).DebyeWallerCoeff(20).round(6)
        0.077454
        """
        return self.beta_grid(T).DebyeWallerCoeff

    def tau1(self, T: float) -> pd.Series:
        """
        Get the Tau(-beta) function for 1 phonon expansion in LEAPR formalism.

        Parameters
        ----------
        T : 'float'
            Temperature in K.

        Returns
        -------
        "pd.Series"
            Tau(-beta) function for the 1 phonon.

        Examples
        --------
        Object initialization:
        >>> Epdos.from_dE(rho_in_energy, interv_in_energy).tau1(20).iloc[:10]
        beta
        0.000000    0.004250
        0.464181    0.005313
        0.928361    0.006524
        1.392542    0.007875
        1.856723    0.009344
        2.320904    0.010932
        2.785084    0.012606
        3.249265    0.014359
        3.713446    0.016167
        4.177627    0.018020
        Name: 1, dtype: float64
        """
        return self.beta_grid(T).tau1

    def tauN(self, T: float, nphonon: int, threshold: float, check: bool = True,
              values: bool = False) -> [np.ndarray, pd.DataFrame]:
        """
        Get the Tau(-beta) function for n phonon expansion in LEAPR formalism
        for a certain temperature.

        Parameters
        ----------
        T : 'float'
            Temperature in K.
        nphonon : 'int'
            Number of phonons.
        threshold : 'float'
            Threshold to check the tauN normalization.
        check : 'bool', optional
            Check the normalization of the tauN functions. The default is True.
        values : 'bool', optional
            Return the tauN values. The default is False.

        Returns
        -------
        "pd.DataFrame", (len(rho) * nphonon, nphonon)
            Tau(-beta) function for n phonon.

        Examples
        --------
        Object initialization:
        >>> T = 800
        >>> p = Epdos.from_dE(rho_in_energy, interv_in_energy)
        >>> threshold = 0.0
        >>> tauN = p.tauN(T, 5, threshold)
        >>> tauN.iloc[::, :100:20].round(6)
           0.000000  0.232090  0.464181  0.696271  0.928361
        1  0.862582  1.322890  0.341423  0.000000  0.000000
        2  1.068786  0.835423  0.650492  0.397400  0.067640
        3  0.721827  0.778009  0.645243  0.431710  0.257169
        4  0.649349  0.669368  0.608380  0.476611  0.305529
        5  0.572522  0.608795  0.572271  0.475181  0.348585
        """
        return self.beta_grid(T).tauN(nphonon, threshold, check, values)


class Npdos:
    """
    Object containing the method and properties of N phonon density of states
    for N temperatures.
    """
    def __init__(self, pdos_dict: dict):
        """
        Initialize of the pdos object containing the N temperatures.

        Parameters
        ----------
        pdos_dict: 'dict' of Tpdos objects
            Dictionary containing the pdos objects for N temperatures.
        """
        self.instance = pdos_dict
        self.interp_spline = None

    @property
    def data(self) -> pd.DataFrame:
        """
        Get the data of the pdos objects in a DataFrame format with the index
        being the energy grid.

        Returns
        -------
        "pd.DataFrame"
            Data of the pdos objects in a DataFrame format.
        """
        data = pd.DataFrame({key: pdos.dE_grid.data
                             for key, pdos in self.instance.items()})
        if np.any(data.isna()):
            data.interpolate(method="linear", axis=0).fillna(0, inplace=True)
        return data

    @classmethod
    def from_file(cls, T: [float, list], file: [str, list], header=None,
                  index_col=None, usecols=None, engine="python", grid="dE"):
        """
        Extract rho in energy from the introduced file and create a Tpdos object
        for each temperature.

        Parameters
        ----------
        file: 'str' or 'list'
            File path or list of file paths.
        T: 'float' or 'list'
            Temperature in K or list of temperatures.
        header: 'int', optional
            Header of the file. The default is None.
        index_col: 'int', optional
            Index column of the file. The default is None.
        usecols: 'list', optional
            Columns to use. The default is None.
        engine: 'str', optional
            Engine to read the file. The default is "python".
        grid: 'str', optional
            Grid of the file. The default is "dE". Options are "dE" or "beta".

        Returns
        -------
        "Npdos"
            Rho normalize object.

        Examples
        --------
        Object initialization:
        >>> import os
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("pdos.py", ""))
        >>> os.chdir("../../../data/pdos/")
        >>> file = "interp.300"
        >>> T = 300
        >>> Npdos.from_file(T, file, usecols=[0, 1], index_col=0).data.iloc[0:5]
                     300
        dE
        0.0000  0.000000
        0.0004  0.041879
        0.0008  0.088790
        0.0012  0.211161
        0.0016  0.343320
        >>> os.chdir(wd)
        """
        # Transform file and T to list if they are not
        file_ = [file] if isinstance(file, str) else file
        T_ = [T] if isinstance(T, (int, float)) else T
        if len(file_) != len(T_):
            raise ValueError("The number of files and temperatures must be the same")

        # Prepare arguments for from_file function
        args = (header, index_col, usecols, engine)

        # Create Tpdos objects based on the grid type
        method = Tpdos.from_dE_file if grid == "dE" else Tpdos.from_file
        return cls({T: method(T, file, *args) for file, T in zip(file_, T_)})

    @classmethod
    def from_dE(cls, T: list, rho: [list, np.ndarray], intervalE: list):
        """
        Create a Npdos object from a list of Tpdos objects.

        Parameters
        ----------
        T: list of 'float'
            List of temperatures in K.
        rho: list of '1D iterable'
            rho function values store in each row.
        intervalE: list of 'float'
            List of energy interval in eV.

        Returns
        -------
        "Npdos"
            Rho normalize object.
        """
        return cls({T[i]: Tpdos.from_dE(T[i], rho[i], intervalE[i])
                    for i in range(len(T))})

    def compute_spline(self):
        """
        Compute the spline interpolation of the pdos data if it is not computed
        """
        data = self.data.copy()
        dE, T = data.index.values, data.columns.values
        ky = 3 if len(T) > 3 else len(T) - 1
        self.interp_spline = RectBivariateSpline(dE, T, data.values, ky=ky)

    def Tinterp(self, Tnew: [float, np.ndarray],
                inplace: bool = False) -> [None, dict, Tpdos]:
        """
        Interpolate the pdos data to a new temperature grid. If the new
        temperature is out of the bound of the original temperature grid, the
        previous or next temperature will be used.

        Parameters
        ----------
        Tnew: 'float' or 'np.ndarray'
            New temperature grid.

        Returns
        -------
        "pd.DataFrame"
            Interpolated pdos data.
        """
        Tnew_ = Tnew if hasattr(Tnew, "__len__") else [Tnew]
        if self.interp_spline is None:
            self.compute_spline()
        # Get the index values from the data
        dE = self.data.index.values

        # Use the previously computed spline function to interpolate the data
        interpolated_T = self.interp_spline(dE, Tnew_)

        # Compute the Tpdos object for the Tnew temperature
        intepolated_Tpdos = {T: Epdos(interpolated_T[:, i]).beta_grid(T)
                             for i, T in enumerate(Tnew_)}
        if inplace:
            self.instance.update(intepolated_Tpdos)
        else:
            return intepolated_Tpdos[Tnew] if len(Tnew_) == 1 else intepolated_Tpdos

    def tauN(self, T: float, nphonon: int, threshold: float, check: bool = True,
              values: bool = False) -> [np.ndarray, pd.DataFrame]:
        """
        Get the Tau(-beta) function for n phonon expansion in LEAPR formalism
        for a certain temperature. The tauN function is computed by interpolating
        the pdos data to the temperature T and then computing the tauN function.

        Parameters
        ----------
        T: 'float'
            Temperature in K.
        nphonon: 'int'
            Number of phonons.
        threshold: 'float'
            Threshold to check the tauN normalization.
        check: 'bool', optional
            Check the normalization of the tauN functions. The default is True.
        values: 'bool', optional
            Return the tauN values. The default is False.

        Returns
        -------
        "pd.DataFrame", (len(rho) * nphonon, nphonon)
            Tau(-beta) function for n phonon.
        """
        return self.Tinterp(T).tauN(nphonon, threshold, check, values)

    def __getattr__(self, name):
        if name in ["Teff", "DebyeWallerCoeff", "P", "tau1"]:
            results = {key: getattr(pdos, name)()
                       for key, pdos in self.instance.items()}
            return pd.Series(results, name=name)
        else:
            raise AttributeError(f"Attribute {name} not found in the Npdos object")


class Pdos:
    def __init__(self, *args, **kwargs):
        if isinstance(args[0], (Epdos, Tpdos, Npdos)):
            self.instance = args[0]
        elif isinstance(args[0], (int, float)):
            self.instance = Tpdos(*args, **kwargs)
        elif len(args[-1].shape) == 1:
            self.instance = Epdos(*args, **kwargs)
        else:
            self.instance = Npdos(*args, **kwargs)

    @classmethod
    def from_dE(cls, *args, **kwargs):
        if isinstance(args[0], (int, float)):
            return cls(Tpdos.from_dE(*args, **kwargs))
        elif isinstance(args[-1], (list, np.ndarray)):
            return cls(Npdos.from_dE(*args, **kwargs))
        else:
            return cls(Epdos.from_dE(*args, **kwargs))

    def __getattr__(self, name):
        # assume it is implemented by self.instance
        return getattr(self.instance, name)

