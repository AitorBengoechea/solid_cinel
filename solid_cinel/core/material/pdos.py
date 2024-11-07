"""
Python file for working with Phonon Density Of States.

@author: AB272525
"""
from solid_cinel.core.generic import integrate
from solid_cinel.core.dynamic_structure.beta import Beta
from solid_cinel.core.material.tau import get_tauNfunc, get_tauNbeta
from scipy.interpolate import RectBivariateSpline
import pandas as pd
import numpy as np
import os
import re
from typing import Iterable, Union


# Examples variables:
from solid_cinel.data.examples.Al27 import rho_in_energy, interv_in_energy


class Tpdos:
    """
    Object containing the method and properties of the phonon density of states
    for a certain temperature.
    """
    def __init__(self, T: float, *args, **kwargs):
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
        # Define the data style in the pd.Series
        data_ = pd.Series(rhoData, name="rho")
        data_.index.name = "beta"

        # Check the data properties
        if not len(data_.shape) == 1:
            raise TypeError("Rho must have one dimension")

        if not data_.index.is_monotonic_increasing:
            raise SyntaxError("beta grid is not monotonically increasing")

        # Normalize the data
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
        df = pd.read_csv(file, sep='\s+', header=header,
                         index_col=index_col,
                         usecols=usecols, engine=engine).iloc[::, 0]
        df.index.name = "beta"
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
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("pdos.py", ""))
        >>> file = "../../data/pdos/interp.300"
        >>> T = 300
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
        return Epdos.from_file(file, header, index_col, usecols, engine).get_Tpdos(T)

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

        # Check the initial point of the input DOS:
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
        >>> float(Tpdos.from_dE(20, rho_in_energy, interv_in_energy).Teff.round(4))
        149.1699
        >>> float(Tpdos.from_dE(80, rho_in_energy, interv_in_energy).Teff.round(4))
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
        >>> float(Tpdos.from_dE(20, rho_in_energy, interv_in_energy).DebyeWallerCoeff.round(6))
        0.077454

        >>> float(Tpdos.from_dE(80, rho_in_energy, interv_in_energy).DebyeWallerCoeff.round(6))
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
        # Get the P function data:
        P, beta = self.P, self.beta.data

        # Calculate the tau1 function:
        tau1 = P * np.exp(0.5 * beta) / self.DebyeWallerCoeff

        # Check the normalization condition:
        if integrate(tau1 * (1 + np.exp(-beta))) < 1.e-5:
            raise ValueError("Tau function for 1 phonon expansion doesnt satisfy the normalization condition")

        # Define the style of the tau1 function:
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
        # Get the tau1 function values:
        tau1Values, beta = self.tau1.values, self.beta.data

        # Calculate the tauN function:
        tauN = get_tauNfunc(tau1Values, beta, nphonon, threshold)


        if values:
            # Return the values in a np.array
            return tauN
        else:
            # Define the style of the tauN function dataframe:
            tauN = pd.DataFrame(tauN, columns=get_tauNbeta(beta, tauN.shape[1]))
            tauN.index += 1

            # Check the normalization condition:
            if check:
                # tau1 is not included in the check:
                integrals_value = tauN.apply(integrate, axis=1).iloc[1::]
                if (integrals_value < 1.e-5).any():
                    raise ValueError(
                        "Tau function doesnt satisfy the normalization condition")
            return tauN

    @property
    def to_Epdos(self):
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
        >>> p.to_Epdos.data.iloc[0:5]
        dE
        0.0000    0.000000
        0.0008    0.041157
        0.0016    0.164629
        0.0024    0.370415
        0.0032    0.657892
        Name: rho, dtype: float64
        """
        dE_indexValues = self.beta.get_dE(self.T).round(20)
        return Epdos(self.data.values.copy(), index=pd.Index(dE_indexValues.round(10), name="dE"))


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
        # Define the data style in the pd.Series
        data_ = pd.Series(rhoData, name="rho")
        data_.index.name = "dE"

        # Check the data properties
        if not len(data_.shape) == 1:
            raise TypeError("Rho must have one dimension")

        if not data_.index.is_monotonic_increasing:
            raise SyntaxError("dE grid is not monotonically increasing")

        # Normalize the data
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
    def from_file(cls, file: str, header: [int, list] = None,
                  index_col: [int, list] = 0, usecols: [int, list] = [0, 1],
                  engine: str = "python"):
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
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("pdos.py", ""))
        >>> os.chdir("../../data/pdos/")
        >>> file = "interp.300"
        >>> Epdos.from_file(file).data.iloc[0:5]
        dE
        0.0000    0.000000
        0.0004    0.041879
        0.0008    0.088790
        0.0012    0.211161
        0.0016    0.343320
        Name: rho, dtype: float64
        >>> os.chdir(wd)
        """
        df = pd.read_csv(file, sep='\s+', header=header,
                         index_col=index_col,
                         usecols=usecols, engine=engine).iloc[::, 0]
        df.index.name = "dE"
        return cls(df)

    def get_Tpdos(self, T: float) -> Tpdos:
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
        >>> p.get_Tpdos(T).data.iloc[0:5]
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
        >>> float(pdos.Teff(T).round(4))
        149.1699
        """
        return self.get_Tpdos(T).Teff

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
        >>> float(Epdos.from_dE(rho_in_energy, interv_in_energy).DebyeWallerCoeff(20).round(6))
        0.077454
        """
        return self.get_Tpdos(T).DebyeWallerCoeff

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
        return self.get_Tpdos(T).tau1

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
        return self.get_Tpdos(T).tauN(nphonon, threshold, check, values)


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
        # Get the data of the pdos objects:
        data = pd.DataFrame({key: pdos.to_Epdos.data
                             for key, pdos in self.instance.items()})

        # Define the style of the dataframe:
        data.columns.name = "T"
        data.index.name = "dE"
        data.sort_index(axis=0, inplace=True)
        data.sort_index(axis=1, inplace=True)

        # Interpolate the data if there are NaN values in the data
        if np.any(data.isna()):
            data.interpolate(method="linear", axis=0, inplace=True)
            # Fill the limit values with 0
            data.fillna(0, inplace=True)

        return data

    @staticmethod
    def check_list(temperatures: Union[float, Iterable[float]]):
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
        temperatures: Union[float, Iterable[float]]
            The temperature in K

        Returns
        -------
        pd.Index
            The new temperatures to calculate

        Examples
        --------
        Object initialization:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("pdos.py", ""))
        >>> folder = "../../data/pdos"
        >>> npdos = Npdos.from_directory(folder, usecols=[0, 1], index_col=0)
        >>> assert npdos.get_Tnew([300]).empty == True
        >>> assert npdos.get_Tnew(200) == pd.Index([200])
        >>> assert (npdos.get_Tnew([600, 200]) == pd.Index([200, 600])).all()
        """
        Tnew = pd.Index(self.check_list(temperatures))
        return Tnew.difference(self.data.columns).sort_values()

    @classmethod
    def from_file(cls, T: [float, list], file: [str, list],
                  header: [int, list] = None, index_col: [int, list] = 0,
                  usecols: [int, list] = [0, 1], engine: str = "python",
                  grid: str = "dE"):
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
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("pdos.py", ""))
        >>> file = "../../data/pdos/interp.300"
        >>> T = 300
        >>> Npdos.from_file(T, file).data.iloc[0:5]
        T            300
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
    def from_directory(cls, pathToDirectory: str, **kwargs):
        """
        Create a Npdos object from a directory containing the pdos files. The
        files must have the temperature in the name of the file.

        Parameters
        ----------
        pathToDirectory : 'str'
            Path to the directory containing the pdos files.
        kwargs :  'dict'
            Arguments to pass to the from_file function.

        Returns
        -------
        "Npdos"
            Rho normalize object.

        Examples
        --------
        Object initialization:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("pdos.py", ""))
        >>> folder = "../../data/pdos"
        >>> npdos = Npdos.from_directory(folder, usecols=[0, 1], index_col=0).data
        >>> npdos.iloc[0:5]
        T         300.0     500.0     1000.0    1700.0
        dE
        0.0000  0.000000  0.000000  0.000000  0.000000
        0.0004  0.041879  0.047919  0.055963  0.070446
        0.0008  0.088790  0.117714  0.141224  0.170520
        0.0012  0.211161  0.239444  0.278530  0.354508
        0.0016  0.343320  0.397844  0.462783  0.604724
        >>> os.chdir(wd)
        """
        # Get the files in the directory:
        files = os.listdir(pathToDirectory)
        if pathToDirectory[-1] != "/":
            pathToDirectory = "".join([pathToDirectory, "/"])

        # Create a dictionary with the temperature and the file path:
        file_dict = {}
        for file in files:
            # Get the floats and the int:
            match = re.search(r'\d+(\.\d+)?', file)
            if match:
                file_dict[float(match.group())] = "".join([pathToDirectory, file])

        return cls.from_file(file_dict.keys(), file_dict.values(), **kwargs)

    @classmethod
    def from_dE(cls, T: [float, list], rho: np.ndarray, intervalE: [float, list]):
        """
        Create a Npdos object from a list of Tpdos objects.

        Parameters
        ----------
        T: list of 'float'
            List of temperatures in K.
        rho: np.ndarray
            rho function values store in each row.
        intervalE: list of 'float'
            List of energy interval in eV.

        Returns
        -------
        "Npdos"
            Rho normalize object.

        Examples
        --------
        Object initialization:
        >>> T = [300, 500]
        >>> p = Npdos.from_dE(T, rho_in_energy, interv_in_energy)
        >>> p.instance[300].data.iloc[0:5]
        beta
        0.000000    0.000000
        0.030945    0.001064
        0.061891    0.004256
        0.092836    0.009576
        0.123782    0.017008
        Name: rho, dtype: float64
        >>> p.instance[500].data.iloc[0:5]
        beta
        0.000000    0.000000
        0.018567    0.001773
        0.037134    0.007093
        0.055702    0.015960
        0.074269    0.028346
        Name: rho, dtype: float64
        """
        # Check the temperature input:
        T_ = np.array(cls.check_list(T))

        # Dont use zero temperature:
        if 0 in T_:
            raise ValueError("Zero temperature is not allowed")
        else:
            Ntemp = len(T_)

        # Check the energy interval input
        intervalE_ = np.array(cls.check_list(intervalE))
        if len(intervalE_) == 1:
            intervalE_ = np.repeat(intervalE_, Ntemp)

        # Reshape the rho values:
        rho_ = np.repeat(rho[::, np.newaxis], Ntemp, axis=1)

        # Create the Npdos object:
        return cls({T_[i]: Tpdos.from_dE(T_[i], rho_[::, i], intervalE_[i]) for i in range(Ntemp)})


    def compute_spline(self):
        """
        Compute the spline interpolation of the pdos data if it is not computed
        """
        data_ = self.data.copy()
        T = data_.columns
        ky = 3 if len(T) > 3 else len(T) - 1
        self.interp_spline = RectBivariateSpline(data_.index, T, data_.values,
                                                 ky=ky)

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

        Examples
        --------
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("pdos.py", ""))
        >>> folder = "../../data/pdos"
        >>> npdos = Npdos.from_directory(folder, usecols=[0, 1], index_col=0)
        >>> npdos.Tinterp(600).data.iloc[0:5]
        beta
        0.000000    0.000000
        0.007736    0.002592
        0.015473    0.006558
        0.023209    0.012920
        0.030945    0.021546
        Name: rho, dtype: float64
        >>> npdos.Tinterp([600, 200], inplace=True)
        >>> npdos.data.iloc[0:5]
        T         200.0     300.0     500.0     600.0     1000.0    1700.0
        dE
        0.0000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
        0.0004  0.041879  0.041879  0.047919  0.050128  0.055963  0.070446
        0.0008  0.088790  0.088790  0.117714  0.126835  0.141224  0.170520
        0.0012  0.211161  0.211161  0.239444  0.249884  0.278530  0.354508
        0.0016  0.343320  0.343320  0.397844  0.416720  0.462783  0.604724

        >>> os.chdir(wd)
        """
        # Check the input values to get the new T values
        Tnew_ = self.get_Tnew(Tnew)

        # If there are no new temperatures to calculate, return the data
        if Tnew_.empty:
            Textract = self.check_list(Tnew)
            pdosExtract = {T: self.instance[T] for T in Textract}
            return pdosExtract[Tnew] if len(Textract) == 1 else pdosExtract

        # Compute the spline interpolation if it is not computed
        if self.interp_spline is None:
            self.compute_spline()

        # Get the index values from the data
        dE = self.data.index.values

        # Use the previously computed spline function to interpolate the data
        interpolated_T = self.interp_spline(dE, Tnew_)
        interpolated_T[0, ::] = 0.0

        # Compute the Tpdos object for the Tnew temperature
        intepolated_Tpdos = {T: Epdos(interpolated_T[:, i], index=dE).get_Tpdos(T)
                             for i, T in enumerate(Tnew_)}

        # Update the data if inplace is True
        if inplace:
            self.instance.update(intepolated_Tpdos)
        else:
            return intepolated_Tpdos[Tnew] if len(Tnew_) == 1 else intepolated_Tpdos

    def get_Tpdos(self, T: float) -> Tpdos:
        """
        Get the Tpdos object for a certain temperature.

        Parameters
        ----------
        T: 'float'
            Temperature in K.

        Returns
        -------
        "Tpdos"
            Tpdos object for the temperature T.

        Examples
        --------
        Object initialization:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("pdos.py", ""))
        >>> folder = "../../data/pdos"
        >>> npdos = Npdos.from_directory(folder, usecols=[0, 1], index_col=0)
        >>> npdos.get_Tpdos(300).data.iloc[0:5]
        beta
        0.000000    0.000000
        0.015473    0.001083
        0.030945    0.002295
        0.046418    0.005459
        0.061891    0.008876
        Name: rho, dtype: float64

        >>> npdos.get_Tpdos(500).data.iloc[0:5]
        beta
        0.000000    0.000000
        0.009284    0.002065
        0.018567    0.005072
        0.027851    0.010317
        0.037134    0.017142
        Name: rho, dtype: float64
        >>> os.chdir(wd)
        """
        # Interpolate the data to the temperature T for avoiding the
        # numerical errors
        return self.Tinterp(T)

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
        if len(args) == 0:
            raise TypeError("No arguments provided")
        elif isinstance(args[0], (int, float)):
            return cls(getattr(Tpdos, 'from_dE')(*args, **kwargs))
        elif len(args) == 3:
            return cls(getattr(Npdos, 'from_dE')(*args, **kwargs))
        else:
            return cls(getattr(Epdos, 'from_dE')(*args, **kwargs))

    @classmethod
    def from_file(cls, *args, **kwargs):
        if len(args) == 0:
            raise TypeError("No arguments provided")
        elif len(args) == 1:
            return cls(getattr(Epdos, 'from_file')(*args, **kwargs))
        elif isinstance(args[0], (int, float)):
            return cls(getattr(Tpdos, 'from_file')(*args, **kwargs))
        else:
            return cls(getattr(Npdos, 'from_file')(*args, **kwargs))

    @classmethod
    def from_directory(cls, *args, **kwargs):
        """
        Create a Pdos object from a directory containing the pdos files. The
        files must have the temperature in the name of the file.

        Parameters
        ----------
        pathToDirectory : 'str'
            Path to the directory containing the pdos files.
        kwargs :  'dict'
            Arguments to pass to the Npdos.from_file function.

        Returns
        -------
        Object initialization:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("pdos.py", ""))
        >>> folder = "../../data/pdos"
        >>> npdos = Pdos.from_directory(folder, usecols=[0, 1], index_col=0).data
        >>> npdos.iloc[0:5]
        T         300.0     500.0     1000.0    1700.0
        dE
        0.0000  0.000000  0.000000  0.000000  0.000000
        0.0004  0.041879  0.047919  0.055963  0.070446
        0.0008  0.088790  0.117714  0.141224  0.170520
        0.0012  0.211161  0.239444  0.278530  0.354508
        0.0016  0.343320  0.397844  0.462783  0.604724
        >>> os.chdir(wd)
        """
        return cls(Npdos.from_directory(*args, **kwargs))

    @property
    def type(self) -> str:
        """
        Get the type of the pdos object instace (Epdos, Tpdos, Npdos).

        Returns
        -------
        "str"
            Type of the pdos object.

        Examples
        --------
        Object initialization:
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("pdos.py", ""))
        >>> folder = "../../data/pdos"
        >>> pdos = Pdos.from_directory(folder, usecols=[0, 1], index_col=0)
        >>> pdos.type
        'Npdos'
        """
        return type(self.instance).__name__

    def fix_T(self, T: float) -> Tpdos:
        """
        Check if the Pdos object is fixed for 1 temperature.

        Parameters
        ----------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        T : 'float' or 'Iterable', optional
            Temperature in K. The default is None.

        Returns
        -------
        Tpdos
            TPdos object.
        """
        if type(self.instance).__name__ == "Tpdos":
            if self.instance.T != T:
                raise ValueError(f"The temperature of the pdos object {self.instance.T} is different from the input temperature {T}")
            return self.instance
        else:
            return self.instance.get_Tpdos(T)

    def __getattr__(self, name):
        if hasattr(self.instance, name):
            return getattr(self.instance, name)
        else:
            raise AttributeError(f"'{type(self.instance).__name__}' object has no attribute '{name}'")


