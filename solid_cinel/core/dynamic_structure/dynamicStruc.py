"""
Python file for working with scattering functions.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
from scipy.constants import physical_constants as const
from solid_cinel.core.generic import integrate, interp_multyParallel
from solid_cinel.core.dynamic_structure.beta import get_AbsBeta, calc_Beta
from solid_cinel.core.dynamic_structure.alpha import get_alphaMatMod, AlphaBase, calc_alpha, calc_alphaRecoil
from solid_cinel.core.dynamic_structure.sab import get_SabSct, phonon_expansion
from solid_cinel.core.material.pdos import Pdos
from solid_cinel.core.material.tau import get_tauNbeta
from typing import Iterable
import warnings
from dataclasses import dataclass

# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]


@dataclass
class DoubleDiff:
    """
    Abstract class for the creation of Double Diff Data.

    Parameters
    ----------
    Ein : float
        The incident energy of the neutron in eV
    M : float
        The mass of the target material in amu
    T : float
        Temperature of the material in K
    Eout : np.ndarray
        The neutron outgoing energy
    mu : np.ndarray
        The cosine of the angle of the distribution in degrees
    """
    Ein: float
    M: float
    T: float
    Eout: np.ndarray
    mu: np.ndarray

    def __post_init__(self):
        self.Eout = np.unique(self.Eout)
        self.mu = np.unique(self.mu)
        self.mu2D = self.mu[::, np.newaxis]

    @property
    def A(self) -> float:
        """
        The mass ratio of the neutron to the target material.

        Returns
        -------
        float
            The mass ratio of the neutron to the target material
        """
        return self.M / m

    @property
    def aws(self) -> float:
        """
        The average atomic weight of the target material.

        Returns
        -------
        float
            The average atomic weight of the target material
        """
        return ((self.A + 1) / self.A) ** 2
    @property
    def beta(self) -> np.ndarray:
        """
        Calculate the beta values.

        Returns
        -------
        np.ndarray
            The beta values
        """
        return calc_Beta(self.Eout, self.Ein, self.T)

    @property
    def betaAbs(self) -> np.ndarray:
        """
        Calculate the absolute beta values.

        Returns
        -------
        np.ndarray
            The absolute beta values
        """
        return np.absolute(self.beta)

    @property
    def downScatIndex(self) -> int:
        """
        Calculate the downscattering index.

        Returns
        -------
        int
            The downscattering index
        """
        return (self.Eout <= self.Ein).sum()

    @property
    def alpha(self) -> np.ndarray:
        """
        Calculate the alpha values.

        Returns
        -------
        np.ndarray
            The alpha values
        """
        return calc_alpha(self.Ein, self.M, self.T, self.Eout, self.mu2D)

    @property
    def recoil(self) -> np.ndarray:
        """
        Calculate the recoil energy.

        Returns
        -------
        np.ndarray
            The recoil energy
        """
        return calc_alphaRecoil(self.Ein, self.M, self.Eout, self.mu2D)


class DoubleDiffData:
    """
    Abstract class for handeling Double Differential data.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the DoubleDiffData class.

        Parameters
        ----------
        args: Iterable
            The values for the pd.DataFrame
        kwargs: dict
            Optional arguments for the construction of the pd.DataFrame

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> dd.data
        Eout  1  2  3
        mu
        1     1  2  3
        2     4  5  6
        """
        self.data = pd.DataFrame(*args, **kwargs)

    @property
    def data(self) -> pd.DataFrame:
        """
        DataFrame

        Returns
        -------
        pd.Series
            Transfer function data
        """
        return self._data

    @data.setter
    def data(self, dd_pdf: Iterable):
        """
        Set the data and check the normalization.

        Parameters
        ----------
        dd_pdf : pd.Series
            Transfer function data

        """
        # Sort and define the style of the dataframe:
        dd_pdf_ = pd.DataFrame(dd_pdf).sort_index(axis=0).sort_index(axis=1)
        dd_pdf_.index.name = "mu"
        dd_pdf_.columns.name = "Eout"

        # Save the data:
        self._data = dd_pdf_

    @property
    def values(self) -> np.ndarray:
        """
        The values of the Double Differential data.

        Returns
        -------
        np.ndarray
            The values of the Double Differential data
        """
        return self.data.values

    @property
    def Eout(self) -> np.ndarray:
        """
        The outgoing energy grid.

        Returns
        -------
        np.ndarray
            The outgoing energy grid
        """
        return self.data.columns.values

    @property
    def mu(self) -> np.ndarray:
        """
        The cosine of the angle of the distribution in degrees.

        Returns
        -------
        np.ndarray
            The cosine of the angle of the distribution in degrees
        """
        return self.data.index.values

    @property
    def shape(self) -> tuple:
        """
        The shape of the Double Differential data.

        Returns
        -------
        tuple
            The shape of the Double Differential data
        """
        return self.data.shape

    @property
    def theta(self) -> np.ndarray:
        """
        The angle of the distribution in degrees.

        Returns
        -------
        np.ndarray
            The angle of the distribution in degrees
        """
        return np.rad2deg(np.arccos(self.mu)).round(6)

    @property
    def rowIntegral(self) -> pd.Series:
        """
        Integral of the Double Differential data by row.

        Returns
        -------
        pd.Series
            Integral of the Double Differential data by row

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> dd.rowIntegral
                mu
        1     4.0
        2    10.0
        dtype: float64
        """
        return self.integrate(axis=1)

    @property
    def columsIntegral(self) -> pd.Series:
        """
        Integral of the Double Differential data by column.

        Returns
        -------
        pd.Series
            Integral of the Double Differential data by column

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> dd.columsIntegral
        Eout
        1    2.5
        2    3.5
        3    4.5
        dtype: float64
        """
        return self.integrate(axis=0)

    @property
    def doubleIntegral(self) -> float:
        """
        Double integral of the Double Differential data.

        Returns
        -------
        float
            Double integral of the Double Differential data

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> assert dd.doubleIntegral == 7.0
        """
        return integrate(self.rowIntegral)

    @property
    def pdf(self) -> pd.DataFrame:
        """
        Probability density function of the Double Differential data.

        Returns
        -------
        pd.Series
            Probability density function of the Double Differential data

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> dd.pdf
        Eout         1         2         3
        mu
        1     0.142857  0.285714  0.428571
        2     0.571429  0.714286  0.857143
        """
        return self.data / self.doubleIntegral

    @property
    def rowPdf(self) -> pd.Series:
        """
        Probability density function of the Double Differential data by row.

        Returns
        -------
        pd.Series
            Probability density function of the Double Differential data by row

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> dd.rowPdf
        mu
        1    0.571429
        2    1.428571
        dtype: float64
        """
        return self.integrate(axis=1) / self.doubleIntegral

    @property
    def columsPdf(self) -> pd.Series:
        """
        Probability density function of the Double Differential data by column.

        Returns
        -------
        pd.Series
            Probability density function of the Double Differential data by column

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> dd.columsPdf
        Eout
        1    0.357143
        2    0.500000
        3    0.642857
        dtype: float64
        """
        return self.integrate(axis=0) / self.doubleIntegral

    @property
    def cdf(self) -> pd.DataFrame:
        """
        Cumulative distribution function of the Dynamic Structure Factor.

        Returns
        -------
        pd.Series
            Cumulative distribution function of the Dynamic Structure Factor

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> dd.cdf
        Eout         1         2         3
        mu
        1     0.047619  0.142857  0.285714
        2     0.238095  0.571429  1.000000

        """
        cdf = self.data.cumsum(axis=0).cumsum(axis=1)
        return cdf / cdf.iloc[-1, -1]

    def integrate(self, axis: int) -> pd.Series:
        """
        Integrate the Double Differential data.

        Parameters
        ----------
        axis: int
            The axis to integrate

        Returns
        -------
        pd.Series
            The integrated Double Differential data

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> dd.integrate(axis=0)
        Eout
        1    2.5
        2    3.5
        3    4.5
        dtype: float64

        >>> dd.integrate(axis=1)
        mu
        1     4.0
        2    10.0
        dtype: float64
        """
        return self.data.apply(integrate, axis=axis)


    def update_axis(self, newAxis:[pd.Index, np.ndarray], axis: int = 0) -> None:
        """
        Update the axis of the Double Differential data.

        Parameters
        ----------
        newAxis : pd.Index or np.ndarray
            The new axis for the Double Differential data
        axis : int, optional
            The axis to update. The default is 0.

        Returns
        -------
        DoubleDiffData
            The updated Double Differential data
        """
        axisName = self.data.index.name if axis == 0 else self.data.columns.name

        if not isinstance(newAxis, pd.Index):
            newAxis = pd.Index(newAxis)

        if axis == 0:
            self.data.index = newAxis
            self.data.index.name = axisName
        else:
            self.data.columns = newAxis
            self.data.columns.name = axisName


    def update(self, newValues: [np.ndarray, pd.DataFrame]) -> None:
        """
        Update the Double Differential data with data of the same dimensions.

        Parameters
        ----------
        newValues : np.ndarray or pd.DataFrame
            The new values for the Double Differential data

        Returns
        -------
        DoubleDiffData
            The updated Double Differential data

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> newValues = pd.DataFrame([[2, 4, 6], [8, 10, 12]], index=[2, 4], columns=[2, 4, 6])
        >>> dd.update(newValues)
        >>> dd.data
        Eout  2   4   6
        mu
        2     2   4   6
        4     8  10  12

        >>> newValues = np.array([[1, 2, 3], [4, 5, 6]])
        >>> dd.update(newValues)
        >>> dd.data
        Eout  2  4  6
        mu
        2     1  2  3
        4     4  5  6
        """
        # Check if the dimensions match before performing any operation
        if newValues.shape != self.data.shape:
            raise ValueError(
                f"The dimension of the new values must be {self.data.shape} and they are {newValues.shape}")

        # If it's a DataFrame, optimize the copy of values, index, and columns
        if isinstance(newValues, pd.DataFrame):
            # Copy the values without creating a new matrix
            np.copyto(self.data.values, newValues.values)

            # Assign index and columns directly to avoid the overhead of np.copyto
            # Mantaining the names:
            self.update_axis(newValues.index, axis=0)
            self.update_axis(newValues.columns, axis=1)

        else:
            np.copyto(self.data.values, newValues)

    def inplace(self, NewData: pd.DataFrame) -> None:
        """
        Make the Double Differential data in place for changing the dimensions
        of the data.
        """
        self.data = NewData

    def cut_axis(self, threshold: float = 0.0, axis: int = 0,
                 inplace: bool = False) -> [pd.DataFrame, None]:
        """
        Drop the axis of the Double Differential data where the maximum value is
        below the threshold.

        Parameters
        ----------
        threshold : float, optional
            The threshold to drop the axis. The default is 0.0.
        axis : int, optional
            The axis to drop. The default is 0.
        inplace : bool, optional
            If True, do operation in place. The default is False.

        Returns
        -------
        pd.DataFrame or None
            The Double Differential data with the axis dropped

        Examples
        --------
        >>> data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, 2], columns=[1, 2, 3])
        >>> dd = DoubleDiffData(data)
        >>> dd.cut_axis(threshold=3, axis=0)
        Eout  1  2  3
        mu
        2     4  5  6

        >>> dd.cut_axis(threshold=4, axis=1)
        Eout  2  3
        mu
        1     2  3
        2     5  6

        >>> dd.cut_axis(threshold=3, axis=0, inplace=True)
        >>> dd.data
        Eout  1  2  3
        mu
        2     4  5  6
        """
        axis_pd = 1 if axis == 0 else 0

        # Identify axis where the maximum value is below the threshold
        mask = (self.data.max(axis=axis_pd) <= threshold).values

        # Drop the axis
        if axis == 0:
            dataCut = self.data.drop(index=self.data.index[mask])
        elif axis == 1:
            dataCut = self.data.drop(columns=self.data.columns[mask])
        else:
            raise ValueError("Axis must be 0 (rows) or 1 (columns).")

        return self.inplace(dataCut) if inplace else dataCut


class Sab_to_DynamicStruc(DoubleDiff):
    """
    Abstract class for Dynamic Structure Factor calculations.
    """
    def __init__(self, *args):
        """
        Initialize the Sab_to_DynamicStruc class.

        Parameters
        ----------
        Ein : float
            The incident energy of the neutron in eV
        M : float
            The mass of the target material in amu
        T : float
            Temperature of the material in K
        Eout : np.array
            The neutron outgoing energy
        mu : np.ndarray
            The cosine of the angle of the distribution in degrees
        """
        super().__init__(*args)

    @property
    def normFactor(self) -> np.ndarray:
        """
        Calculate the normalization factor.

        Returns
        -------
        np.ndarray
            The normalization factor
        """
        return self.aws / (2 * kb * self.T) * np.sqrt(self.Eout / self.Ein)

    def apply_norm(self, dynamicStructure: np.ndarray):
        """
        Normalize the Dynamic Structure Factor.

        Parameters
        ----------
        dynamicStructure : np.ndarray
            The Dynamic Structure Factor values

        Returns
        -------
        np.ndarray
            The normalized Dynamic Structure Factor values
        """
        return dynamicStructure * self.normFactor

    def sct(self, Tratio: float, ws: float = 1.0) -> np.ndarray:
        """
        Calculate the Dynamic Structure Factor from a S(alpha, -beta) table based
        on Short Collision Time model.

        Parameters
        ----------
        Tratio: float
            Ratio of the effective temperature to the real temperature
        ws : float, optional
            Normalization for continuous (vibrational) part. For solid is 1.

        Returns
        -------
        np.ndarray
            The Dynamic Structure Factor values from a S(alpha, -beta) table
            based on Short Collision Time model

        Examples
        --------
        >>> from solid_cinel.data.examples.UO2 import rho_in_energy_U238, interv_in_energy_U238
        >>> Ein = 7.2
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
        >>> mu = np.cos(np.deg2rad(theta))[::-1]
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> sabValues = Sab_to_DynamicStruc(Ein, M, T, Eout, mu)
        >>> values = sabValues.sct(pdos.fix_T(T).Teff / T, ws=1)
        >>> index = pd.Index(mu, name="mu")
        >>> columns = pd.Index(Eout, name="Eout")
        >>> pd.DataFrame(values, index=index, columns=columns).round(6)
        Eout             6.7554    6.9050    7.0439     7.2000    7.3157    7.4480
        mu
        -9.659258e-01  0.094001  0.636412  1.342345   0.987382  0.367669  0.054937
        -8.660254e-01  0.075434  0.592611  1.358169   1.031486  0.377620  0.053100
        -7.071068e-01  0.050039  0.516194  1.377318   1.109089  0.393570  0.049515
        -5.000000e-01  0.025312  0.406041  1.386155   1.227206  0.413997  0.043483
        -2.588190e-01  0.008381  0.269913  1.359573   1.397842  0.435292  0.034377
         6.123234e-17  0.001348  0.133285  1.255372   1.641602  0.449328  0.022419
         2.588190e-01  0.000056  0.037238  1.014880   1.995877  0.438696  0.010033
         5.000000e-01  0.000000  0.003057  0.602973   2.535640  0.370193  0.001978
         7.071068e-01  0.000000  0.000011  0.156817   3.436247  0.206125  0.000047
         8.660254e-01  0.000000  0.000000  0.002116   5.225195  0.024538  0.000000
         9.659258e-01  0.000000  0.000000  0.000000  10.545191  0.000000  0.000000
        """
        return self.apply_norm(get_SabSct(self.alpha, self.beta, Tratio, ws))


    def tau(self, tauN: np.ndarray, tauNbeta: np.ndarray,
            DebyeWallerCoeff: float) -> np.ndarray:
        """
        Calculate the Dynamic Structure Factor from a S(alpha, -beta) table based
        on phonon expansion tau functions.

        Parameters
        ----------
        tauN: np.ndarray, (Z, T)
            tauN function. Z is the number of phonon expansion order and T is
            the number of beta grid points.
        tauNbeta: np.ndarray, (T,)
            Beta grid for the tauN function
        DebyeWallerCoeff: float
            Debye-Waller coefficient in LEAPR formalism

        Returns
        -------
        np.ndarray
            The Dynamic Structure Factor values from a S(alpha, -beta) table
            based on phonon expansion tau functions

        Examples
        --------
        >>> from solid_cinel.data.examples.UO2 import rho_in_energy_U238, interv_in_energy_U238
        >>> from solid_cinel import calc_alpha, AlphaBase
        >>> Ein = 7.2
        >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
        >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([40, 80, 120, 160])
        >>> mu = np.cos(np.deg2rad(theta))[::-1]
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> DebyeWallerCoeff = pdos.DebyeWallerCoeff
        >>> alpha = AlphaBase(calc_alpha(Ein, M, T, Eout, mu.min()))
        >>> nphonon = alpha.expansionOrder(DebyeWallerCoeff, 1.0e-6, 5000)
        >>> tauN = pdos.tauN(nphonon, 1.0e-14, values=True)
        >>> tauNbeta = get_tauNbeta(pdos.beta.data, tauN.shape[1])
        >>> sabValues = Sab_to_DynamicStruc(Ein, M, T, Eout, mu)
        >>> values = sabValues.tau(tauN, tauNbeta, DebyeWallerCoeff)
        >>> index = pd.Index(mu, name="mu")
        >>> columns = pd.Index(Eout, name="Eout")
        >>> pd.DataFrame(values, index=index, columns=columns).loc[::, Eout_test].round(6)
        Eout         6.7554    6.9050    7.0439    7.2000    7.3157    7.4480
        mu
        -0.939693  0.090037  0.621304  1.346632  1.002256  0.369299  0.053688
        -0.500000  0.026675  0.403624  1.383023  1.232575  0.412462  0.042739
         0.173648  0.000332  0.065453  1.101820  1.874502  0.442317  0.013838
         0.766044  0.000000  0.000009  0.077047  3.956722  0.133747  0.000021
        """
        # Interpolation of tauN functions to reduce the number of calculations:
        tauNinterp = interp_multyParallel(self.betaAbs, tauNbeta, tauN)

        # Get the S(alpha, -beta) values for the alpha and beta combinations:
        # Correct alpha values for absolute beta values:
        sabValues = phonon_expansion(self.alpha, tauN.shape[0],
                                     tauNinterp, DebyeWallerCoeff)

        # Dynamic Structure factor values selection:
        col = self.downScatIndex
        dynamicStruc = np.concatenate(
            (sabValues[::, :col],
             np.exp(-self.betaAbs[col:]) * sabValues[::, col:]),
            axis=1
        )
        # Normalization constant
        return self.apply_norm(dynamicStruc)

    def calc_values(self, *args, model: str = "fgm", **kwargs) -> np.ndarray:
        """
        Calculate the Dynamic Structure Factor values. If no model is introduced,
        is going to be suposed that the values are precomputed tau functions.

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

        Parameters for precomputed tau functions
        ----------------------------------------
        tauN: np.ndarray, (Z, T)
            tauN function. Z is the number of phonon expansion order and T is
            the number of beta grid points.
        tauNbeta: np.ndarray, (T,)
            Beta grid for the tauN function
        DebyeWallerCoeff: float
            Debye-Waller coefficient in LEAPR formalism

        Returns
        -------
        np.ndarray
            The Dynamic Structure Factor values
        """
        # FGM:
        if model.lower() == "fgm":
            return self.sct(1.0, kwargs.get("ws", 1.0))

        # SCT:
        elif model.lower() == "sct":
            Teff = args[0].fix_T(self.T).Teff
            return self.sct(Teff / self.T, kwargs.get("ws", 1.0))

        # PDOS:
        elif model.lower() == "pdos":
            # Get Tpdos:
            Tpdos = args[0].fix_T(self.T)

            # Get the Debye-Waller coefficient:
            DebyeWallerCoeff = Tpdos.DebyeWallerCoeff

            # Get the expansion order:
            if kwargs.get("nphonon"):
                warnings.warn("Is posible that the expansion order is not enough to get the correct results")
                nphonon = kwargs.get("nphonon")
            else:
                nphonon = AlphaBase(self.alpha).expansionOrder(DebyeWallerCoeff,
                                                    kwargs.get("decimal", 1.0e-6),
                                                    kwargs.get("order_max", 5000))

            # Get tauN function:
            tauN = Tpdos.tauN(nphonon, kwargs.get("threshold", 0.0), values=True)
            tauNbeta = get_tauNbeta(Tpdos.beta.data, tauN.shape[1])
            return self.tau(tauN, tauNbeta, DebyeWallerCoeff)

        # Use precomputed tau functions
        else:
            return self.tau(*args)

    def update(self, Ein: float, M: float = None, T: float = None,
               Eout: np.ndarray = None, mu: np.ndarray = None):
        # Update Ein and consequently Eout:
        self.Ein = Ein
        if Eout is None:
            dE = self.Eout - self.Ein
            self.Eout = Ein + dE
        else:
            self.Eout = Eout

        # Update the rest of attributes if is needed:
        if M is not None:
            self.M = M
        if T is not None:
            self.T = T
        if mu is not None:
            self.mu = mu


class DynamicStruc(DoubleDiffData):
    """
    Dynamic structure factor class.
    """

    def __init__(self, *args, sabValues: Sab_to_DynamicStruc = None,
                 **kwargs):
        """
        Initialize the DynamicStruc class.

        Parameters
        ----------
        Ein : float
            The neutron incident energy in eV
        T : float
            Temperature of the material in K
        M : float
            Mass of the material in amu
        args : Iterable, (N, M)
            The Transfer function data for the pd.DataFrame
        kwargs : dict
            Optional arguments for the construction of the pd.DataFrame
        """
        # The Dynamic Structure data:
        self.sabValues = sabValues
        super().__init__(*args, **kwargs)

    def alpha(self, values=True) -> [np.ndarray, pd.DataFrame]:
        """
        Return the alpha values for the Dynamic Structure Factor.

        Parameters
        ----------
        values: bool, optional
            If True return the values of the alpha. If False return
            the alpha as a pd.DataFrame. The default is True.

        Returns
        -------
        np.ndarray or pd.DataFrame
            The alpha values for the Dynamic Structure Factor
        """
        if values:
            return self.sabValues.alpha
        else:
            return pd.DataFrame(self.sabValues.alpha, index=self.mu, columns=self.Eout)

    def recoil(self, values=True) -> [np.ndarray, pd.DataFrame]:
        """
        Return the recoil energy for the Dynamic Structure Factor.

        Parameters
        ----------
        values: bool, optional
            If True return the values of the recoil energy. If False return
            the recoil energy as a pd.DataFrame. The default is True.

        Returns
        -------
        np.ndarray or pd.DataFrame
            The recoil energy for the Dynamic Structure Factor
        """
        if values:
            return self.sabValues.recoil
        else:
            return pd.DataFrame(self.sabValues.recoil, index=self.mu, columns=self.Eout)

    @property
    def alpha0(self) -> float:
        """
        The $\alpha_0$ parameter of the Dynamic Structure Factor.

        Returns
        -------
        float
            The $\alpha_0$ parameter of the Dynamic Structure Factor

        Examples
        --------
        >>> from solid_cinel.data.examples.UO2 import rho_in_energy_U238, interv_in_energy_U238
        >>> Ein = 7.2
        >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
        >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([40, 80, 120, 160])
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> float(round(DynamicStruc.from_model(Ein, M, T, Eout, theta, pdos, model="pdos", threshold=1.0e-14).alpha0, 6))
        0.32328
        """
        # Get the alpha0 parameter:
        return integrate((self.data * self.sabValues.alpha).apply(integrate)) / 2

    @property
    def norm(self) -> float:
        """
        Normalization of the Dynamic Structure Factor.

        Returns
        -------
        float
            Normalization of the Dynamic Structure Factor
        """
        return super().doubleIntegral

    @property
    def transferFunc(self) -> pd.Series:
        """
        Return the Transference function of the Dynamic Structure Factor.

        Returns
        -------
        pd.Series
            The transfer function

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
        >>> dynamicStructure = DynamicStruc.from_model(Ein, M, T, Eout, theta)
        >>> dynamicStructure.transferFunc.round(6)
        Eout
        6.7554    0.031423
        6.9050    0.404638
        7.0439    1.888728
        7.2000    4.340047
        7.3157    0.688071
        7.4480    0.045257
        dtype: float64
        """
        return super().columsIntegral

    @property
    def angularDistr(self):
        """
        Return the angle distribution of the Dynamic Structure Factor.

        Returns
        -------
        pd.Series
            The angle distribution

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
        >>> dynamicStructure = DynamicStruc.from_model(Ein, M, T, Eout, theta)
        >>> dynamicStructure.angularDistr.round(6)
        mu
        -9.659258e-01    0.480322
        -8.660254e-01    0.482041
        -7.071068e-01    0.484205
        -5.000000e-01    0.485923
        -2.588190e-01    0.486289
         6.123234e-17    0.484720
         2.588190e-01    0.481267
         5.000000e-01    0.479932
         7.071068e-01    0.515987
         8.660254e-01    0.714575
         9.659258e-01    1.435551
        dtype: float64
        """
        return super().rowIntegral

    @classmethod
    def from_model(cls, Ein: float, M: float, T: float, Eout: np.ndarray,
                   theta: np.ndarray, *args, model: str = "fgm", **kwargs):
        """
        Generate Dynamic Structure Factor from a S(alpha, -beta) table.
        ..math::
        S(\theta, E^\prime, E, M, T) = \frac{1}{2 * k_B * T}\sqrt{\frac{^\prime}{E}} S(\alpha(\theta, E^\prime, E, M, T), \beta( E^\prime, E, T))

        Parameters
        ----------
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

        Returns
        -------
        DynamicStruc
            Dynamic Structure Factor from a S(alpha, -beta) table.

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])

        # Using the Free Gas Model:
        >>> DynamicStruc.from_model(Ein, M, T, Eout, theta, model="fgm").data.round(6)
        Eout             6.7554    6.9050    7.0439     7.2000    7.3157    7.4480
        mu
        -9.659258e-01  0.093290  0.635800  1.344517   0.987905  0.366598  0.054415
        -8.660254e-01  0.074800  0.591841  1.360299   1.032095  0.376520  0.052584
        -7.071068e-01  0.049539  0.515196  1.379332   1.109853  0.392419  0.049014
        -5.000000e-01  0.024994  0.404827  1.387900   1.228207  0.412767  0.043015
        -2.588190e-01  0.008241  0.268643  1.360778   1.399190  0.433942  0.033969
         6.123234e-17  0.001317  0.132279  1.255634   1.643445  0.447804  0.022111
         2.588190e-01  0.000054  0.036774  1.013814   1.998435  0.436944  0.009862
         5.000000e-01  0.000000  0.002991  0.600838   2.539266  0.368245  0.001932
         7.071068e-01  0.000000  0.000010  0.155387   3.441598  0.204433  0.000045
         8.660254e-01  0.000000  0.000000  0.002062   5.233842  0.024125  0.000000
         9.659258e-01  0.000000  0.000000  0.000000  10.563289  0.000000  0.000000

        # Using the Short Collision Time model:
        >>> from solid_cinel.data.examples.UO2 import rho_in_energy_U238, interv_in_energy_U238
        >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
        >>> DynamicStruc.from_model(Ein, M, T, Eout, theta, pdos, model="sct").data.round(6)
        Eout             6.7554    6.9050    7.0439     7.2000    7.3157    7.4480
        mu
        -9.659258e-01  0.094001  0.636412  1.342345   0.987382  0.367669  0.054937
        -8.660254e-01  0.075434  0.592611  1.358169   1.031486  0.377620  0.053100
        -7.071068e-01  0.050039  0.516194  1.377318   1.109089  0.393570  0.049515
        -5.000000e-01  0.025312  0.406041  1.386155   1.227206  0.413997  0.043483
        -2.588190e-01  0.008381  0.269913  1.359573   1.397842  0.435292  0.034377
         6.123234e-17  0.001348  0.133285  1.255372   1.641602  0.449328  0.022419
         2.588190e-01  0.000056  0.037238  1.014880   1.995877  0.438696  0.010033
         5.000000e-01  0.000000  0.003057  0.602973   2.535640  0.370193  0.001978
         7.071068e-01  0.000000  0.000011  0.156817   3.436247  0.206125  0.000047
         8.660254e-01  0.000000  0.000000  0.002116   5.225195  0.024538  0.000000
         9.659258e-01  0.000000  0.000000  0.000000  10.545191  0.000000  0.000000


        # Using the Phonon expansion model:
        >>> Ein = 7.2
        >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
        >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([40, 80, 120, 160])

        >>> DynamicStruc.from_model(Ein, M, T, Eout, theta, pdos, threshold=1.0e-14, model="pdos").data.loc[::, Eout_test].round(6)
        Eout         6.7554    6.9050    7.0439    7.2000    7.3157    7.4480
        mu
        -0.939693  0.090037  0.621304  1.346632  1.002256  0.369299  0.053688
        -0.500000  0.026675  0.403624  1.383023  1.232575  0.412462  0.042739
         0.173648  0.000332  0.065453  1.101820  1.874502  0.442317  0.013838
         0.766044  0.000000  0.000009  0.077047  3.956722  0.133747  0.000021
        """
        # Get the cosine of the angle of the distribution:
        mu = np.cos(np.deg2rad(theta))

        # Create the object for the calculation of the Dynamic Structure Factor:
        sabValues = Sab_to_DynamicStruc(Ein, M, T, Eout, mu)

        return cls(sabValues.calc_values(*args, model=model, **kwargs),
                   sabValues=sabValues, index=sabValues.mu, columns=sabValues.Eout)

    def update(self, Ein: float, *args, model: str = "fgm",  M: float = None,
               T: float = None, Eout: np.ndarray = None, mu: np.ndarray = None,
               **kwargs) -> None:
        """
        Update the Dynamic Structure Factor object with new values.

        Parameters
        ----------
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

        Returns
        -------
        None
            Update the Dynamic Structure Factor object with new values.

        Examples
        --------
        >>> Ein = 7.2
        >>> Eout = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> theta = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
        >>> test = DynamicStruc.from_model(Ein, M, T, Eout, theta, model="fgm")
        >>> test.data.round(6)
        Eout             6.7554    6.9050    7.0439     7.2000    7.3157    7.4480
        mu
        -9.659258e-01  0.093290  0.635800  1.344517   0.987905  0.366598  0.054415
        -8.660254e-01  0.074800  0.591841  1.360299   1.032095  0.376520  0.052584
        -7.071068e-01  0.049539  0.515196  1.379332   1.109853  0.392419  0.049014
        -5.000000e-01  0.024994  0.404827  1.387900   1.228207  0.412767  0.043015
        -2.588190e-01  0.008241  0.268643  1.360778   1.399190  0.433942  0.033969
         6.123234e-17  0.001317  0.132279  1.255634   1.643445  0.447804  0.022111
         2.588190e-01  0.000054  0.036774  1.013814   1.998435  0.436944  0.009862
         5.000000e-01  0.000000  0.002991  0.600838   2.539266  0.368245  0.001932
         7.071068e-01  0.000000  0.000010  0.155387   3.441598  0.204433  0.000045
         8.660254e-01  0.000000  0.000000  0.002062   5.233842  0.024125  0.000000
         9.659258e-01  0.000000  0.000000  0.000000  10.563289  0.000000  0.000000

        >>> Ein = 10.0
        >>> test.update(Ein, Eout = np.array([9.7554, 9.905 , 10.0439, 10.2   , 10.3157, 10.448 ]))
        >>> test.data.round(6)
        Eout            9.7554    9.9050    10.0439   10.2000   10.3157   10.4480
        mu
        -9.659258e-01  1.053806  1.084871  0.548853  0.115122  0.021235  0.001782
        -8.660254e-01  1.047362  1.131481  0.576358  0.116741  0.020394  0.001562
        -7.071068e-01  1.026262  1.210979  0.624597  0.118662  0.018762  0.001216
        -5.000000e-01  0.973528  1.325333  0.697483  0.119428  0.016063  0.000793
        -2.588190e-01  0.864619  1.475017  0.801224  0.116404  0.012126  0.000390
         6.123234e-17  0.674457  1.654099  0.945262  0.105390  0.007276  0.000117
         2.588190e-01  0.404434  1.836635  1.143672  0.081310  0.002790  0.000014
         5.000000e-01  0.135451  1.938151  1.415728  0.043497  0.000406  0.000000
         7.071068e-01  0.009922  1.717377  1.773530  0.008798  0.000005  0.000000
         8.660254e-01  0.000003  0.750993  2.088814  0.000056  0.000000  0.000000
         9.659258e-01  0.000000  0.001715  1.005235  0.000000  0.000000  0.000000

        >>> test.update(Ein, Eout=np.array([9.7554, 9.905]))
        >>> test.data.round(6)
        Eout             9.7554    9.9050
        mu
        -9.659258e-01  1.053806  1.084871
        -8.660254e-01  1.047362  1.131481
        -7.071068e-01  1.026262  1.210979
        -5.000000e-01  0.973528  1.325333
        -2.588190e-01  0.864619  1.475017
         6.123234e-17  0.674457  1.654099
         2.588190e-01  0.404434  1.836635
         5.000000e-01  0.135451  1.938151
         7.071068e-01  0.009922  1.717377
         8.660254e-01  0.000003  0.750993
         9.659258e-01  0.000000  0.001715
        """
        # Update the calculation class:
        self.sabValues.update(Ein, M=M, T=T, Eout=Eout, mu=mu)

        # Get the new values:
        dynamicStructure = self.sabValues.calc_values(*args, model=model,
                                                      **kwargs)

        # Update the values
        if dynamicStructure.shape == self.data.shape:
            # Update the values:
            super().update(dynamicStructure)

            # Update the Eout values:
            super().update_axis(Eout, axis=1)

            # Update the mu values if they change:
            if mu is not None:
                super().update_axis(mu, axis=0)
        else:
            if mu is None:
                mu = self.mu
            super().inplace(pd.DataFrame(dynamicStructure, index=mu, columns=Eout))


@nb.jit(nopython=True, cache=True)
def normFactor(Eout: np.ndarray, Ein: float, T: float, M: float) -> np.ndarray:
    """
    Normalization factor for the Transfer function calculation.

    Parameters
    ----------
    Eout: 'np.ndarray', (N,)
        Outgoing energy grid in eV.
    Ein: 'float'
        Incident energy in eV.
    T: 'float'
        Temperature in K.
    M: 'float'
        Mass of the target in amu.

    Returns
    -------
    'np.ndarray', (N,)
        Normalization factor for the Transfer function calculation.
    """
    M_div_m = M / m
    aws = ((M_div_m + 1) / M_div_m) ** 2
    two_kb_T = 2 * kb * T
    return aws * np.sqrt(Eout / Ein) / two_kb_T

@nb.jit(nopython=True, cache=True)
def get_ScatFuncClm(Ein: float, M: float, T: float, Eout: np.ndarray, mu: np.ndarray,
                    tauN: np.ndarray, tauNbeta: np.ndarray,
                    DebyeWallerCoeff: float, alpha0: float) -> np.ndarray:
    """
    Generate the Transfer function from a S(alpha, -beta) table based on
    the phonon expansion model.

    Parameters
    ----------
    Ein : float
        The incident energy of the neutron in eV
    M : float
        The mass of the target material in amu
    T : float
        Temperature of the material in K
    Eout : np.ndarray, (N,)
        The neutron outgoing energy grid in eV
    mu : float
        Cosine of the scattering angle
    tauN : 'np.ndarray', (M, T)
        all tau n functions in one array.
    tauNbeta : 'np.ndarray', (M,)
        Space between beta grid points of tau n functions.
    DebyeWallerCoeff : float
        Debye Waller coefficient

    Returns
    -------
    S_diag : 'np.ndarray', (N,)
        Transfer function values for a single angle.

    Examples
    --------
    >>> from solid_cinel.data.examples.UO2 import rho_in_energy_U238, interv_in_energy_U238
    >>> from solid_cinel import AlphaBase, calc_alpha
    >>> Ein = 7.2
    >>> Eout = np.linspace(6.7554, 7.448, num=1000, endpoint=True)
    >>> Eout_test = np.array([6.7554, 6.905 , 7.0439, 7.2   , 7.3157, 7.448 ])
    >>> Eout = np.unique(np.concatenate((Eout, Eout_test), axis=None))
    >>> T = 1000.0
    >>> M = 238.05077040419212
    >>> mu = np.cos(np.deg2rad([120]))
    >>> pdos = Pdos.from_dE(T, rho_in_energy_U238, interv_in_energy_U238)
    >>> DebyeWallerCoeff = pdos.DebyeWallerCoeff
    >>> alpha = AlphaBase(calc_alpha(Ein, M, T, Eout, mu.min()))
    >>> nphonon = alpha.expansionOrder(DebyeWallerCoeff, 1.0e-6, 5000)
    >>> tauN = pdos.tauN(nphonon, 1.0e-14, values=True)
    >>> tau1beta = pdos.beta.data
    >>> tauNbeta = get_tauNbeta(tau1beta, tauN.shape[1])

    # Using the alpha0 parameter:
    >>> alpha0 = Ein / M / (kb * T)
    >>> sd_pdf = get_ScatFuncClm(Ein, M, T, Eout, mu, tauN, tauNbeta, DebyeWallerCoeff, alpha0)
    >>> pd.Series(sd_pdf[0], index=Eout).loc[Eout_test].round(6)
    6.7554    0.000002
    6.9050    0.005037
    7.0439    0.605736
    7.2000    2.560414
    7.3157    0.360692
    7.4480    0.002042
    dtype: float64
    """
    # Get the beta grid:
    betaAbs = get_AbsBeta(Eout, Ein, T, unique=False, sort=False)

    # Define the dowscattering mask:
    mask = Eout <= Ein

    # Interpolation of tauN functions to reduce the number of calculations:
    tauNinterp = interp_multyParallel(betaAbs, tauNbeta, tauN)

    # Get the S(alpha, -beta) values for the alpha and beta combinations:
    sabValues = phonon_expansion(get_alphaMatMod(Eout, Ein, M, T, mu, DebyeWallerCoeff, alpha0),
                                 tauN.shape[0], tauNinterp, DebyeWallerCoeff)

    # Dynamic Structure factor values selection:
    dynamicStruc = np.concatenate(
        (sabValues[::, mask],
         np.exp(-betaAbs[~mask]) * sabValues[::, ~mask]),
        axis=1
    )
    # Normalization constant
    return dynamicStruc * normFactor(Eout, Ein, T, M)
