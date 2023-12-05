"""
Python file for working with Phonon Density Of States.

@author: AB272525
"""
from solid_cinel.core.generic import integrate
from solid_cinel.core.material.scattering_function.beta import Beta
from solid_cinel.core.material.vibration.tau import tau_n_functions
import pandas as pd
import numpy as np
import scipy as sp
from typing import Iterable, Union
import matplotlib


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


class Pdos:
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
    to_beta_grid: Pdos
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
        Calculate the tau_n functions
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
        self.rho = pd.Series(*args, **kwargs)

    @property
    def rho(self) -> pd.Series:
        """Pandas Series containing the rho values in energy (index)."""
        return self.data

    @rho.setter
    def rho(self, rho_data: Union[pd.Series, Iterable]) -> pd.Series:
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
        >>> p = pdos.from_dE(rho_in_energy, interv_in_energy)

        Test the results:
        >>> assert sp.integrate.trapezoid(p.data.values, p.data.index) == 1.0
        """
        rho_ = pd.Series(rho_data, name="rho")

        if rho_.index.name != "beta" and rho_.index.name != "E":
            raise SyntaxError("Energy index must be E or beta")

        if not len(rho_.shape) == 1:
            raise TypeError("Rho must have one dimension")

        if not rho_.index.is_monotonic_increasing:
            raise SyntaxError("energy grid is not monotonically increasing")

        self.data = rho_ / integrate(rho_)

    @property
    def grid(self) -> [float, np.ndarray]:
        """
        Grid of the Phonon Density of States function

        Returns
        -------
        "float" or "pd.Index"
            Grid of the Phonon Density of States function

        Examples
        --------
        Object initialization:
        >>> p = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> round(p.grid, 4)
        0.0008
        """
        return np.ediff1d(self.rho.index)[0]

    @classmethod
    def from_dE(cls, rho: Iterable, interval_energy: float):
        """
        Extract rho in energy from the introduced data.

        Parameters
        ----------
        rho : 1D iterable
            rho values.
        interval_energy : 'float'
            Energy interval in eV.

        Returns
        -------
        "Pdos"
            Rho normalize object.

        Examples
        --------
        Object initialization:
        >>> p = Pdos.from_dE(rho_in_energy, interv_in_energy)

        Test the results:
        >>> p.rho.iloc[0:10]
        E
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
        index = pd.Index(np.arange(len(rho_)) * interval_energy)
        index.name = "E"
        return cls(rho_, index=index)

    def to_beta_grid(self, T: float):
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
        "Pdos"
            pdos object with the new grid.

        Examples
        --------
        Object initialization:
        >>> p = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> p.rho.iloc[0:5]
        E
        0.0000    0.000000
        0.0008    0.041157
        0.0016    0.164629
        0.0024    0.370415
        0.0032    0.657892
        Name: rho, dtype: float64

        Test the results:
        >>> T = 300
        >>> p.to_beta_grid(T).rho.iloc[0:5]
        beta
        0.000000    0.000000
        0.030945    0.001064
        0.061891    0.004256
        0.092836    0.009576
        0.123782    0.017008
        Name: rho, dtype: float64

        >>> p.to_beta_grid(T).to_beta_grid(T).rho.iloc[0:5]
        beta
        0.000000    0.000000
        0.030945    0.001064
        0.061891    0.004256
        0.092836    0.009576
        0.123782    0.017008
        Name: rho, dtype: float64
        """
        if self.rho.index.name == "beta":
            return self
        else:
            grid = self.rho.index.values
            return Pdos(self.rho.values,
                        index=Beta.from_dE(grid, T).to_index)

    def plot(self) -> matplotlib:
        """Plot rho (y) vs grid (x)."""
        return self.data.plot(title='PDOS')

    def P(self, T: float, threshold: float = 1.e-6) -> pd.Series:
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
        >>> pdos = Pdos.from_dE(rho_in_energy, interv_in_energy)

        Test the results:
        >>> T = 300
        >>> pdos.P(T).iloc[0:6].round(6)
        beta
        0.000000    1.111089
        0.030945    1.111045
        0.061891    1.110912
        0.092836    1.110690
        0.123782    1.109328
        0.154727    1.109309
        Name: P, dtype: float64
        """
        beta_rho = self.to_beta_grid(T).rho
        rho_in_beta, beta_values = beta_rho.values, beta_rho.index.values
        if abs(beta_values[0]) > threshold:
            raise ValueError("Initial point of input DOS is not zero")
        P_values = np.zeros(len(rho_in_beta))

        # rho_in_beta is assumed to vary as beta^2 in the nearby of 0
        P_values[0] = rho_in_beta[1] / beta_values[1] ** 2

        # Rest of P values calculation:
        P_values[1:] = 0.5 * rho_in_beta[1:] / beta_values[1:] / np.sinh(0.5 * beta_values[1:])
        return pd.Series(P_values, index=beta_rho.index, name="P")

    def Teff(self, T: float, twt: float = None) -> float:
        """
        Calculate the effective temperature for a certain pdos information.
        .. math::
            T_{eff} = w_t * T + \frac{1}{2k_B}\int_{0}^{\infty}\varepsilon \rho(\varepsilon)\coth\left(\frac{\varepsilon}{2k_BT}\right)d\varepsilon = \left(w_t+\int_{0}^{\infty}\beta^2P(\beta)\cosh(\beta/2)d\beta\right)T

        Parameters
        ----------
        T : 'float'
            Temperature in K.
        twt : 'float', optional
            Translational weight, for solid is zero. The default is None.

        Returns
        -------
        "float"
            Effective temperature for certain pdos.

        Example
        -------
        Object initialization:
        >>> p = Pdos.from_dE(rho_in_energy, interv_in_energy)

        Test the results:
        >>> p.Teff(T=20).round(4)
        149.1699
        >>> p.Teff(T=80).round(4)
        159.1632
        """
        P = self.P(T)
        P_values, beta = P.values, P.index.values
        Teff_weight = sp.integrate.trapezoid(beta ** 2 * P_values * np.cosh(0.5 * beta),
                                             x=beta)
        if twt is not None:
            Teff_weight += twt
        return Teff_weight * T

    def DebyeWallerCoeff(self, T: float) -> float:
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
        >>> p = Pdos.from_dE(rho_in_energy, interv_in_energy)

        Test the results:
        >>> p.DebyeWallerCoeff(T=20).round(6)
        0.077454
        >>> p.DebyeWallerCoeff(T=80).round(6)
        0.379937
        """
        P = self.P(T)
        P_values, beta = P.values, P.index.values
        return 2 * sp.integrate.trapezoid(P_values * np.cosh(0.5 * beta),
                                          x=beta)

    def get_tau_1(self, T: float) -> pd.Series:
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
        >>> p = Pdos.from_dE(rho_in_energy, interv_in_energy)

        Test the results:
        >>> p.get_tau_1(20).iloc[:10]
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
        P = self.P(T)
        beta = P.index.values
        tau1 = P * np.exp(0.5 * beta) / self.DebyeWallerCoeff(T)
        if integrate(tau1 * (1 + np.exp(-beta))) < 1.e-5:
            raise ValueError("Tau function for 1 phonon expansion doesnt satisfy the normalization condition")
        tau1.name = 1
        return tau1

    def get_tau(self, T: float, nphonon: int = 1000,
                threshold: float = 0.0, check: bool = True) -> pd.DataFrame:
        """
        Get the Tau(-beta) function for n phonon expansion in LEAPR formalism.

        Parameters
        ----------
        T: 'float'
            Temperature in K.
        nphonon: 'int'
            Number of phonon to calculate the tau functions.
        threshold: 'float'
            Minimun value to take into account.

        Returns
        -------
        "pd.DataFrame", (len(rho) * nphonon, nphonon)
            Tau(-beta) function for n phonon.

        Examples
        --------
        Object initialization:
        >>> p = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> T = 800
        >>> nphonon = 5
        >>> tau_n = p.get_tau(T, nphonon=nphonon)
        >>> tau_n.iloc[::, :100:10].round(6)
           0.000000  0.116045  0.232090  0.348136  0.464181  0.580226  0.696271  0.812316  0.928361  1.044407
        1  0.862582  0.912907  1.322890  1.664078  0.341423  0.000000  0.000000  0.000000  0.000000  0.000000
        2  1.068786  0.904301  0.835423  0.709480  0.650492  0.638436  0.397400  0.246076  0.067640  0.031901
        3  0.721827  0.754397  0.778009  0.742670  0.645243  0.539548  0.431710  0.347367  0.257169  0.164468
        4  0.649349  0.670815  0.669368  0.646839  0.608380  0.553518  0.476611  0.391567  0.305529  0.230576
        5  0.572522  0.598559  0.608795  0.600007  0.572271  0.529334  0.475181  0.414218  0.348585  0.281873

        >>> nphonon = 10
        >>> tau_n = p.get_tau(T, nphonon=nphonon, check=False)
        >>> tau_n.iloc[::2, :100:10].round(6)
           0.000000  0.116045  0.232090  0.348136  0.464181  0.580226  0.696271  0.812316  0.928361  1.044407
        1  0.862582  0.912907  1.322890  1.664078  0.341423  0.000000  0.000000  0.000000  0.000000  0.000000
        3  0.721827  0.754397  0.778009  0.742670  0.645243  0.539548  0.431710  0.347367  0.257169  0.164468
        5  0.572522  0.598559  0.608795  0.600007  0.572271  0.529334  0.475181  0.414218  0.348585  0.281873
        7  0.479140  0.502375  0.515558  0.517681  0.508476  0.488495  0.458931  0.421466  0.378055  0.330937
        9  0.416041  0.437155  0.451569  0.458534  0.457651  0.448911  0.432693  0.409732  0.381067  0.347970
        """
        delta_beta = self.to_beta_grid(T).grid
        tau_n = pd.DataFrame(
            tau_n_functions(self.get_tau_1(T).values, delta_beta,
                            nphonon, threshold)
        )
        tau_n.columns *= delta_beta
        tau_n.index += 1
        if check:
            # tau1 is not included in the check:
            integrals_value = tau_n.apply(integrate, axis=1).iloc[1::]
            if (integrals_value < 1.e-5).any():
                raise ValueError(
                    "Tau function doesnt satisfy the normalization condition")
        return tau_n