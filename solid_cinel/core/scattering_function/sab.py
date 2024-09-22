"""
Python file for working with S(alpha, -beta) matrixs.

@author: AB272525
"""
from scipy.constants import physical_constants as const
from solid_cinel.core.generic import integrate, reshape_differential, interp_multyParallel
from solid_cinel.core.material.pdos import Pdos
from solid_cinel.core.material.tau import get_tauNfunc, get_tauNbeta
from solid_cinel.core.scattering_function.beta import Beta
from solid_cinel.core.scattering_function.alpha import Alpha, get_expansionOrder
from typing import Iterable, Union
import numpy as np
import pandas as pd
import numba as nb
from math import pi
from numba import float64, int32
import warnings

# Constants:
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]


class Sab:
    """
    Class containing all the methods and properties of a asymmetric
    S(alpha, beta) matrix.

    Attributes
    ----------
    data : pd.DataFrame
        Dataframe with the S(alpha, -beta) matrix values.
    alpha : Alpha
        Initialize the Alpha class with the information of S(alpha, beta)
    beta : Beta
        Initialize the Beta class with the information of S(alpha, beta)

    Methods
    -------
    to_sym -> pd.DataFrame
        Return the symmetric S(alpha, -beta) matrix
    full -> pd.DataFrame
        Return the full S(alpha, beta) matrix
    from_fgm -> Sab
        Return the S(alpha, beta) matrix from the FGM model
    from_sct -> Sab
        Return the S(alpha, beta) matrix from the SCT model
    from_pdos -> Sab
        Return the S(alpha, beta) matrix from the PDOS model
    SumRule_check -> bool
        Check if the sum rule is satisfied
    NormCheck -> bool
        Check if the normalization is satisfied
    get_momentum -> Sab or pd.DataFrame
        Return the S(alpha, beta) matrix n momentum
    interp_beta -> Sab or pd.DataFrame
        Quadratic interpolation to get the probability of the new beta value
    interp_alpha -> Sab or pd.DataFrame
        Unit base interpolation to get the probability of the new alpha values
    get_value_from_alpha_beta -> pd.DataFrame
        Return the S(alpha, beta) matrix value for a given alpha and beta
    get_matrix_from_parameters -> Sab or pd.DataFrame
        Based on the set of variables introduced, interpolate the existing
        S(alpha, -beta) to make a new S(alpha, beta) matrix with the alpha and
        beta values created from the set of variables
    get_scattering_function -> pd.DataFrame
        Return the scattering function from S(alpha, beta) matrix
    get_inelastic_Xs -> pd.DataFrame
        Return the inelastic cross section from S(alpha, beta) matrix
    """

    def __init__(self, *args, DebyeWallerCoeff: float = .0, **kwargs):
        """
        Initialize the S(alpha, beta) matrix class.

        Parameters
        ----------
        args : "np.array"
            Array containing the S(alpha, -beta) matrix.
        kwargs : "dict"
            Dictionary containing the S(alpha, -beta) matrix.
        """
        # Get the Debye-Waller coefficient for checking the normalization:
        self.DebyeWallerCoeff = DebyeWallerCoeff

        # Get the S(alpha, -beta) matrix:
        self.data = pd.DataFrame(*args, **kwargs)


    @property
    def alpha(self) -> Alpha:
        """
        Initialize the Alpha class with the information of S(alpha, beta)
        matrix.
        """
        return Alpha(self.data.index.values)

    @staticmethod
    def check_alpha(alpha: Union[Alpha, Iterable, str]) -> Alpha:
        """
        Generate the Alpha class for the creation of S(alpha, beta) table.

        Parameters
        ----------
        alpha : Union[Alpha, Iterable, str]
            Alpha grid information in different formats.

        Returns
        -------
        Alpha
            Alpha class with the alpha grid information.
        """
        if isinstance(alpha, Alpha):
            return alpha
        elif isinstance(alpha, str):
            return Alpha.from_file(alpha)
        else:
            return Alpha(alpha)

    @property
    def beta(self) -> Beta:
        """
        Initialize the Beta class with the information of S(alpha, -beta).
        matrix
        """
        return Beta(self.data.columns.values)

    @staticmethod
    def check_beta(beta: Union[Beta, Iterable, str]) -> Beta:
        """
        Generate the Beta class for the creation of S(alpha, beta) table.

        Parameters
        ----------
        beta : Union[Beta, Iterable, str]
            Beta grid information in different formats.

        Returns
        -------
        Beta
            Beta class with the beta grid information in absolute values.
        """
        # Create the Beta class:
        if isinstance(beta, Beta):
            beta_ = beta
        elif isinstance(beta, str):
            beta_ = Beta.from_file(beta)
        else:
            beta_ = Beta(beta)

        # Check if the beta grid is absolute or not:
        if beta_.kind == "abs":
            return beta_
        else:
            raise ValueError("The beta grid contains negative values and the input is the absolute beta grid")

    @classmethod
    def setup_alpha_beta(cls, alpha: Union[Beta, Iterable, str],
                          beta: Union[Beta, Iterable, str]) -> [np.array, np.array]:
        """
        Setup the Alpha and Beta grids for the calculation of S(alpha, -beta) matrix.

        Parameters
        ----------
        alpha : Union[Beta, Iterable, str]
            Alpha grid information in different formats.
        beta : Union[Beta, Iterable, str]
            Beta grid information in different formats.

        Returns
        -------
        np.array, np.array
            Alpha and Beta grid arrays.
        """
        # Get the Alpha and Beta classes:
        return cls.check_alpha(alpha).data, cls.check_beta(beta).data

    @property
    def data(self) -> pd.DataFrame:
        """Dataframe with the S(alpha, -beta) matrix values."""
        return self._data

    @data.setter
    def data(self, df: Iterable):
        """
        Construct the S(alpha, -beta) matrix and check if the data achieve the
        normalization and sum rule constrain.

        Parameters
        ----------
        df : 2D iterable, (N, M)
            Iterable containing the S(alpha, -beta) matrix.
        """
        # Sort and define the style of the dataframe:
        df_ = pd.DataFrame(df).sort_index(axis=0).sort_index(axis=1)
        df_.index.name = "alpha"
        df_.columns.name = "beta"

        # Normalization constrains:
        self.NormCheck(df_)
        self.SumRule_check(df_)

        # save the data:
        self._data = df_

    def to_sym(self, detail_balance: bool = True) -> pd.DataFrame:
        """
        Generate the symmetric S(alpha, -beta) matrix from the asymmetric
        S(alpha, -beta) matrix.

        Parameters
        ----------
        detail_balance : 'bool', optional
            Relationships between upscatter and downscatter. The default is
            True.

        Returns
        -------
        "pd.DataFrame"
            Dataframe containing the symmetric S(alpha, -beta) matrix.

        Example:
        --------
        >>> beta_grid = Beta.generate_grid(300).data
        >>> alpha = Alpha.generate_grid(300, 26).data
        >>> Sab.from_fgm(alpha, beta_grid).to_sym().iloc[:10, :5].round(6) #doctest: +NORMALIZE_WHITESPACE
        beta      0.000000  0.012894  0.025788  0.038682  0.051576
        alpha
        0.001050  8.701463  8.363896  7.427755  6.094516  4.620122
        0.001087  8.553363  8.232522  7.340419  6.063184  4.639515
        0.001125  8.407781  8.102844  7.252800  6.029567  4.655632
        0.001164  8.264674  7.974859  7.164978  5.993787  4.668553
        0.001205  8.124000  7.848564  7.077028  5.955964  4.678361
        0.001247  7.985718  7.723951  6.989018  5.916214  4.685142
        0.001291  7.849787  7.601016  6.901017  5.874652  4.688985
        0.001336  7.716166  7.479752  6.813088  5.831389  4.689983
        0.001382  7.584817  7.360149  6.725291  5.786534  4.688230
        0.001431  7.455701  7.242199  6.637682  5.740191  4.683821
        """
        return self.data * np.exp(- self.beta.data / 2) if detail_balance else self.data

    @property
    def full(self) -> [pd.Series, pd.DataFrame]:
        """
        Get the full S(alpha, beta) matrix.

        Returns
        -------
        'pd.DataFrame' or 'pd.Series'
            the full S(alpha, beta) matrix. If the matrix has only one row, it
            will return a Series.

        Example
        -------
        >>> beta_grid = Beta.generate_grid(300)
        >>> alpha = Alpha.generate_grid(300, 26)
        >>> Sab_matrix = Sab.from_fgm(alpha, beta_grid)
        >>> Sab_matrix_norm = Sab_matrix.data.apply(_norm, axis=1)
        >>> Sab_matrix_full = Sab_matrix.full
        >>> assert (Sab_matrix_full.apply(integrate, axis=1).round(6) == Sab_matrix_norm.round(6)).all()

        >>> Sab_matrix_sum = Sab_matrix.data.apply(_SumRule, axis=1)
        >>> assert (Sab_matrix_full.apply(lambda x: - integrate(x * x.index), axis=1).round(6) == Sab_matrix_sum.round(6)).all()
        """
        # Get the the beta grid:
        beta = self.beta.to_index

        # Get the S(alpha, -beta) matrix:
        Sab_negative = self.data.set_axis(-beta, axis=1).sort_index(axis=1)

        # Get the S(alpha, +beta) matrix:
        Sab_positive = self.data.apply(lambda x: x * np.exp(-beta), axis=1)

        # Concatenate to create the S(alpha, beta) matrix:
        Sab_complete = pd.concat([Sab_negative, Sab_positive.iloc[::, 1::]],
                                 axis=1)

        # Return the matrix:
        return Sab_complete if len(Sab_complete.index) > 1 else Sab_complete.iloc[0]

    @staticmethod
    def get_Tratio(T: float, pdos: Pdos = None) -> float:
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
        if pdos is None:
            return 1.0
        else:
            # Fix the pdos to the temperature:
            pdosFix = pdos.fix_T(T)

            # Get the temperature ratio:
            ratio = pdosFix.Teff / T
            if np.isnan(ratio):
                warnings.warn("The effective temperature is not defined, the ratio will be 1")
                return 1.0
            else:
                return ratio

    @classmethod
    def from_fgm(cls, alpha: Union[Alpha, Iterable, str], beta: Union[Beta, Iterable, str],
                 T: float = None, wt: float = 1):
        """
        Generate S(alpha, -beta) matrix using Free Gas Model.
        .. math::
            S_t(\alpha,\,-\beta)=\dfrac{1}{\sqrt{4\pi w_t\alpha}}\exp\left(-\dfrac{(w_t\alpha+\beta)^2}{4w_t\alpha}\right)\end{equation}

        Parameters
        ----------
        alpha : 1D iterable or "Alpha", (N,)
            Alpha grid.
        beta_grid : 1D iterable or "Beta", (M,)
            Absolute beta grid.
        model : 'str', optional
            The model to calculate matrix values. The default is "FGM".
        wt: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Returns
        -------
        "Sab", (N, M)
            S(alpha, -beta) based on Free Gas Model.

        Example
        -------
        FGM:
        >>> beta = Beta.generate_grid(300).data
        >>> alpha = Alpha.generate_grid(300, 26).data
        >>> Sab.from_fgm(alpha, beta).data.iloc[:10, :5].round(6) #doctest: +NORMALIZE_WHITESPACE
        beta	      0.000000	0.012894	 0.025788	0.038682 	0.051576
        alpha
        0.001050	  8.701463	8.417992     7.524148	6.213536	4.740815
        0.001087      8.553363	8.285768	 7.435678	6.181592	4.760714
        0.001125	  8.407781	8.155251	 7.346923	6.147319	4.777252
        0.001164	  8.264674	8.026439	 7.257961	6.110841	4.790511
        0.001205	  8.124000	7.899326	 7.168869	6.072279	4.800575
        0.001247	  7.985718	7.773908	 7.079717	6.031753	4.807533
        0.001291	  7.849787	7.650178	 6.990574	5.989379	4.811476
        0.001336	  7.716166	7.528129	 6.901504	5.945271	4.812500
        0.001382	  7.584817	7.407753	 6.812568	5.899540	4.810701
        0.001431	  7.455701	7.289040	 6.723822	5.852292	4.806177
        """
        # Set the beta grid and the alpha grid:
        alpha_, beta_ = cls.setup_alpha_beta(alpha, beta)

        # Get the temperature ratio (1):
        Tratio = cls.get_Tratio(T)

        # Get the S(alpha, -beta) matrix:
        S_values = get_SabSct(alpha_[::, np.newaxis], - beta_, Tratio, wt)

        return cls(S_values, index=alpha_, columns=beta_)

    @classmethod
    def from_sct(cls, alpha: Union[Alpha, Iterable, str], beta: Union[Beta, Iterable, str],
                 T: float, pdos: Pdos, ws: float = 1):
        """
        Generate S(alpha, -beta) matrix using Short Collision Time.
        .. math::
            S(\alpha,\,-\beta)=\sqrt{\dfrac{1}{4\pi\omega_{s}\alpha T_{\textrm{eff}}/T}}\exp\left(-\dfrac{(\omega_{s}\alpha-\beta)^2}{4\omega_{s}\alpha T_{\textrm{eff}}/T}\right)

        Parameters
        ----------
        alpha : 1D iterable or "Alpha", (N,)
            Alpha grid.
        beta_grid : 1D iterable or "Beta", (M,)
            beta grid.
        T : 'float'
            Temperature in K.
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        ws: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Returns
        -------
        "Sab", (N, M)
            S(alpha, -beta) based on Short Collision Time

        Example
        -------
        SCT:
        Dont fit the normalization and sum rule with the correct precision
        >>> from solid_cinel.data.examples.Al27 import rho_in_energy, interv_in_energy
        >>> T = 300
        >>> pdos = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> beta = Beta.generate_grid(T)
        >>> alpha = Alpha.generate_grid(T, 26)
        >>> S = Sab.from_sct(alpha, beta, T, pdos)
        >>> S.data.iloc[:10, :5].round(6) #doctest: +NORMALIZE_WHITESPACE
        beta      0.000000  0.012894  0.025788  0.038682  0.051576
        alpha
        0.001050  8.342190  8.092079  7.298835  6.121534  4.773978
        0.001087  8.200211  7.964121  7.209904  6.084151  4.785744
        0.001125  8.060646  7.837859  7.120876  6.044744  4.794361
        0.001164  7.923454  7.713288  7.031821  6.003428  4.799921
        0.001205  7.788595  7.590401  6.942804  5.960320  4.802517
        0.001247  7.656028  7.469191  6.853888  5.915532  4.802243
        0.001291  7.525715  7.349649  6.765132  5.869173  4.799196
        0.001336  7.397618  7.231765  6.676593  5.821349  4.793476
        0.001382  7.271698  7.115530  6.588322  5.772162  4.785181
        0.001431  7.147919  7.000933  6.500370  5.721713  4.774412
        """
        # Set the beta grid and the alpha grid:
        alpha_, beta_ = cls.setup_alpha_beta(alpha, beta)

        # Get the temperature ratio (1):
        Tratio = cls.get_Tratio(T, pdos)

        # Get the S(alpha, -beta) matrix:
        S_values = get_SabSct(alpha_[::, np.newaxis], - beta_, Tratio, ws)

        return cls(S_values, index=alpha_, columns=beta_)

    @classmethod
    def from_pdos(cls, alpha: Union[Alpha, Iterable, str], beta: Union[Beta, Iterable, str],
                  T: float, pdos: Pdos, nphonon: int = None, decimal: float = 1.0e-6,
                  orderMax: int = 5000, threshold: float = 0.0):
        """
        Generate S(alpha, -beta) matrix using phonon expansion.
        .. math::
            S(\alpha,\,-\beta)=\exp(-\alpha\lambda)\sum_{n=0}^{\infty}\dfrac{1}{n!}(\alpha\lambda)^n\mathcal{T}_n(-\beta)

        Numerical appoximation to get convergence in large exponentiation and
        factorial numbers. Each element of the array is related with one alpha
        and represent the following term of the previous equation:
        ..math::
           \sum_{n=0}^{\infty}\dfrac{1}{n!}(\alpha\lambda)^n = \exp(\log(\dfrac{1}{1}(\alpha\lambda)) + \log(\dfrac{1}{2}(\alpha\lambda)) + ...)

        Parameters
        ----------
        alpha : 1D iterable or "Alpha", (N,)
            Alpha grid.
        beta : 1D iterable or "Beta", (M,)
            beta grid.
        T : 'float'
            Temperature in K.
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        threshold : 'float', optional
            Minimun value to take into account in the creation of tauN
            functions. For T>200 is convenient to set into 1.0e-14 to speed up
            the calculations. The default is 0.0.
        nphonon : 'int', optional
            Phonon expansion order. The default is calculated with the function
            get_expansionOrder.
        decimal : 'float', optional
            Decimal precision to calculate the expansion order. The default is
            1.0e-6.
        order_max : 'int', optional
            Maximum expansion order. The default is 5000.

        Returns
        -------
        "Sab", (N, M)
            S(alpha, -beta) based on Phonon Density Of States model.

        Example
        -------
        >>> from solid_cinel.data.examples.Al27 import beta0_, alpha0_, rho_in_energy, interv_in_energy
        >>> T = 800
        >>> pdos = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> alpha = Alpha(alpha0_).scale(T)
        >>> beta = Beta(beta0_).scale(T)
        >>> S_mat = Sab.from_pdos(alpha, beta, T, pdos)
        >>> S_mat.data.round(6).iloc[:10, :5]
        beta      0.000000  0.009175  0.018350  0.027524  0.036699
        alpha
        0.001835  0.038004  0.038171  0.038333  0.038492  0.038645
        0.003670  0.074701  0.075013  0.075307  0.075590  0.075857
        0.005505  0.110103  0.110542  0.110941  0.111315  0.111663
        0.007340  0.144226  0.144776  0.145255  0.145693  0.146093
        0.009175  0.177088  0.177733  0.178272  0.178749  0.179174
        0.011010  0.208709  0.209435  0.210015  0.210509  0.210937
        0.012845  0.239108  0.239904  0.240509  0.241002  0.241412
        0.014680  0.268310  0.269164  0.269779  0.270255  0.270631
        0.016515  0.296336  0.297239  0.297853  0.298297  0.298625
        0.018350  0.323212  0.324156  0.324758  0.325158  0.325425
        """
        # Set the beta grid and the alpha grid:
        alpha_, beta_ = cls.setup_alpha_beta(alpha, beta)

        # Fix the pdos to the temperature:
        Tpdos = pdos.fix_T(T)

        # Get the Debye-Waller coefficient:
        DebyeWallerCoeff = Tpdos.DebyeWallerCoeff

        # Get the Expansion order:
        if nphonon is not None:
            warnings.warn("Is posible that the expansion order is not enough to get the correct results")
        else:
            nphonon = get_expansionOrder(alpha_, DebyeWallerCoeff, decimal, orderMax)

        # Get tauN function:
        tauN = Tpdos.tauN(nphonon, threshold=threshold, values=True)

        # Get tauN beta grid values:
        tauNbeta = get_tauNbeta(Tpdos.beta.data, tauN.shape[1])

        return cls.from_tau(alpha_, beta_, tauN, tauNbeta, DebyeWallerCoeff)

    @classmethod
    def from_tau(cls, alpha: Union[Alpha, Iterable, str], beta: Union[Beta, Iterable, str],
                 tauN: np.ndarray, tauNbeta: np.ndarray, DebyeWallerCoeff: float):
        """
        Generate S(alpha, -beta) matrix using tauN functions.
        .. math::
            S(\alpha,\,-\beta)=\exp(-\alpha\lambda)\sum_{n=0}^{\infty}\dfrac{1}{n!}(\alpha\lambda)^n\mathcal{T}_n(-\beta)

        Numerical appoximation to get convergence in large exponentiation and
        factorial numbers. Each element of the array is related with one alpha
        and represent the following term of the previous equation:
        ..math::
           \sum_{n=0}^{\infty}\dfrac{1}{n!}(\alpha\lambda)^n = \exp(\log(\dfrac{1}{1}(\alpha\lambda)) + \log(\dfrac{1}{2}(\alpha\lambda)) + ...)

        Parameters
        ----------
        alpha : 1D iterable or "Alpha", (N,)
            Alpha grid.
        beta_grid : 1D iterable or "Beta", (M,)
            beta grid.
        tauN: np.ndarray, (Z, T)
            tauN functions. The first dimension is the number of the expansion
            and the second dimension is the number of the beta grid.
        delta_beta: float
            Delta beta value.
        DebyeWallerCoeff: float
            Debye Waller coefficient.

        Returns
        -------
        "Sab", (N, M)
            S(alpha, -beta) based on Phonon Density Of States model.

        Example
        -------
        >>> from solid_cinel.data.examples.Al27 import beta0_, alpha0_, rho_in_energy, interv_in_energy
        >>> T = 800
        >>> pdos = Pdos.from_dE(T, rho_in_energy, interv_in_energy)
        >>> alpha = Alpha(alpha0_).scale(T)
        >>> beta = Beta(beta0_).scale(T)
        >>> DebyeWallerCoeff = pdos.DebyeWallerCoeff
        >>> tau1 = pdos.tau1.values
        >>> tau1beta = pdos.beta.data
        >>> nphonon = alpha.expansionOrder(DebyeWallerCoeff, 1.0e-6, 5000)
        >>> tauN = get_tauNfunc(tau1, tau1beta, nphonon, 0.0)
        >>> tauNbeta = get_tauNbeta(tau1beta, tauN.shape[1])
        >>> S_mat = Sab.from_tau(alpha, beta, tauN, tauNbeta, DebyeWallerCoeff)
        >>> S_mat.data.round(6).iloc[:10, :5]#doctest: +NORMALIZE_WHITESPACE
        beta      0.000000  0.009175  0.018350  0.027524  0.036699
        alpha
        0.001835  0.038004  0.038171  0.038333  0.038492  0.038645
        0.003670  0.074701  0.075013  0.075307  0.075590  0.075857
        0.005505  0.110103  0.110542  0.110941  0.111315  0.111663
        0.007340  0.144226  0.144776  0.145255  0.145693  0.146093
        0.009175  0.177088  0.177733  0.178272  0.178749  0.179174
        0.011010  0.208709  0.209435  0.210015  0.210509  0.210937
        0.012845  0.239108  0.239904  0.240509  0.241002  0.241412
        0.014680  0.268310  0.269164  0.269779  0.270255  0.270631
        0.016515  0.296336  0.297239  0.297853  0.298297  0.298625
        0.018350  0.323212  0.324156  0.324758  0.325158  0.325425
        """
        # Set the beta grid and the alpha grid:
        alpha_, beta_ = cls.setup_alpha_beta(alpha, beta)

        # Get the number of phonon expansion:
        nphonon = tauN.shape[0]

        # Interpolation of the tauN functions to avoid extra calculations:
        tauNinterp = interp_multyParallel(beta_, tauNbeta, tauN)

        # Get the S(alpha, -beta) matrix (alpha in matrix form to avoid using
        # outer product):
        S_values = phonon_expansion(alpha_[:, np.newaxis], nphonon, tauNinterp,
                                    DebyeWallerCoeff)

        return cls(S_values, DebyeWallerCoeff=DebyeWallerCoeff,
                   columns=beta_, index=alpha_)

    @classmethod
    def from_model(cls, *args, model: str = "pdos", **kwargs):
        """
        Create Sab object from different models. The models available are:
            - "phonon expansion": Phonon expansion model.
            - "fgm": Free Gas Model.
            - "sct": Short Collision Time model.

        Parameters
        ----------
        model : 'str'
            The model to calculate matrix values. The default is "pdos". The
            available models are:
                - "pdos": Phonon expansion model
                - "fgm" : Free Gas Model
                - "sct" : Short Collision Time model

        Parameters for FGM model:
        -------------------------
        alpha : 1D iterable or "Alpha", (N,)
            Alpha grid.
        beta_grid : 1D iterable or "Beta", (M,)
            Absolute beta grid.
        model : 'str', optional
            The model to calculate matrix values. The default is "FGM".
        wt: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Parameters for SCT model:
        -------------------------
        alpha : 1D iterable or "Alpha", (N,)
            Alpha grid.
        beta_grid : 1D iterable or "Beta", (M,)
            beta grid.
        T : 'float'
            Temperature in K.
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        ws: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Parameters for Phonon Expansion model:
        --------------------------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        T : 'float'
            Temperature in K.
        alpha : 1D iterable or "Alpha", (N,)
            Alpha grid.
        beta_grid : 1D iterable or "Beta", (M,)
            beta grid.
        threshold : 'float', optional
            Minimun value to take into account in the creation of tauN
            functions. For T>200 is convenient to set into 1.0e-14 to speed up
            the calculations. The default is 0.0.
        nphonon : 'int', optional
            Phonon expansion order. The default is 1000.

        Returns
        -------
        Sab
            S(alpha, -beta) matrix based on the chosen model.

        Examples
        --------
        >>> from solid_cinel.data.examples.Al27 import beta0_, alpha0_, rho_in_energy, interv_in_energy

        FGM:
        >>> T = 300
        >>> beta = Beta.generate_grid(T).data
        >>> alpha = Alpha.generate_grid(T, 26).data
        >>> Sab.from_model(alpha, beta, model="fgm").data.iloc[:10, :5].round(6) #doctest: +NORMALIZE_WHITESPACE
        beta	      0.000000	0.012894	0.025788	0.038682	0.051576
        alpha
        0.001050	8.701463	8.417992	7.524148	6.213536	4.740815
        0.001087	8.553363	8.285768	7.435678	6.181592	4.760714
        0.001125	8.407781	8.155251	7.346923	6.147319	4.777252
        0.001164	8.264674	8.026439	7.257961	6.110841	4.790511
        0.001205	8.124000	7.899326	7.168869	6.072279	4.800575
        0.001247	7.985718	7.773908	7.079717	6.031753	4.807533
        0.001291	7.849787	7.650178	6.990574	5.989379	4.811476
        0.001336	7.716166	7.528129	6.901504	5.945271	4.812500
        0.001382	7.584817	7.407753	6.812568	5.899540	4.810701
        0.001431	7.455701	7.289040	6.723822	5.852292	4.806177

        SCT:
        >>> pdos = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> Sab.from_model(alpha, beta, T, pdos, model="sct").data.iloc[:10, :5].round(6) #doctest: +NORMALIZE_WHITESPACE
        beta      0.000000  0.012894  0.025788  0.038682  0.051576
        alpha
        0.001050  8.342190  8.092079  7.298835  6.121534  4.773978
        0.001087  8.200211  7.964121  7.209904  6.084151  4.785744
        0.001125  8.060646  7.837859  7.120876  6.044744  4.794361
        0.001164  7.923454  7.713288  7.031821  6.003428  4.799921
        0.001205  7.788595  7.590401  6.942804  5.960320  4.802517
        0.001247  7.656028  7.469191  6.853888  5.915532  4.802243
        0.001291  7.525715  7.349649  6.765132  5.869173  4.799196
        0.001336  7.397618  7.231765  6.676593  5.821349  4.793476
        0.001382  7.271698  7.115530  6.588322  5.772162  4.785181
        0.001431  7.147919  7.000933  6.500370  5.721713  4.774412

        Phonon Expansion:
        >>> T = 800
        >>> beta = Beta(beta0_).scale(T).data
        >>> alpha = Alpha(alpha0_).scale(T).data
        >>> Sab.from_model(alpha, beta, T, pdos, model="pdos", nphonon=700).data.iloc[:10, :5].round(6) #doctest: +NORMALIZE_WHITESPACE
        beta      0.000000  0.009175  0.018350  0.027524  0.036699
        alpha
        0.001835  0.038004  0.038171  0.038333  0.038492  0.038645
        0.003670  0.074701  0.075013  0.075307  0.075590  0.075857
        0.005505  0.110103  0.110542  0.110941  0.111315  0.111663
        0.007340  0.144226  0.144776  0.145255  0.145693  0.146093
        0.009175  0.177088  0.177733  0.178272  0.178749  0.179174
        0.011010  0.208709  0.209435  0.210015  0.210509  0.210937
        0.012845  0.239108  0.239904  0.240509  0.241002  0.241412
        0.014680  0.268310  0.269164  0.269779  0.270255  0.270631
        0.016515  0.296336  0.297239  0.297853  0.298297  0.298625
        0.018350  0.323212  0.324156  0.324758  0.325158  0.325425
        """
        if model.lower() == "fgm":
            return cls.from_fgm(*args, **kwargs)
        elif model.lower() == "sct":
            return cls.from_sct(*args, **kwargs)
        elif model.lower() == "pdos":
            return cls.from_pdos(*args, **kwargs)

    @classmethod
    def from_alpha0(cls, Ein: [int, float, np.ndarray], T: float, M: float,
                    beta: Union[Beta, Iterable], *args,
                    model: str = "pdos", **kwargs):
        """
        Generate S(alpha, -beta) matrix using gressier recoil energy alpha grid.

        Parameters
        ----------
        Ein: 'float' or 'np.ndarray'
            Incident neutron energy in eV.
        T: 'float'
            Temperature in K.
        M: 'float'
            Mass of the target in amu.
        beta: 'Beta' or 'Iterable'
            Beta grid.
        model: 'str'
            Model to calculate the S(alpha, -beta) matrix.

        Parameters for FGM model:
        -------------------------
        wt: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Parameters for SCT model:
        -------------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        ws: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Parameters for Phonon Expansion model:
        --------------------------------------
        threshold : 'float', optional
            Minimun value to take into account in the creation of tauN
            functions. For T>200 is convenient to set into 1.0e-14 to speed up
            the calculations. The default is 0.0.
        nphonon : 'int', optional
            Phonon expansion order. The default is 1000.

        Returns
        -------
        "Sab", (N, M)
            S(alpha, -beta) matrix.

        Example
        -------
        >>> from solid_cinel.data.examples.Al27 import rho_in_energy, interv_in_energy
        >>> T = 300
        >>> beta = Beta.generate_grid(T).data
        >>> Ein = np.array([6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0])
        >>> M = 26
        >>> Sab.from_alpha0(Ein, T, M, beta, model="fgm").data.iloc[::, :5].round(6)
        beta       0.000000  0.012894  0.025788  0.038682  0.051576
        alpha
        9.045004   0.009776  0.009839  0.009902  0.009966  0.010030
        9.189465   0.009354  0.009415  0.009476  0.009537  0.009598
        9.333925   0.008953  0.009010  0.009069  0.009127  0.009186
        9.478386   0.008569  0.008624  0.008680  0.008736  0.008792
        9.622847   0.008203  0.008256  0.008309  0.008363  0.008416
        9.767307   0.007853  0.007904  0.007955  0.008006  0.008058
        9.911768   0.007519  0.007568  0.007616  0.007666  0.007715
        10.056229  0.007200  0.007247  0.007293  0.007340  0.007388

        SCT:
        >>> pdos = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> Sab.from_alpha0(Ein, T, M, beta, pdos, model="sct").data.iloc[::, :5].round(6)
        beta       0.000000  0.012894  0.025788  0.038682  0.051576
        alpha
        9.045004   0.011253  0.011320  0.011387  0.011455  0.011522
        9.189465   0.010800  0.010864  0.010929  0.010993  0.011058
        9.333925   0.010366  0.010428  0.010490  0.010552  0.010614
        9.478386   0.009951  0.010010  0.010070  0.010129  0.010189
        9.622847   0.009554  0.009610  0.009667  0.009725  0.009782
        9.767307   0.009173  0.009228  0.009282  0.009337  0.009393
        9.911768   0.008809  0.008861  0.008914  0.008966  0.009019
        10.056229  0.008460  0.008510  0.008560  0.008611  0.008662

        Phonon Expansion:
        >>> Sab.from_alpha0(Ein, T, M, beta, pdos, model="pdos").data.iloc[::, :5].round(6)
        beta       0.000000  0.012894  0.025788  0.038682  0.051576
        alpha
        9.045004   0.010582  0.010650  0.010719  0.010788  0.010857
        9.189465   0.010132  0.010198  0.010264  0.010330  0.010396
        9.333925   0.009703  0.009766  0.009829  0.009893  0.009956
        9.478386   0.009294  0.009354  0.009414  0.009475  0.009536
        9.622847   0.008903  0.008960  0.009018  0.009076  0.009135
        9.767307   0.008529  0.008584  0.008639  0.008695  0.008751
        9.911768   0.008172  0.008225  0.008278  0.008331  0.008385
        10.056229  0.007830  0.007881  0.007932  0.007983  0.008034
        """
        alpha = Alpha.from_recoil(Ein, T, M)
        return cls.from_model(alpha, beta, T, *args, model=model, **kwargs)

    @staticmethod
    def SumRule_check(S: pd.DataFrame) -> None:
        """
        Check if the S(alpha, beta) matrix satifies the sum rule constrain.
        .. math::
            \int_{-\infty}^{\infty}\beta S(\alpha,\,\beta)d\beta = \int_{0}^{\infty}\beta S(\alpha,\,-\beta)(1-\exp(-\beta))d\beta = \alpha

        For SCT and for phonon expansion, the normalization is only satisfy to
        large alpha values, for the rest:
        .. math::
            \int_{0}^{\infty}\beta S(\alpha,\,-\beta)(-1+\exp(-\beta))d\beta = -\alpha

        Parameters
        ----------
        S : 'pd.DataFrame', (N, M)
            S(alpha, beta) matrix.

        Returns
        -------
        "None"
            If the sum rule is not satisfied with good accuracy a warning is
            raise. If the accuracy is very low, a ValueError will be raise.

        Raises
        ------
        ValueError
            The sum rule constrain is not satified.

        """
        # Check the sum rule:
        SumRule = S.apply(_SumRule, axis="columns")
        SumRule /= S.index.values
        if (abs(1 - abs(SumRule)) > 0.6).any():
            raise ValueError("Sum rule of S(alpha, -beta) not satisfied")
        if (abs(1 - abs(SumRule)) > 1.0e-3).any():
            warnings.warn(
                "Sum rule of S(alpha, -beta) not satisfied with an precision of 1.0e-3")
        return

    def NormCheck(self, S: pd.DataFrame) -> None:
        """
        Check if the S(alpha, beta) matrix satifies the normalization constrain.
        .. math::
            \int_{-\infty}^{\infty}S(\alpha,\,\beta)d\beta = \int_{0}^{\infty}S(\alpha,\,-\beta)(1+\exp(-\beta))d\beta = 1

        For SCT and for phonon expansion, the normalization is only satisfy to
        large alpha values, for the rest:
        .. math::
            \int_{0}^{\infty}S(\alpha,\,-\beta)(1+\exp(-\beta))d\beta = 1 - \exp(-\alpha\lambda)

        Parameters
        ----------
        S : 'pd.DataFrame', (N, M)
            S(alpha, beta) matrix.

        Returns
        -------
        "None"
            If the normalization is not satisfied with good accuracy a warning
            is raise. If the accuracy is very low, a ValueError will be raise.

        Raises
        ------
        ValueError
            The normalization constrain is not satified.

        """
        # Check the normalization:
        norm = S.apply(_norm, axis="columns")
        norm /= 1 - np.exp(- S.index.values * self.DebyeWallerCoeff)
        if (abs(norm - 1.0) > 1.0e-2).any():
            warnings.warn(
                "Normalization of S(alpha, -beta) not satisfied with an precision of 1.0e-2")
        return

    def update_data(self, sabNew: pd.DataFrame, inplace: bool, axis: int):
        """
        Update the S(alpha, -beta) matrix with new values.

        Parameters
        ----------
        sabNew : "pd.DataFrame"
            New S(alpha, -beta) matrix.
        inplace : "bool"
            If True, the S(alpha, -beta) matrix is updated in place. If False,
            a new Sab object is returned.
        axis : "int", optional
            Axis to concatenate the new values. The default is 1.

        Returns
        -------
        "None" or "Sab"
            If inplace is True, the S(alpha, -beta) matrix is updated in place.
            If False, a new Sab object is returned.
        """
        # Concatenate the S(alpha, -beta) matris:
        dataNew = pd.concat([self.data, sabNew], axis=axis)

        # Sort the concatenate axis:
        dataNew = dataNew.sort_index(axis=axis)

        # Update the S(alpha, -beta) matrix:
        if inplace:
            self.data = dataNew
            return self
        else:
            return Sab(dataNew)

    def interp_beta(self, betaNew: Union[Iterable, float], inplace: bool = False) -> pd.DataFrame:
        """
        Quadratic interpolation to get the probability of the new beta value
        for all the alpha existing in the S(alpha, -beta) matrix:
        .. math::
            \left\{ \mid\beta_{new}\mid = P\left(\mid\betaNew\mid, \alpha_k\right)\right\} \text{ for }k=0, 1, ...

        The method do not make extrapolation.

        Parameters
        ----------
        betaNew : "float" or 1D iterable
            New beta values.
        inplace : "bool", optional
            Optional argument to add the output to the existing S(alpha, -beta)
            matrix or only get the pd.Dataframe. The default is False.

        Returns
        -------
        "pd.Dataframe" or "Sab"
            pd.Dataframe with the requested probabilities for the new betas for
            all the existing alpha. If add is True, the pd.Dataframe is merge
            in the S(alpha, -beta) matrix.

        Example
        -------
        >>> from solid_cinel.data.examples.Al27 import beta0_, alpha0_, rho_in_energy, interv_in_energy
        >>> T = 300
        >>> pdos = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> alpha = Alpha(alpha0_).scale(T)
        >>> beta = Beta(beta0_).scale(T)
        >>> S_mat = Sab.from_pdos(alpha, beta, T, pdos, threshold=1.0e-14)
        >>> betaNew = 0.01
        >>> S_mat.interp_beta(betaNew).data.iloc[1, 0:10].round(6)
        beta
        0.000000    0.010712
        0.010000    0.010765
        0.024466    0.010842
        0.048932    0.010972
        0.073399    0.011103
        0.097865    0.011232
        0.122331    0.011355
        0.146797    0.011491
        0.171263    0.011625
        0.195730    0.011759
        Name: 0.009786476949338778, dtype: float64

        >>> betaNew = [0.01, 0.03]
        >>> S_mat.interp_beta(betaNew, inplace=True).data.iloc[0:10, 0:4] #doctest: +NORMALIZE_WHITESPACE
        beta      0.000000  0.010000  0.024466  0.030000
        alpha
        0.004893  0.005396  0.005423  0.005462  0.005477
        0.009786  0.010712  0.010765  0.010842  0.010872
        0.014680  0.015948  0.016027  0.016140  0.016184
        0.019573  0.021104  0.021208  0.021357  0.021414
        0.024466  0.026181  0.026309  0.026493  0.026563
        0.029359  0.031179  0.031331  0.031549  0.031632
        0.034253  0.036100  0.036275  0.036526  0.036621
        0.039146  0.040943  0.041141  0.041423  0.041530
        0.044039  0.045710  0.045929  0.046243  0.046361
        0.048932  0.050400  0.050641  0.050984  0.051114
        """
        # Interpolation arguments:
        interpArgs = {"bounds_error": True, "kind": "quadratic"}

        # New beta values in the appropriate format:
        betaNew_ = betaNew if hasattr(betaNew, '__len__') else [betaNew]

        # Interpolation of the new beta values:
        betaValues = self.data.apply(lambda x: reshape_differential(x, betaNew_, **interpArgs), axis=1)

        # DataFrame construction with the new beta values:
        beta_df = pd.DataFrame.from_records(betaValues.values,
                                            index=betaValues.index,
                                            columns=pd.Index(betaNew_, name="beta"))

        # Add the new beta values to the S(alpha, -beta) matrix or return:
        return self.update_data(beta_df, inplace, 1)

    def interp_alpha(self, alphaNew: Union[Iterable, float],
                     inplace: bool = False) -> pd.DataFrame:
        """
        Unit base interpolation to get the probability of the new alpha values
        for all the beta existing in the S(alpha, -beta) matrix.
        .. math::
            \left\{ \mid\alpha_{new}\mid = P\left(\mid\beta_{k}\mid, \alpha_{new}\right) \text{ for }k=0, 1, ...

        The method do not make extrapolation.

        Parameters
        ----------
        alphaNew : "float" or 1D iterable
            New alpha values.
        add : "bool", optional
            Optional argument to add the output to the existing S(alpha, -beta)
            matrix or only get the pd.Dataframe. The default is False.

        Returns
        -------
        "pd.Dataframe" or "Sab"
            pd.Dataframe with the requested probabilities for the new alpha for
            all the existing alpha. If add is True, the pd.Dataframe is merge
            in the S(alpha, -beta) matrix.

        Example
        -------
        >>> from solid_cinel.data.examples.UO2 import rho_in_energy_U238, interv_in_energy_U238, alpha0_U238, beta0_U238
        >>> T = 300
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> beta_grid = Beta(beta0_U238).scale(T)
        >>> alpha = Alpha(alpha0_U238).scale(T)
        >>> S_mat = Sab.from_pdos(alpha, beta_grid, T, pdos, threshold=1.0e-14)
        >>> betaTest = beta_grid.data[0:5]
        >>> alphaNew = 0.00013
        >>> S_mat.interp_alpha(alphaNew).data.loc[alphaNew, betaTest] #doctest: +NORMALIZE_WHITESPACE
        beta
        0.000000    0.000484
        0.025237    0.000490
        0.050474    0.000453
        0.075712    0.000448
        0.100949    0.000449
        Name: 0.00013, dtype: float64

        >>> alphaNew = [1.25e-4, 1.35e-4]
        >>> S_mat.interp_alpha(alphaNew).data.loc[alphaNew, betaTest] #doctest: +NORMALIZE_WHITESPACE
        beta      0.000000  0.025237  0.050474  0.075712  0.100949
        alpha
        0.000125  0.000465  0.000471  0.000436  0.000431  0.000432
        0.000135  0.000502  0.000508  0.000471  0.000466  0.000467

        >>> alphaNew = [1.25e-4, 1.35e-4]
        >>> S_mat.interp_alpha(alphaNew, inplace=True).data.iloc[0:5, 0:5] #doctest: +NORMALIZE_WHITESPACE
        beta      0.000000  0.025237  0.050474  0.075712  0.100949
        alpha
        0.000112  0.000418  0.000423  0.000392  0.000387  0.000388
        0.000120  0.000447  0.000453  0.000420  0.000415  0.000416
        0.000125  0.000465  0.000471  0.000436  0.000431  0.000432
        0.000129  0.000479  0.000485  0.000449  0.000444  0.000445
        0.000135  0.000502  0.000508  0.000471  0.000466  0.000467
        """
        # New alpha values in the appropriate format:
        alphaNew_ = self.check_alpha(alphaNew)

        # Interpolation of the new alpha values:
        alphaVec = []
        for new_alpha in alphaNew_.data:
            alphaVec.append(self._interp_alpha(new_alpha))

        # DataFrame construction with the new alpha values:
        alphaNew_df = pd.concat(alphaVec, axis=1).T

        # Add the new alpha values to the S(alpha, -beta) matrix or return:
        return self.update_data(alphaNew_df, inplace, 0)

    def _interp_alpha(self, alphaNew: float) -> pd.DataFrame:
        """
        Interpolate S(alpha, -beta) using unit base interpolation to get the
        probabilities for the new alpha values:
        .. math::
            \left\{ \mid\alpha_{new}\mid = P\left(\mid\beta_{k}\mid, \alpha_{new}\right) \text{ for }k=0, 1, ...

        The method do not make extrapolation.

        Parameters
        ----------
        alphaNew : "float"
            Alpha value to get interpolated.

        Returns
        -------
        "pd.Series"
            Interpolated beta grid vector for the introduced alpha.

        Example
        -------
        >>> from solid_cinel.data.examples.UO2 import rho_in_energy_U238, interv_in_energy_U238, alpha0_U238, beta0_U238
        >>> T = 300
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> beta_grid = Beta(beta0_U238).scale(T)
        >>> alpha = Alpha(alpha0_U238).scale(T)
        >>> S_mat = Sab.from_pdos(alpha, beta_grid, T, pdos, threshold=1.0e-14)
        >>> alphaNew = 0.00013
        >>> alphaVec = S_mat._interp_alpha(alphaNew)
        >>> alphaVec.iloc[0:10]  #doctest: +NORMALIZE_WHITESPACE
        alpha      0.00013
        beta
        0.000000  0.000484
        0.025237  0.000490
        0.050474  0.000453
        0.075712  0.000448
        0.100949  0.000449
        0.126186  0.000459
        0.151423  0.000476
        0.176660  0.000495
        0.201898  0.000513
        0.227135  0.000537

        Check the contrains:
        >>> debyeWeller = pdos.DebyeWallerCoeff(T)
        >>> float(round(integrate(alphaVec.iloc[::, 0] * (1 + np.exp(-beta_grid.data))) / (1 - np.exp(-debyeWeller * alphaNew)), 6))
        1.005782

        >>> float(round(integrate(alphaVec.iloc[::, 0] * beta_grid.data * (1 -  np.exp( - beta_grid.data))), 6))
        0.000131
        """
        # Check if the new alpha values are already in the S(alpha, -beta) matrix:
        alpha = self.alpha.data
        if alphaNew > alpha.max():
            raise SyntaxError(r"alpha out of range($\alpha_{max}=$"
                              + str(alpha.max()) + ")")
        elif alphaNew < alpha.min():
            raise SyntaxError(r"alpha out of range($\alpha_{min}=$"
                              + str(alpha.min()) + ")")
        elif alphaNew in alpha:
            return self.data.loc[alphaNew]

        # Get the alpha values to perform the interpolation:
        upper_bound = alpha.searchsorted(alphaNew, side="right")
        alpha0 = alpha[upper_bound - 1]
        alpha2 = alpha[upper_bound]
        prob = self.data.loc[[alpha0, alpha2]].T

        # Normalization of the alpha rows in the S(alpha, -beta) matrix:
        if hasattr(self, "DebyeWallerCoeff"):
            debyeWeller = self.DebyeWallerCoeff
            probNorm = prob.apply(lambda x: (1 + np.exp(-x.index)) * x / (
                    1 - np.exp(-debyeWeller * x.name)))
        else:
            probNorm = prob.apply(lambda x: (1 + np.exp(-x.index)) * x)

        # Interpolation of the new alpha values:
        q = proportionality_factor(alphaNew, alpha0, alpha2, mode="linlog")
        alphaNew_escale = (1 - q) * probNorm.loc[::, alpha0] + q * probNorm.loc[::, alpha2]

        # Undo the normalization of the new alpha values:
        alphaNewVec = alphaNew_escale / (1 + np.exp(-self.beta.data))
        if hasattr(self, "DebyeWallerCoeff"):
            alphaNewVec *= (1 - np.exp(- debyeWeller * alphaNew))

        # Return the new alpha values:
        columns = pd.Index([alphaNew], name="alpha")
        return pd.DataFrame(alphaNewVec, columns=columns)

    def interp_alphaBeta(self, alpha: Union[Iterable, float, str],
                         beta: Union[Iterable, float, str]) -> pd.DataFrame:
        """
        Get intepolated values for the beta and alpha values from the
        S(alpha, beta) matrix. This method take into account the sing of the
        beta that is introduced.
        .. math::
            Beta < 0:
                \left\{ \mid\alpha\mid = P\left(- \mid\beta\mid, \alpha\right)
            Beta > 0:
                \left\{ \mid\alpha\mid = P\left(-\mid\beta\mid, \alpha\right) * exp(-\mid\beta_{new}\mid)

        Parameters
        ----------
        alpha : "float" or 1D iterable
            Alpha values to interpolate.
        beta : "float" or 1D iterable
            Beta values to interpolate.

        Returns
        -------
        "pd.Dataframe"
            Interpolated S(alpha, beta)

        Example
        -------
        >>> from solid_cinel.data.examples.UO2 import rho_in_energy_U238, interv_in_energy_U238, alpha0_U238, beta0_U238
        >>> T = 300
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> beta_grid = Beta(beta0_U238).scale(T)
        >>> alpha = Alpha(alpha0_U238).scale(T)
        >>> S_mat = Sab.from_pdos(alpha, beta_grid, T, pdos, threshold=1.0e-14)
        >>> alphaNew = [1.25e-4, 1.35e-4]
        >>> betaNew = [0.01, 0.03, -0.01, -0.03]
        >>> S_mat.interp_alphaBeta(alphaNew, betaNew) #doctest: +NORMALIZE_WHITESPACE
        beta         -0.03     -0.01      0.01      0.03
        alpha
        0.000125  0.000466  0.000474  0.000469  0.000452
        0.000135  0.000503  0.000512  0.000506  0.000488
        """
        # New alpha values in the appropriate format:
        alpha_ = self.check_alpha(alpha)

        # New beta values in the appropriate format:
        beta_ = np.array(beta) if hasattr(beta, '__len__') else np.array([beta])
        betaUnique = np.unique(abs(beta_))

        # Interpolation of the new alpha and beta values:
        interp_AlphaBeta = self.interp_alpha(alpha_).interp_beta(betaUnique)

        return interp_AlphaBeta.full.loc[alpha_.data, np.unique(beta_)]


def _SumRule(x: pd.Series) -> float:
    """
    Calculate the "n" sum rule value for a fix alpha value.
    .. math::
        \int_{-\infty}^{\infty}\beta^n S(\alpha,\,\beta)d\beta = \int_{0}^{\infty}\beta^n S(\alpha,\,-\beta)(1-\exp(-\beta))d\beta
    Parameters
    ----------
    x : 'pd.Series', (N)
        S(alpha, beta) matrix values for fix alpha.
    n: 'int', optional
        The number of the SumRule

    Returns
    -------
    "float"
        Sum rule value for a fix alpha.

    Example
    -------
    >>> beta_grid = Beta.generate_grid(300)
    >>> alpha = Alpha.generate_grid(300, 26)
    >>> s = Sab.from_fgm(alpha, beta_grid).data
    >>> float(_SumRule(s.iloc[1, ::]).round(6))
    0.001087
    """
    beta = x.index.values
    return integrate(beta * x * (1 - np.exp(- beta)))


def _norm(x: pd.Series) -> float:
    """
    Normalization rule value for a fix alpha value of the S(alpha, beta) matrix.

    Parameters
    ----------
    x : 'pd.Series', (N,)
        S(alpha, beta) matrix values for fix alpha.

    Returns
    -------
    normalization_values : "float"
        Normalization value for a fix alpha.

    Example
    -------
    >>> beta_grid = Beta.generate_grid(300)
    >>> alpha = Alpha.generate_grid(300, 26)
    >>> s = Sab.from_fgm(alpha, beta_grid).data
    >>> float(_norm(s.iloc[0, ::]).round(6))
    1.0
    """
    beta = x.index.values
    return integrate((1 + np.exp(- beta)) * x)


def proportionality_factor(alpha: float, alpha_i: float,
                           alpha_i_plus1: float,
                           mode: str = "linlog") -> float:
    """
    Get the proportionality factor for unit-base interpolation.

    Parameters
    ----------
    alpha : "float"
        Alpha value to be interpolated.
    alpha_i : "float"
        lower alpha bound.
    alpha_i_plus1 : "float"
        upper alpha bound.
    mode : "str", optional
        Is define by how the probability table is interpolated. The default is
        "linlog".

    Returns
    -------
    "float"
        Proportionality factor for unit-base interpolation
    """
    if mode == "linlog":
        q = np.log(alpha / alpha_i) / np.log(alpha_i_plus1 / alpha_i)
    elif mode == "linlin":
        q = (alpha - alpha_i) / (alpha_i_plus1 - alpha_i)
    elif mode == "const":
        q = 1
    return q


@nb.jit(float64[:, :](float64[:, :], int32, float64[:, :], float64),
        nopython=True, cache=True)
def phonon_expansion(alpha: np.ndarray, nphonon: int, tauNinterp: np.ndarray,
                     DebyeWallerCoeff: float) -> np.ndarray:
    """
    Generate S(alpha, -beta) matrix using tauN functions:
    .. math::
        S(\alpha,\,-\beta)=\exp(-\alpha\lambda)\sum_{n=0}^{\infty}\dfrac{1}{n!}(\alpha\lambda)^n\mathcal{T}_n(-\beta)

    Parameters
    ----------
    alpha: 'xp.ndarray', (N,)
        alpha grid values in the cpu as numpy array or in the gpu as cupy array.
    beta: 'xp.ndarray', (M,)
        beta grid values in the cpu as numpy array or in the gpu as cupy array.
    nphonon: 'int'
        Number of phonon expansion.
    tauN: 'xp.ndarray', (Z, T)
        tauN functions. The first dimension is the number of the expansion
        and the second dimension is the number of the beta grid. In the cpu as
        numpy array or in the gpu as cupy array.
    tauNbeta: 'np.ndarray', (T,)
        Beta values of the tauN functions.
    DebyeWallerCoeff: 'float'
        Debye Waller coefficient.

    Returns
    -------
    'xp.ndarray', (N, M)
        S(alpha, -beta) matrix values.
    """
    # Common variables:
    alphaDebye = alpha * DebyeWallerCoeff

    # Zero phonon expansion:
    IterSum = np.log(alphaDebye)
    sabValues = alphaDebye * tauNinterp[0]

    # Higher phonon expansion (nphonon >= 1):
    for n in range(1, nphonon):
        # Compute S(alpha, -beta) for tauN reshape
        IterSum += np.log(alphaDebye / (n + 1))
        sabValues += np.exp(IterSum) * tauNinterp[n]
    return np.exp(- alphaDebye) * sabValues


@nb.jit(nopython=True, cache=True)
def get_SabSct(alpha: np.ndarray, beta: np.ndarray, Tratio: float,
               ws: float) -> np.ndarray:
    """
    Generate S(alpha, beta) matrix using Short Collision Time:
    .. math::
        S(\alpha, \beta)=\dfrac{1}{\sqrt{4\pi\omega_{s}\alpha T_{\textrm{eff}}/T}}\exp\left(-\dfrac{(\mid\beta\mid - \omega_{s}\alpha)^2}{4\omega_{s}\alpha T_{\textrm{eff}}/T} - \frac{\mid\beta\mid - \beta}{2}\right)

    Parameters
    ----------
    alpha : 'np.ndarray', (N,)
        alpha grid values.
    beta : 'np.ndarray', (M,)
        beta grid values.
    Tratio : "float"
        Effective temperature divide by the temperature.
    ws: 'float', optional
        normalization for continuous (vibrational) part. For solid is 1.

    Returns
    -------
    'np.ndarray', (N, M)
        S(alpha, beta) matrix values.
    """
    alphaCommon = 4 * alpha * Tratio * ws
    sabValues = np.exp(-(ws * alpha + beta) ** 2 / alphaCommon)
    return sabValues / np.sqrt(pi * alphaCommon)
