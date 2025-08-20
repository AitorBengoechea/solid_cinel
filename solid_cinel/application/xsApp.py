import argparse
import numpy as np
import pandas as pd
from scipy.constants import physical_constants as const

# Import from solid_cinel:
from solid_cinel import Pdos, AlphaBase, Beta, Xs0K, Sab
from solid_cinel.core.generic import integrate, reshape_differential

# Import from application:
from solid_cinel.application.pdosApp import get_Pdos


kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]


def add_XsArgs(parser: argparse.ArgumentParser):
    """
    Add arguments to the parser for the calculation of double differential
    scattering cross section.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the arguments should be added.
    """
    parser.add_argument('model', type=str.lower,
                        choices=['4pcf', 'alpha0'],
                        help='Model to use for the calculation of the algorithm')
    parser.add_argument('xs0K', type=str,
                        help='Cross section at 0 K')
    parser.add_argument('Ein', type=str,
                        help='Incident energy in eV')
    parser.add_argument('M', type=float,
                        help='Mass of the target atom in a.m.u.')
    parser.add_argument('T', type=float,
                        help='Temperature in K')
    parser.add_argument('--nphonon', type=int,
                        default=None,
                        help='Number of phonon')
    parser.add_argument('--theta', type=str, nargs='+',
                        help='Grid for the scattering angle in degrees')


def calc_alpha0(xs0K: pd.Series, Ein: np.ndarray, M: float, T: float,
                pdos: Pdos, nphonon = None) -> np.ndarray:
    """
    Calculate the XS using alpha0 model asymtotic value.

    Parameters
    ----------
    xs0K : pd.Series
        The cross section at 0 K.
    Ein : np.ndarray
        The incident energy in eV.
    M : float
        The mass of the target atom in a.m.u.
    T : float
        The temperature in K.
    pdos :  Pdos
        The phonon density of states object.
    nphonon : int, optional
        The number of phonon modes to consider. If None, it will be calculated
        based on the maximum alpha value. Default is None.

    Returns
    -------
    np.ndarray
        The Xs values.
    """
    # Get the temperature dependent pdos and the Debye Waller coefficient:
    pdos_ = pdos.get_Tpdos(T)
    DebyeWallerCoeff = pdos_.DebyeWallerCoeff

    # Get the Expansion order:
    if nphonon is None:
        alphaMax = m / (m + M) * (Ein[-1]) / (kb * T)
        nphonon = AlphaBase(alphaMax).expansionOrder(DebyeWallerCoeff, 1.0e-6, 7000)

    print(nphonon)

    # Calculate the recoil and the alpha values:
    recoil = m / (m + M) * (Ein)
    alpha = recoil / (kb * T)

    # Get the maximun order get the tauN functions:
    tauNdf = pdos_.tauN(nphonon, 0.0)
    tauN, tauNbeta = tauNdf.values, tauNdf.columns.values

    # Get the scattering function and calculate the cross section:
    beta = Beta.from_default(T).data
    scatfunc = Sab.from_tau(alpha, beta, tauN, tauNbeta, DebyeWallerCoeff).full

    # Get the outgoing energy grid:
    EoutCalc = scatfunc.columns.values * kb * T + (Ein + recoil)[::, np.newaxis]

    # Integrate the cross section:
    xsDb = (scatfunc * reshape_differential(xs0K, EoutCalc)).apply(integrate,
                                                                   axis=1)
    return xsDb


def get_Xs(args: argparse.Namespace) -> np.ndarray:
    """
    Handle the arguments for the calculation of the double differential scattering
    cross section.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    np.ndarray
        The Xs values.
    """
    # Get the extra arguments for Pdos
    argsPdos = get_Pdos(args)

    # Initialize the Xs class with 0K cross section data
    xs = Xs0K.from_file(args.xs0K, args.M)

    Ein = np.loadtxt(args.Ein)

    if args.model == "4pcf":
        pass
    elif args.model == "alpha0":
        # Calculate the Xs using alpha0 model asymtotic value:
        return calc_alpha0(xs.data, Ein, args.M, args.T, argsPdos,
                           nphonon=args.nphonon)
    else:
        raise ValueError(f"Unknown model: {args.model}")


def handle_XsArgs(args: argparse.Namespace) -> dict:
    """
    Handle the arguments for the calculation of the double differential scattering
    cross section.

    Parameters
    ----------
    args: argparse.Namespace
        The parsed arguments.

    Returns
    -------
    np.ndarray
        An array containing the double differential scattering cross section
        values.
    """
    # Return the results as a dictionary:
    return {"values": get_Xs(args)}
