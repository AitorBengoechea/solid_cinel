import argparse
import numpy as np
from solid_cinel.application.pdosApp import get_Pdos
from solid_cinel.core.xs.xs import Xs
from solid_cinel.core.xs.ddxs import DDxs


def add_DDxsArgs(parser: argparse.ArgumentParser):
    """
    Add arguments to the parser for the calculation of double differential
    scattering cross section.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the arguments should be added.
    """
    parser.add_argument('algorithm', type=str,
                        help='Algorithm to use for the calculation of the DDxs: Sab or 4PCF')
    parser.add_argument('model', type=str,
                        help='Model to use for the calculation of the algorithm')
    parser.add_argument('xs0K', type=str,
                        help='Cross section at 0 K')
    parser.add_argument('Ein', type=float,
                        help='incident energy in eV')
    parser.add_argument('M', type=float,
                        help='mass of the target atom in a.m.u.')
    parser.add_argument('T', type=float,
                        help='temperature in K')
    parser.add_argument('Eout', type=str,
                        help='Grid for the output energy in eV')
    parser.add_argument('theta', type=str,
                        help='Grid for the scattering angle in degrees')

def check_algirithm(algorithm: str):
    """
    Check if the algorithm is valid.

    Parameters
    ----------
    algorithm : str
        The algorithm to check.

    Returns
    -------
    method
        The method to use for the calculation.
    """
    if algorithm.lower() == "sab":
        return DDxs.from_Sab
    elif algorithm.lower() == "4pcf":
        return DDxs.from_4PCF
    else:
        raise ValueError(f'Invalid algorithm: {algorithm}')


def handle_DDxsArgs(args: argparse.Namespace) -> np.array:
    """
    Handle the arguments for the calculation of the double differential scattering
    cross section.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    np.array
        An array containing the input temperature and the calculated effective temperature.
    """
    # Read the data from files:
    theta, Eout = np.loadtxt(args.theta), np.loadtxt(args.Eout)

    # Initialize the Xs class with 0K cross section data
    xs = Xs.from_xs0K(args.xs0K, args.M)

    # Define the method to use
    method = check_algirithm(args.algorithm)

    # Get the extra arguments for Pdos
    argsPdos = [get_Pdos(args)] if args.model != "fgm" else []

    # Compute the function:
    ddxs = method(xs, args.Ein, args.T, Eout, theta, *argsPdos, model=args.model)

    # Return the values of the scattering function
    return ddxs.data.values
