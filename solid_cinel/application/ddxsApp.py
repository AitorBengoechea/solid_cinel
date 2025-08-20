import argparse
import numpy as np
from solid_cinel.application.pdosApp import get_Pdos
from solid_cinel.core.xs.xs0K import Xs0K
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
    parser.add_argument('model', type=str.lower,
                        choices=['fgm', 'sct', 'pdos'],
                        help='Model to use for the calculation of the algorithm')
    parser.add_argument('xs0K', type=str,
                        help='Cross section at 0 K')
    parser.add_argument('Ein', type=float,
                        help='Incident energy in eV')
    parser.add_argument('M', type=float,
                        help='Mass of the target atom in a.m.u.')
    parser.add_argument('T', type=float,
                        help='Temperature in K')
    parser.add_argument('Eout', type=str,
                        help='Grid for the output energy in eV')
    parser.add_argument('theta', type=str,
                        help='Grid for the scattering angle in degrees')
    parser.add_argument('--output', type=str, nargs='+',
                        choices=['ddxs', 'scatFunc', 'angleDistr'],
                        default=['ddxs'],
                        help='What to return: ddxs, scatFunc, angleDistr')


def get_DDxs(args: argparse.Namespace) -> DDxs:
    """
    Handle the arguments for the calculation of the double differential scattering
    cross section.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    DDxs
        The double differential scattering cross section.
    """
    # Read the data from files:
    theta, Eout = np.loadtxt(args.theta), np.loadtxt(args.Eout)

    # Initialize the Xs class with 0K cross section data
    xs = Xs0K.from_file(args.xs0K, args.M)

    # Get the extra arguments for Pdos
    argsPdos = [get_Pdos(args)] if args.model != "fgm" else []

    # Compute the function:
    return DDxs.from_4PCF(xs, args.Ein, args.T, Eout, theta, *argsPdos,
                          model=args.model)


def handle_DDxsArgs(args: argparse.Namespace) -> dict:
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
    results = {}
    # Compute the double differential scattering cross section:
    ddxs = get_DDxs(args)

    # Return the values of the double differential scattering cross section:
    if 'ddxs' in args.output:
        results["values"] = ddxs.values
    if 'scatFunc' in args.output:
        results["scatFunc"] = ddxs.scatFunc.values
    if 'angleDistr' in args.output:
        results["angleDistr"] = ddxs.angleDistr.values

    # Return the results as a dictionary:
    return results
