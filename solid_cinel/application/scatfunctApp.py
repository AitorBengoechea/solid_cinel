import argparse
import numpy as np
from solid_cinel.application.pdosApp import get_Pdos
from solid_cinel.core.scattering_function.scatfunc import ScatFunc, TransferFunc


def str_or_float(value):
    try:
        return float(value)
    except ValueError:
        return value


def add_ScatFuncArgs(parser: argparse.ArgumentParser):
    """
    Add arguments to the parser for the calculation of the effective temperature.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the arguments should be added.
    """
    parser.add_argument('model', type=str,
                        help='Model to use for the calculation of the S(alpha, -beta) table')
    parser.add_argument('Ein', type=float,
                        help='incident energy in eV')
    parser.add_argument('M', type=float,
                        help='mass of the target atom in a.m.u.')
    parser.add_argument('T', type=float,
                        help='temperature in K')
    parser.add_argument('Eout', type=str,
                        help='Grid for the output energy in eV')
    parser.add_argument('theta', type=str_or_float,
                        help='Grid for the scattering angle in degrees')


def handle_ScatFuncArgs(args: argparse.Namespace) -> np.array:
    """
    Handle the arguments for the calculation of the effective temperature.

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
    theta = np.loadtxt(args.theta) if isinstance(args.theta, str) else args.theta
    Eout = np.loadtxt(args.Eout)

    # If theta is a single value, use the TransferFunc class -> 1D array
    # Otherwise, use the ScatFunc class -> 2D array
    method = TransferFunc.from_theta if isinstance(theta, (int, float)) else ScatFunc.from_model

    # Get the extra arguments for Pdos
    argsPdos = [get_Pdos(args)] if args.model != "fgm" else []

    # Compute the function:
    scatfunc = method(args.Ein, args.M, args.T, Eout, theta, *argsPdos,
                      model=args.model)

    # Return the values of the scattering function
    return scatfunc.data.values