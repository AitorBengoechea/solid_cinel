import argparse
import numpy as np
from solid_cinel.application.pdosApp import get_Pdos
from solid_cinel.core.dynamic_structure.dynamicStruc import DynamicStruc


def str_or_float(value):
    try:
        return float(value)
    except ValueError:
        return value


def add_DynamicStrucArgs(parser: argparse.ArgumentParser):
    """
    Add arguments to the parser for the calculation of the scattering function.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the arguments should be added.
    """
    parser.add_argument('model', type=str.lower,
                        choices=['fgm', 'sct', 'pdos'],
                        help='Model to use for the calculation of the scattering function')
    parser.add_argument('Ein', type=float,
                        help='Incident energy in eV')
    parser.add_argument('M', type=float,
                        help='Mass of the target atom in a.m.u.')
    parser.add_argument('T', type=float,
                        help='Temperature in K')
    parser.add_argument('Eout', type=str,
                        help='Grid for the output energy in eV')
    parser.add_argument('theta', type=str_or_float,
                        help='Grid for the scattering angle in degrees')
    parser.add_argument('--output', type=str, nargs='+',
                        choices=['dynamicStruc', 'transferFunc', 'angularDistr'],
                        default=['dynamicStruc'],
                        help='What to return: dynamicStruc, transferFunc, angularDistr')

def get_DynamicStruc(args: argparse.Namespace) -> DynamicStruc:
    """
    Handle the arguments for the calculation of scattering function.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    DynamicStruc
        The scattering function object.
    """
    # Read the data from files:
    theta = np.loadtxt(args.theta) if isinstance(args.theta, str) else args.theta
    Eout = np.loadtxt(args.Eout)

    # Get the extra arguments for Pdos
    argsPdos = [get_Pdos(args)] if args.model != "fgm" else []

    # Compute the function:
    return DynamicStruc.from_model(args.Ein, args.M, args.T, Eout, theta,
                                   *argsPdos, model=args.model)

def handle_DynamicStrucArgs(args: argparse.Namespace) -> dict:
    """
    Handle the arguments for the calculation of scattering function.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    np.ndarray
        An array containing the values of the scattering function.
    """
    result = {}

    # Get the DynamicStruc object based on the arguments:
    dynamicStructure = get_DynamicStruc(args)

    # Get the requested outputs:
    if 'dynamicStruc' in args.output:
        result['values'] = dynamicStructure.values
    if 'transferFunc' in args.output:
        result['transferFunc'] = dynamicStructure.transferFunc
    if 'angleDistr' in args.output:
        result['angleDistr'] = dynamicStructure.angularDistr

    # Return the results as a dictionary:
    return result