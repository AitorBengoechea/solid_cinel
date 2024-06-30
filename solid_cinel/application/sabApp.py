import argparse
import numpy as np
from solid_cinel.application.pdosApp import get_Pdos
from solid_cinel.core.scattering_function.sab import Sab




def add_SabArgs(parser: argparse.ArgumentParser):
    """
    Add arguments to the parser for the calculation of the effective temperature.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the arguments should be added.
    """
    parser.add_argument('model', type=str,
                        help='Model to use for the calculation of the S(alpha, -beta) table')
    parser.add_argument('alpha', type=str,
                        help='alpha grid')
    parser.add_argument('beta', type=str,
                        help='beta grid')


def handle_SabArgs(args: argparse.Namespace) -> np.array:
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
    # Get Sab class based on the model
    if args.model != "fgm":
        sab = Sab.from_model(args.alpha, args.beta, args.T, get_Pdos(args))
    else:
        sab = Sab.from_fgm(args.alpha, args.beta)

    # Return the values of the S(alpha, -beta) table
    return sab.data.values