import argparse
import numpy as np
from solid_cinel.application.pdosApp import get_Pdos
from solid_cinel.core.dynamic_structure.sab import Sab


def add_SabArgs(parser: argparse.ArgumentParser):
    """
    Add arguments to the parser for the calculation of S(alpha, -beta) tables.

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
    parser.add_argument('T', type=float,
                        help='Temperature in Kelvin')
    parser.add_argument('--output', type=str, nargs='+',
                        choices=['values'],
                        default=['values'],
                        help='What to return: values')

def get_sab(args: argparse.Namespace) -> Sab:
    # Validate temperature
    if args.T < 0:
        raise ValueError("Temperature must be greater than 0")

    # Get Sab class based on the model
    if args.model == "pdos":
        return Sab.from_pdos(args.alpha, args.beta, args.T, get_Pdos(args))
    elif args.model == "sct":
        return Sab.from_sct(args.alpha, args.beta, args.T, get_Pdos(args))
    else:
        return Sab.from_fgm(args.alpha, args.beta, args.T)

def handle_SabArgs(args: argparse.Namespace) -> dict:
    """
    Handle the arguments for the calculation of the S(alpha, -beta) tables.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    np.ndarray
        An array containing the values of the S(alpha, -beta) table.
    """
    results = {}
    # Get the S(alpha, -beta) table
    sab = get_sab(args)

    if "values" in args.output:
        results["values"] = sab.data.values

    return results
