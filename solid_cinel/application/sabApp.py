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
    parser.add_argument('model', type=str.lower,
                        choices=['fgm', 'sct', 'pdos'],
                        help='Model to use for the calculation of the S(alpha, -beta) table')
    parser.add_argument('alpha', type=str,
                        help='Alpha grid')
    parser.add_argument('beta', type=str,
                        help='Beta grid')
    parser.add_argument('T', type=float,
                        help='Temperature in Kelvin')
    parser.add_argument('--nphonon', type=int,
                        default=None,
                        help='Number of phonon')
    parser.add_argument('--decimal', type=float,
                        default=1.0e-6,
                        help='Decimal precision for the calculation of the number of phonons')
    parser.add_argument('--orderMax', type=int,
                        default=5000,
                        help='Maximum order for the calculation of the number of phonons')
    parser.add_argument('--threshold', type=float,
                        default=0.0,
                        help='Threshold for the calculation of the number of phonons')
    parser.add_argument('--p0', action='store_true',
                        help='Include the elastic peak in the calculation of the number of phonons')

def get_sab(args: argparse.Namespace) -> Sab:
    # Validate temperature
    if args.T < 0:
        raise ValueError("Temperature must be greater than 0")

    # Get the extra arguments for Pdos
    argsPdos = [get_Pdos(args)] if args.model != "fgm" else []
    kwargsPdos = {"nphonon": args.nphonon, "decimal": args.decimal,
                  "orderMax": args.orderMax, "threshold": args.threshold,
                  "p0": args.p0} if args.model == "pdos" else {}

    # Get Sab class based on the model
    return Sab.from_model(args.alpha, args.beta, args.T, *argsPdos,
                          model=args.model, **kwargsPdos)


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
    # Get the S(alpha, -beta) table
    sab = get_sab(args)

    return {"values": sab.data.values}
