import argparse
import numpy as np
from solid_cinel.core.material.vibration.pdos import Tpdos, Epdos


def add_TeffArgs(parser: argparse.ArgumentParser):
    """
    Add arguments to the parser for the calculation of the effective temperature.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the arguments should be added.
    """
    # Required arguments
    parser.add_argument('T', type=float, help='Temperature in Kelvin')
    parser.add_argument('rho', type=str, help='pdos file or values file')

    # Optional arguments for the reading of the pdos file
    parser.add_argument('-dE', type=float, default=None, help='dE (optional, must be provided with values file)')
    parser.add_argument("-header", type=int, default=None)
    parser.add_argument("-usecols", type=list, default=[0, 1])
    parser.add_argument("-index_col", type=int, default=0)
    parser.add_argument("-engine", type=str, default='python')
    parser.add_argument("-grid", type=str, default="dE")


def handle_TeffArgs(args: argparse.Namespace) -> np.array:
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
    # Check if dE is provided
    if args.dE:
        # If dE is provided, load the rho values from the file and create a pdos object
        rho_ = np.loadtxt(args.rho)
        pdos = Epdos.from_dE(rho_, args.dE).get_Tpdos(args.T)
    else:
        # If dE is not provided, create a Tpdos object from the file
        pdos = Tpdos.from_dE_file(args.T, args.rho, header=args.header,
                                  usecols=args.usecols, index_col=args.index_col,
                                  engine=args.engine)

    # Return the input temperature and the calculated effective temperature
    return np.array([args.T, pdos.Teff])
