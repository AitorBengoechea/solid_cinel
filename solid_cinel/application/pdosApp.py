import argparse
import numpy as np
from solid_cinel.core.material.vibration.pdos import Pdos

def add_PdosArgs() -> argparse.ArgumentParser:
    """
    Add arguments to the parser for the calculation of the effective temperature.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the arguments should be added.
    """
    pdos_parser = argparse.ArgumentParser(description='Pdos file arguments')
    # Required arguments
    pdos_parser.add_argument('rho', type=str, help='pdos file or values file')

    # Optional arguments for the reading of the pdos file
    pdos_parser.add_argument('-dE', type=float, default=None, help='dE (optional, must be provided with values file)')
    pdos_parser.add_argument("-header", type=int, default=None)
    pdos_parser.add_argument("-usecols", type=list, default=[0, 1])
    pdos_parser.add_argument("-index_col", type=int, default=0)
    pdos_parser.add_argument("-engine", type=str, default='python')
    pdos_parser.add_argument("-grid", type=str, default="dE")
    return pdos_parser


def get_PdosArgs(pdos_args: list) -> argparse.Namespace:
    """
    Add arguments to the parser for the reading of the pdos file.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the arguments should be added.
    """
    return add_PdosArgs().parse_args(pdos_args)

def get_Pdos(argsPdos: argparse.Namespace) -> Pdos:
    """
    Get the pdos object based on the arguments.

    Parameters
    ----------
    argsPdos : argparse.Namespace
        The parsed arguments for the pdos file.

    Returns
    -------
    Pdos
        The pdos object.
    """
    # Check if dE is provided
    if argsPdos.dE:
        # If dE is provided, load the rho values from the file and create a pdos object
        pdos = Pdos.from_dE(np.loadtxt(argsPdos.rho), argsPdos.dE)
    else:
        # If dE is not provided, create a Tpdos object from the file
        pdos = Pdos.from_file([argsPdos.T], [argsPdos.rho], header=argsPdos.header,
                                  usecols=argsPdos.usecols, index_col=argsPdos.index_col,
                                  engine=argsPdos.engine, grid=argsPdos.grid)
    return pdos


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
    # Return the input temperature and the calculated effective temperature
    return np.array([args.T, get_Pdos(args).fix_T(args.T).Teff])
