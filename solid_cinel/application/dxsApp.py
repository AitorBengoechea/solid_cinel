import argparse
import numpy as np
from solid_cinel.application.pdosApp import get_Pdos
from solid_cinel.application.scatfunctApp import str_or_float
from solid_cinel.core.xs.scatfunc import ScatFunc
from solid_cinel.core.xs.xs import Xs


def add_DxsArgs(parser: argparse.ArgumentParser):
    """
    Add arguments to the parser for the calculation of the differential cross
    section.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the arguments should be added.
    """
    parser.add_argument('model', type=str,
                        help='Model to use for the calculation of the Dxs')
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
    parser.add_argument('theta', type=str_or_float,
                        help='Grid for the scattering angle in degrees')
    parser.add_argument("--recoil", type=bool, default=True,
                        help='Include the recoil effect in the calculation')


def handle_DxsArgs(args: argparse.Namespace) -> np.array:
    """
    Handle the arguments for the calculation of the differential cross section.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    np.array
        An array containing the differential cross section. The columns are:
            - The energy in eV
            - The scattering angle in degrees
            - The differential cross section in barns
    """
    # Read the data from files:
    theta = np.loadtxt(args.theta) if isinstance(args.theta, str) else args.theta
    Eout = np.loadtxt(args.Eout)
    xs0K = Xs.read_xs(args.xs0K)

    # Define the method to use
    method = ScatFunc.from_theta if isinstance(theta, (int, float)) else ScatFunc.from_sab

    # Get the extra arguments for Pdos
    argsPdos = [get_Pdos(args)] if args.model != "fgm" else []
    dictRecoil = {'recoil': args.recoil} if method == ScatFunc.from_sab else {}

    # Compute the function:
    dxs = method(xs0K, args.Ein, args.M, args.T, Eout, theta, *argsPdos,
                 model=args.model, **dictRecoil)

    # Return the values of the scattering function
    return dxs.data.values
