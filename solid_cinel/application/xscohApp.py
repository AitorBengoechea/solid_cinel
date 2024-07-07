import argparse
import numpy as np
from solid_cinel.application.pdosApp import get_Pdos
from solid_cinel.core.material import Solid


def add_xsCohArgs(parser: argparse.ArgumentParser):
    """
    Add arguments to the parser for the calculation of the differential cross section.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the arguments should be added.
    """
    parser.add_argument('compositon_file', type=str,
                        help='File containing the atomic composition of the material')
    parser.add_argument('structure_file', type=str,
                        help='File containing the crystaline structure of the material')
    parser.add_argument('atomPos_file', type=str,
                        help='File containing the atoms positions in the unit cell of the material')
    parser.add_argument('energyCut', type=float,
                        help='Energy cut in eV')
    parser.add_argument('T', type=float,
                        help='Temperature in Kelvin')
    parser.add_argument('--precision', type=list, default=[6, 6], nargs=2,
                        help='Precision of the calculation')
    parser.add_argument('--d_min', type=float, default=None,
                        help='Minimum espace to calculate the multiplicity')
    parser.add_argument('--pddf_kind', type=str, default=None,
                        help='Key to calculate PDDF')
    parser.add_argument('--pddf_val', type=str, default=None,
                        help='Value to calculate PDDF')
    parser.add_argument('--threshold', type=float, default=1.e-30,
                        help='Threshold to consider a value as zero')


def handle_xsCohArgs(args: argparse.Namespace) -> np.array:
    """
    Handle the arguments for the calculation of the differential cross section.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    np.array
        An array containing the input temperature and the calculated effective temperature.
    """
    # Initialize the solid object:
    solid = Solid.from_files(args.compositon_file, args.structure_file, args.atomPos_file)

    # Introduce the partial density of states in the solid object:
    solid.set_pdos(get_Pdos(args))

    # Calculate the coherent cross section:
    xsCoh = solid.get_xsCoh(args.energyCut, args.T, precision=args.precision,
                            d_min=args.d_min, pddf_kind=args.pddf_kind,
                            pddf_val=args.pddf_val, threshold=args.threshold)
    return np.column_stack((xsCoh.index.values, xsCoh.values))
