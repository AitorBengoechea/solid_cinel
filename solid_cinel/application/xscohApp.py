import argparse
import numpy as np
from solid_cinel.application.pdosApp import get_Pdos
from solid_cinel.core.material import Solid


def add_BraggEdgesArgs(parser: argparse.ArgumentParser):
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
    Handle the arguments for the calculation of the coherant cross section.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    np.array
        An array containing the coherent cross section. The columns are:
            - The energy in eV
            - The coherent cross section in barns
    """
    # Initialize the solid object:
    solid = Solid.from_files(args.compositon_file, args.structure_file, args.atomPos_file)

    # Introduce the partial density of states in the solid object:
    solid.set_pdos(get_Pdos(args))

    # Calculate the coherent cross section:
    xsCoh = solid.get_XsCoh(args.energyCut, args.T, precision=args.precision,
                            d_min=args.d_min, pddf_kind=args.pddf_kind,
                            pddf_val=args.pddf_val, threshold=args.threshold)
    return np.column_stack((xsCoh.index.values, xsCoh.values))


def handle_BraggEdgesArgs(args: argparse.Namespace) -> np.array:
    """
    Handle the arguments for the calculation of the Bragg Edges.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    np.array
        An array containing all the information of the Bragg Edges. The columns
        are:
            - hkl: The Miller indexes of the Bragg Edge (h, k, l)
            - The d-spacing of the Bragg Edge in Angstrom (d)
            - The structure factor of the Bragg Edge (Fsq)
            - Orientation angle of the Bragg Edge in degrees
            - multiplicity of the Bragg Edge
            - The energy of the Bragg Edge in eV
            - The Pole-Density Distribution Function (PDDF) of the Bragg Edge
            - The coherent cross section of the Bragg Edge in barns
            - The difraction angle in degrees
    """
    # Initialize the solid object:
    solid = Solid.from_files(args.compositon_file, args.structure_file, args.atomPos_file)

    # Introduce the partial density of states in the solid object:
    solid.set_pdos(get_Pdos(args))

    # Calculate all the information of the Bragg Edges:
    braggEdges = solid.get_BraggEdges(args.energyCut, args.T,
                                      precision=args.precision,
                                      d_min=args.d_min, pddf_kind=args.pddf_kind,
                                      pddf_val=args.pddf_val, threshold=args.threshold)
    return braggEdges.reset_index().values
