import argparse
import numpy as np
from solid_cinel.application.pdosApp import get_PdosArgs, add_TeffArgs, handle_TeffArgs
from solid_cinel.application.sabApp import add_SabArgs, handle_SabArgs
from solid_cinel.application.scatfunctApp import add_ScatFuncArgs, handle_ScatFuncArgs
from solid_cinel.application.dxsApp import add_DxsArgs, handle_DxsArgs
from solid_cinel.application.ddxsApp import add_DDxsArgs, handle_DDxsArgs
from solid_cinel.application.xscohApp import add_BraggEdgesArgs, handle_xsCohArgs, handle_BraggEdgesArgs

# Map keywords to their respective functions
KEYWORD_TO_FUNCTION_MAP = {
    'teff': {
        'add': add_TeffArgs,
        'handle': handle_TeffArgs,
    },
    "sab": {
        "add": add_SabArgs,
        "handle": handle_SabArgs,
    },
    "scatfunc": {
        "add": add_ScatFuncArgs,
        "handle": handle_ScatFuncArgs,
    },
    "dxs": {
        "add": add_DxsArgs,
        "handle": handle_DxsArgs,
    },
    "ddxs": {
        "add": add_DDxsArgs,
        "handle": handle_DDxsArgs,
    },
    "xscoh": {
        "add": add_BraggEdgesArgs,
        "handle": handle_xsCohArgs,
    },
    "braggedges": {
        "add": add_BraggEdgesArgs,
        "handle": handle_BraggEdgesArgs,
    }
}


def add_args(parser: argparse.ArgumentParser, keyword: str):
    """
    Add the arguments based on the keyword.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser object.
    keyword : str
        The keyword to determine which arguments to add.

    Raises
    ------
    ValueError
        If the keyword is not found in the KEYWORD_TO_FUNCTION_MAP.
    """
    if keyword.lower() in KEYWORD_TO_FUNCTION_MAP:
        KEYWORD_TO_FUNCTION_MAP[keyword.lower()]['add'](parser)
    else:
        raise ValueError(f'Invalid keyword: {keyword}')


def handle_args(keyword: str, args: argparse.Namespace) -> np.ndarray:
    """
    Handle the arguments based on the keyword.

    Parameters
    ----------
    keyword : str
        The keyword to determine how to handle the arguments.
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    np.ndarray
        The results of handling the arguments.

    Raises
    ------
    ValueError
        If the keyword is not found in the KEYWORD_TO_FUNCTION_MAP.
    """
    if keyword.lower() in KEYWORD_TO_FUNCTION_MAP:
        return KEYWORD_TO_FUNCTION_MAP[keyword.lower()]['handle'](args)
    else:
        raise ValueError(f'Invalid keyword: {keyword}')


def merge_namespaces(ns1, ns2):
    """
    Merge two argparse.Namespace objects.

    Parameters
    ----------
    ns1 : argparse.Namespace
        The first namespace.
    ns2 : argparse.Namespace
        The second namespace.

    Returns
    -------
    argparse.Namespace
        The merged namespace.
    """
    # Convert namespaces to dictionaries
    dict1 = vars(ns1)
    dict2 = vars(ns2)

    # Merge dictionaries
    merged_dict = {**dict1, **dict2}

    # Convert merged dictionary back to namespace
    return argparse.Namespace(**merged_dict)

def get_results(args: argparse.Namespace, remaining_args: list) -> np.ndarray:
    """
    Get the results based on the keyword.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.
    remaining_args : list
        The remaining unparsed arguments.

    Returns
    -------
    np.ndarray
        The results of handling the arguments.
    """
    # Second parser to parse the dynamic arguments of the functions
    parserDyn = argparse.ArgumentParser(description='Solid Cinel application',
                                        prog="Solid Cinel")

    # Add the dynamic arguments based on the keyword
    add_args(parserDyn, args.keyword)

    # Parse the dynamic arguments and get the remaining arguments for the pdos file
    argsDyn, pdos_args = parserDyn.parse_known_args(remaining_args)

    # If there are arguments for the pdos file, parse them
    if len(pdos_args) > 0:
        argsPdos = get_PdosArgs(pdos_args)
        argsDyn = merge_namespaces(argsDyn, argsPdos)

    # Call the function based on the keyword to handle the dynamic arguments
    return handle_args(args.keyword, argsDyn)


def write_results(results: np.array, keyword: str):
    """
    Write the results to a file.

    Parameters
    ----------
    results : np.array
        The results to write to the file.
    keyword : str
        The keyword to determine the filename.

    Notes
    -----
    If the keyword is 'Teff', the results are reshaped into a 2D array with one row.
    """
    if keyword.lower() == 'teff':
        # Reshape your 1D array into a 2D array with one row
        results = results[np.newaxis, ::]

    # Save the results in a file
    np.savetxt(f'{keyword}', results)


def main(*command_manual_args: list, write_to_file=True):
    """
    This is the main function for the Solid Cinel application. It uses the
    argparse module to create a command-line interface.

    It accepts a keyword as a positional argument. The keyword determines
    which function the application will execute.

    Parameters
    ----------
    *command_manual_args : list
        The command line arguments.
    write_to_file : bool, optional
        Whether to write the results to a file. If False, the results are returned.

    Returns
    -------
    np.ndarray or None
        The results of handling the arguments, if write_to_file is False.
        Otherwise, None.
    """
    # 'keyword' argument to determine the function to execute
    parser = argparse.ArgumentParser(description='Solid Cinel: Solid State Physics and Materials Science',
                                     prog="Solid Cinel")
    parser.add_argument('keyword', type=str,
                        help='Keyword to determine the function to execute')

    # Parse 'keyword' argument
    if len(command_manual_args) == 0:
        args, remaining_args = parser.parse_known_args()
    else:
        args, remaining_args = parser.parse_known_args(command_manual_args)

    # Get the results based on the keyword
    results = get_results(args, remaining_args)

    # Save the results in a file or return them
    if write_to_file:
        write_results(results, args.keyword)
    else:
        return results


if __name__ == "__main__":
    main()
