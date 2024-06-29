import argparse
import numpy as np
from solid_cinel.application.teff import add_TeffArgs, handle_TeffArgs

# Map keywords to their respective functions
KEYWORD_TO_FUNCTION_MAP = {
    'Teff': {
        'add': add_TeffArgs,
        'handle': handle_TeffArgs,
    },
    # Add other keywords here...
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
    if keyword in KEYWORD_TO_FUNCTION_MAP:
        KEYWORD_TO_FUNCTION_MAP[keyword]['add'](parser)
    else:
        raise ValueError(f'Invalid keyword: {keyword}')


def handle_args(args: argparse.Namespace, keyword: str) -> np.array:
    """
    Handle the arguments based on the keyword.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.
    keyword : str
        The keyword to determine how to handle the arguments.

    Returns
    -------
    np.array
        The results of handling the arguments.

    Raises
    ------
    ValueError
        If the keyword is not found in the KEYWORD_TO_FUNCTION_MAP.
    """
    if keyword in KEYWORD_TO_FUNCTION_MAP:
        return KEYWORD_TO_FUNCTION_MAP[keyword]['handle'](args)
    else:
        raise ValueError(f'Invalid keyword: {keyword}')


def get_results(args: argparse.Namespace, remaining_args: list):
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
    np.array
        The results of handling the arguments.
    """
    # Second parser to parse the dynamic arguments of the functions
    parserDyn = argparse.ArgumentParser(description='Solid Cinel application',
                                        prog="Solid Cinel")

    # Add the dynamic arguments based on the keyword
    add_args(parserDyn, args.keyword)

    # Parse the rest of the arguments
    argsDyn = parserDyn.parse_args(remaining_args)

    # Call the function based on the keyword to handle the dynamic arguments
    return handle_args(argsDyn, args.keyword)


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
    if keyword == 'Teff':
        # Reshape your 1D array into a 2D array with one row
        results = results[np.newaxis, ::]

    # Save the results in a file
    np.savetxt(f'{keyword}', results)


def main(*command_manual_args, write_to_file=True):
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
    np.array or None
        The results of handling the arguments, if write_to_file is False. Otherwise, None.
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
