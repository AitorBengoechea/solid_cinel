"""
This module contains the main function for the Solid Cinel application.

The main function uses the argparse module to create a command-line interface.
It accepts a list of integers as positional arguments and an optional `--sum`
argument. If the `--sum` argument is provided, the function will print the sum
of the provided integers. Otherwise, it will print the maximum of the provided
integers.

This module can be run as a script. When run as a script, the main function is
executed.

Author: Aitor Bengoechea (aitorabf@gmail.com)
"""
import argparse

import numpy as np

from solid_cinel.application.teff import add_TeffArgs, handle_TeffArgs

def add_args(parser: argparse.ArgumentParser, keyword: str):
    # Call the function based on the keyword to extract the dynamic arguments
    if keyword == 'Teff':
        # Call the function for 'Teff' keyword
        add_TeffArgs(parser)
    elif keyword == 'sab':
        # Call the function for 'sab' keyword
        pass
    elif keyword == 'scatfunc':
        # Call the function for 'scatfunc' keyword
        pass
    elif keyword == 'dxs':
        # Call the function for 'dxs' keyword
        pass
    elif keyword == 'xs':
        # Call the function for 'xs' keyword
        pass
    elif keyword == 'ddxs':
        # Call the function for 'ddxs' keyword
        pass
    else:
        print(f'Invalid keyword: {keyword}')

def handle_args(args: argparse.Namespace, keyword: str) -> np.array:
    # Call the function based on the keyword to handle the dynamic arguments
    if keyword == 'Teff':
        # Call the function for 'Teff' keyword
        return handle_TeffArgs(args)
    elif keyword == 'sab':
        # Call the function for 'sab' keyword
        pass
    elif keyword == 'scatfunc':
        # Call the function for 'scatfunc' keyword
        pass
    elif keyword == 'dxs':
        # Call the function for 'dxs' keyword
        pass
    elif keyword == 'xs':
        # Call the function for 'xs' keyword
        pass
    elif keyword == 'ddxs':
        # Call the function for 'ddxs' keyword
        pass
    else:
        print(f'Invalid keyword: {keyword}')

def main():
    """
    This is the main function for the Solid Cinel application. It uses the
    argparse module to create a command-line interface.

    It accepts a keyword as a positional argument. The keyword determines
    which function the application will execute.
    """
    # First parser to parse 'keyword' argument to determine the function to execute
    parser = argparse.ArgumentParser(description='Solid Cinel: Solid State Physics and Materials Science',
                                     prog="Solid Cinel")
    parser.add_argument('keyword', type=str,
                        help='Keyword to determine the function to execute')

    # Parse 'keyword' argument
    args, remaining_args = parser.parse_known_args()

    # Second parser to parse the dynamic arguments of the functions
    parserDyn = argparse.ArgumentParser(description='Solid Cinel application',
                                        prog="Solid Cinel")

    # Add the dynamic arguments based on the keyword
    add_args(parserDyn, args.keyword)

    # Parse the rest of the arguments
    argsDyn = parserDyn.parse_args(remaining_args)

    # Call the function based on the keyword to handle the dynamic arguments
    results = handle_args(argsDyn, args.keyword)

    # Save the results in a file
    np.savetxt(f'{args.keyword}.out', results)


if __name__ == "__main__":
    main()
