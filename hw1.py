"""Main program for homework 1."""
from argparse import ArgumentParser


def read_input(file_name):
    """Read and return all lines from input file."""
    with open(file_name, 'r') as f:
        lines = f.readlines()
    return lines


def main():
    """Parse arguments and pass them to task."""
    parser = ArgumentParser(description='Linear regression of 1-D input')
    parser.add_argument('file_name', type=str, help='Input file name')
    parser.add_argument('n', type=int, help='Number of polynomial bases')
    parser.add_argument('lambda', type=int, help='Lambda parameter for LSE')
    args = parser.parse_args()
    lines = read_input(args.file_name)


if __name__ == '__main__':
    main()