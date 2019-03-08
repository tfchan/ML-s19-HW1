"""Main program for homework 1."""
from argparse import ArgumentParser


def main():
    """Parse arguments and pass them to task."""
    parser = ArgumentParser(description='Linear regression of 1-D input')
    parser.add_argument('file_name', type=str, help='Input file name')
    parser.add_argument('n', type=int, help='Number of polynomial bases')
    parser.add_argument('lambda', type=int, help='Lambda parameter for LSE')
    args = parser.parse_args()


if __name__ == '__main__':
    main()