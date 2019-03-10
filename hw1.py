#!/usr/bin/python3
"""Main program for homework 1."""
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import regressor1d


def read_input(file_name):
    """Read and return all lines from input file."""
    with open(file_name, 'r') as f:
        lines = f.readlines()
    return lines


def prepare_data(lines):
    """Convert raw lines to column major list."""
    rows = [list(map(float, line.strip().split(','))) for line in lines]
    prepared = [list(column) for column in zip(*rows)]
    return prepared


def main():
    """Parse arguments and pass them to task."""
    parser = ArgumentParser(description='Linear regression of 1-D input')
    parser.add_argument('file_name', type=str, help='Input file name')
    parser.add_argument('n', type=int, help='Number of polynomial bases')
    parser.add_argument('lambda_', type=int, help='Lambda parameter for LSE')
    args = parser.parse_args()
    lines = read_input(args.file_name)
    data = prepare_data(lines)

    lse_regressor = regressor1d.LSERegressor(args.n, args.lambda_)
    lse_regressor.fit(data[0], data[1])
    lse_output, lse_error = lse_regressor.predict(data[0], real_output=data[1])

    newtons_regressor = regressor1d.NewtonsRegressor(args.n)
    newtons_regressor.fit(data[0], data[1])
    newtons_output, newtons_error = newtons_regressor.predict(
        data[0], real_output=data[1])

    plt.figure()
    plt.subplot(211)
    plt.plot(data[0], data[1], 'ro')
    plt.plot(data[0], lse_output, 'b-')
    plt.subplot(212)
    plt.plot(data[0], data[1], 'ro')
    plt.plot(data[0], newtons_output, 'b-')
    plt.show()


if __name__ == '__main__':
    main()
