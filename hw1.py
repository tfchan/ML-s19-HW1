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


def print_result(method_name, equation, error):
    """Print found equation and error of a given method name."""
    print(f'{method_name}:')
    print(f'Fitting line: {equation}')
    print(f'Total error: {error}')


def perform_regression(methods, input_data, output_data):
    """Perform regression on data using different methods and show result."""
    plt.figure()
    method_count = 1
    for method_name, method in methods.items():
        method.fit(input_data, output_data)
        prediction, sq_error = method.predict(
            input_data, real_output=output_data)
        print_result(method_name, method.get_equation(), sq_error)
        print()
        ax = plt.subplot(len(methods), 1, method_count)
        ax.set_title(method_name)
        ax.plot(input_data, output_data, 'ro')
        ax.plot(input_data, prediction, 'b-')
        method_count += 1
    plt.show()


def main():
    """Parse arguments and pass them to task."""
    parser = ArgumentParser(description='Linear regression of 1-D input')
    parser.add_argument('file_name', type=str, help='Input file name')
    parser.add_argument('n', type=int, help='Number of polynomial bases')
    parser.add_argument('lambda_', type=int, help='Lambda parameter for LSE')
    args = parser.parse_args()
    lines = read_input(args.file_name)
    data = prepare_data(lines)

    methods = {'LSE': regressor1d.LSERegressor(args.n, args.lambda_),
               "Newton's method": regressor1d.NewtonsRegressor(args.n)}
    perform_regression(methods, data[0], data[1])


if __name__ == '__main__':
    main()
