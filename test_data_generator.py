#!/usr/bin/python3
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


def main():
    """Parse arguments and generate test data."""
    parser = ArgumentParser(
        description='Generate regression data with 1 feature and 1 target')
    parser.add_argument('file_name', type=str, help='Output file name')
    parser.add_argument('-s', '--nsample', default=50, type=int,
                        help='Number of samples')
    parser.add_argument('-n', '--noise', default=0, type=int,
                        help='Noise of data')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Whether or not to plot generated data')
    args = parser.parse_args()
    x, y = make_regression(n_samples=args.nsample, n_features=1, n_targets=1,
                           noise=args.noise)
    if args.plot:
        plt.scatter(x, y)
        plt.show()
    data = pd.DataFrame(x, y).reset_index()
    data.to_csv(args.file_name, index=False, header=False)


if __name__ == '__main__':
    main()
