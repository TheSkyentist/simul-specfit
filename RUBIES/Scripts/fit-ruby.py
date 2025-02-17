#! /usr/bin/env python

"""
Script to fit one set of RUBIES spectra
"""

# Import packages
import os
import json
import argparse

# Numpy packages
import numpy as np

# Astronomy packages
from astropy.table import Table

from simul_specfit.fitting import RubiesFit


def main():
    # Parse arguements
    parser = argparse.ArgumentParser(description='Fit Ruby')
    parser.add_argument('config', type=str, help='Config')
    parser.add_argument('root', type=str, help='Root')
    parser.add_argument('uid', type=int, help='Source ID')
    parser.add_argument('--catalog', type=str, help='Catalog file', default='RUBIES/Targets/targets.fits')
    args = parser.parse_args()

    # Load config from JSON file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Ensure results/Plots directories exist
    for d in ['RUBIES/Results', 'RUBIES/Plots']:
        if not os.path.exists(d):
            os.makedir(d)

    # Load targets
    targets = Table.read(args.catalog)
    rows = targets[
        np.logical_and(targets['root'] == args.root, targets['uid'] == args.uid)
    ]

    # Process
    if len(rows) > 0:
        process(rows, config)
    else:
        raise ValueError('No rows found')


def process(rows, config):
    # Get MCMC
    RubiesFit(config, rows)


if __name__ == '__main__':
    main()
