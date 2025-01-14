#! /usr/bin/env python

"""
Script to fit one set of RUBIES spectra
"""

# Import packages
import os
import json
import argparse

# Astronomy packages
from astropy.table import Table

from simul_specfit.fitting import RubiesFit


def main():
    # Parse arguements
    parser = argparse.ArgumentParser(description='Fit Naidu')
    parser.add_argument('config', type=str, help='Config')
    args = parser.parse_args()

    # Load config from JSON file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Ensure results/Plots directories exist
    for d in ['RUBIES/Results', 'RUBIES/Plots']:
        if not os.path.exists(d):
            os.makedir(d)

    # Load targets
    targets = Table.read('RUBIES/Targets/naidu.fits')

    # Process
    if len(targets) > 0:
        process(targets, config)
    else:
        raise ValueError('No rows found')


def process(rows, config):
    # Get MCMC
    RubiesFit(config, rows)


if __name__ == '__main__':
    main()
