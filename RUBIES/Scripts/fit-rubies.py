#! /usr/bin/env python

"""
Script to fit RUBIES spectra
"""

# Standard library
import os
import json
import argparse
from multiprocessing import Pool, cpu_count

# Numpy packages
import numpy as np

# Astronomy packages
from astropy.table import Table

from simul_specfit.fitting import RubiesFit


def main():
    # Parse arguements
    parser = argparse.ArgumentParser(description='Fit Rubies')
    parser.add_argument('config', type=str, help='Configuration file')
    parser.add_argument('--ncpu', type=int, help='Number of CPUs', default=cpu_count())
    args = parser.parse_args()

    # Load config from JSON file
    with open('config-narrow.json', 'r') as f:
        config = json.load(f)

    # Ensure results/Plots directories exist
    for d in ['RUBIES/Results', 'RUBIES/Plots']:
        if not os.path.exists(d):
            os.makedir(d)

    # Load targets
    targets = Table.read('RUBIES/Targets/targets.fits')

    # Get unique targets
    unique_targets = np.unique(targets['root', 'srcid'])

    # Get rows
    allrows = []
    for u in unique_targets:
        good = np.logical_and(
            targets['srcid'] == u['srcid'], targets['root'] == u['root']
        )

        # Get the rows
        goodrows = targets[good]
        allrows.append(goodrows)

    # Multiprocess
    with Pool(args.ncpu) as pool:
        pool.starmap(process, [(rows, config) for rows in allrows])


def process(rows, config):
    # Get MCMC
    try:
        RubiesFit(config, rows)
    except Exception as e:
        print(rows[0]['root', 'srcid'], e)


if __name__ == '__main__':
    main()
