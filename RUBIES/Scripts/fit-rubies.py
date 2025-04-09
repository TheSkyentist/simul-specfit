#! /usr/bin/env python

"""
Script to fit RUBIES spectra
"""

# Standard library
import os
import json
import argparse
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor

# Numpy packages
import numpy as np

# Astronomy packages
from astropy.table import Table

from simul_specfit.fitting import RubiesFit

# Use safer (but slower) spawn method for multiprocessing
# Necessary for stability on Linux, already the default on macOS
ctx = mp.get_context('spawn')


def main():
    # Parse arguements
    parser = argparse.ArgumentParser(description='Fit Rubies')
    parser.add_argument('config', type=str, help='Configuration file')
    parser.add_argument(
        '--catalog',
        type=str,
        help='Catalog file',
        default='RUBIES/Targets/targets.fits',
    )
    parser.add_argument(
        '--ncpu', type=int, help='Number of CPUs', default=mp.cpu_count()
    )
    args = parser.parse_args()

    # Load config from JSON file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Ensure results/Plots directories exist
    for d in ['RUBIES/Results', 'RUBIES/Plots']:
        if not os.path.exists(d):
            os.makedirs(d)

    # Load targets
    targets = Table.read(args.catalog)

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

    # Create partial function
    partial_process = partial(process, config=config)

    # Multiprocess
    with ProcessPoolExecutor(max_workers=args.ncpu, mp_context=ctx) as executor:
        for rows in allrows:
            executor.submit(partial_process, rows)
        # executor.map(partial_process, allrows)


def process(rows, config):
    # Get MCMC
    try:
        RubiesFit(config, rows)
    except Exception as e:
        print(rows[0]['root', 'srcid'], e)


if __name__ == '__main__':
    main()
