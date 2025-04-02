#! /usr/bin/env python

"""
Script to concatenate results
"""

# Import packages
import os
import json
import tqdm
import argparse

# Numpy packages
import numpy as np

# Astronomy packages
from astropy.table import Table, vstack

def main():
    # Parse arguements
    parser = argparse.ArgumentParser(description='Concat Results')
    parser.add_argument('config', type=str, help='Config')
    parser.add_argument('--catalog', type=str, help='Catalog file', default='RUBIES/Targets/targets.fits')
    args = parser.parse_args()

    # Load config from JSON file
    with open(args.config, 'r') as f:
        config = json.load(f)
    fittype = config['Name']
    if fittype:
        fittype = '_' + fittype

    # Load targets
    targets = Table.read(args.catalog)

    # Get unique
    unique = Table(np.unique(targets['root', 'srcid']))

    # Loop over unique
    rows = []
    for u in tqdm.tqdm(unique):
        root, srcid = u

        # Check if results exist
        file = f'RUBIES/Results/{root}-{srcid}{fittype}_summary.fits'
        if not os.path.exists(file):
            continue

        # Load results
        results = Table.read(file,'PARAMS')

        # Get mean and std of each column
        row = sum([(results[c].mean(), results[c].std()) for c in results.colnames], ())
        names = sum([(c, f'{c}_std') for c in results.colnames], ())

        # Get WAIC
        waic = Table.read(file, 'EXTRAS')['WAIC'][0]
        row += (float(waic),)
        names += ('WAIC',)

        # Add root and srcid
        row = (root, srcid) + row
        names = ('root', 'srcid') + names

        # Create table
        row = Table([[r] for r in row], names=names)
        rows.append(row)

    # Concatenate
    results = vstack(rows)
    results['PRISM_flux'] = results['PRISM_flux'].filled(1)
    results['PRISM_flux_std'] = results['PRISM_flux_std'].filled(0)
    results['PRISM_offset'] = results['PRISM_offset'].filled(0)
    results['PRISM_offset_std'] = results['PRISM_offset_std'].filled(0)

    results.write(f'RUBIES/Results/REH-simul{fittype}.fits', overwrite=True)

    for u in unique:
        rows = targets[np.logical_and(targets['root'] == u['root'],targets['srcid'] == u['srcid'])]
        
if __name__ == '__main__':
    main()