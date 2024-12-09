import os
import tqdm
import numpy as np
from astropy.table import Table, vstack

# Load targets
targets = Table.read('RUBIES/Targets/targets.fits')

# Get unique
unique = Table(np.unique(targets['root', 'srcid']))

# Loop over unique
rows = []
for u in tqdm.tqdm(unique):
    root, srcid = u
    # root = root.decode('utf-8')

    # Check if results exist
    if not os.path.exists(f'RUBIES/Results/{root}-{srcid}_fit.fits'):
        continue

    # Load results
    results = Table.read(f'RUBIES/Results/{root}-{srcid}_fit.fits')

    # Get mean and std of each column
    row = sum([(results[c].mean(), results[c].std()) for c in results.colnames], ())
    names = sum([(c, f'{c}_std') for c in results.colnames], ())

    # Add root and srcid
    row = (root, srcid) + row
    names = ('root', 'srcid') + names

    # Create table
    row = Table(np.array(row), names=names)
    rows.append(row)

# Concatenate
results = vstack(rows)
results['PRISM_flux'] = results['PRISM_flux'].filled(1)
results['PRISM_flux_std'] = results['PRISM_flux_std'].filled(0)
results['PRISM_offset'] = results['PRISM_offset'].filled(0)
results['PRISM_offset_std'] = results['PRISM_offset_std'].filled(0)

results.write('REH-simul.fits', overwrite=True)

#! /usr/bin/env python

for u in unique:
    rows = targets[np.logical_and(targets['root'] == u['root'],targets['srcid'] == u['srcid'])]
    
    