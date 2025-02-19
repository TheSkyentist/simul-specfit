#! /usr/bin/env python

import tqdm
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt


# Load all targets
targets = Table.read('RUBIES/Targets/targets.fits')
targets['root'] = targets['root'].astype(str)

# Get unique targets
unique_targets = np.unique(targets['root', 'uid'])

# Iterate over unique targets
ha_coverage = []
for u in tqdm.tqdm(unique_targets):
    # Get rows
    rows = targets[
        np.logical_and(targets['uid'] == u['uid'], targets['root'] == u['root'])
    ]

    # Ensure at least one grade 3
    bestrow = rows[rows['grade'] == 3]
    if len(bestrow) == 0:
        continue
    if len(bestrow) > 1:
        bestrow = bestrow[bestrow['grating'] == 'G395M']

    # If z isn't -1 else use fitz
    if bestrow['z'] == -1:
        redshift_initial = bestrow['zfit'][0]
    else:
        redshift_initial = bestrow['z'][0]

    # Make sure redshift is in range
    if not np.logical_and(redshift_initial > 3.5, redshift_initial < 7.0):
        continue

    # Keep track
    ha_coverage.append(u)

# For each target
pbroads = []
for u in tqdm.tqdm(ha_coverage):
    uid, root = str(u['uid']), str(u['root'])

    # Get WAICs
    try:
        narrow = Table.read(f'RUBIES/Results/{root}-{uid}_summary.fits', 2)['WAIC']
        broad = Table.read(f'RUBIES/Results/{root}-{uid}_broad_summary.fits', 2)[
            'WAIC'
        ]
        waic_values = np.array([broad, narrow]).flatten()

        # Compute probability
        weights = np.exp(-0.5 * waic_values - np.min(waic_values,0))
        weights /= np.sum(weights,0)
        pbroad = weights[0]

        # Keep track
        pbroads.append(pbroad)
    except Exception as _:
        pbroads.append(0)

# Plot
plt.hist(pbroads, bins=50)
plt.savefig('test.pdf')
plt.close('all')
