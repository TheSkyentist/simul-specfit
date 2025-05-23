#! /usr/bin/env python

import numpy as np
import pandas as pd
from scipy.stats import norm
from astropy.table import Table, join

# Load in the data
narrow = Table.read('RUBIES/Results/REH-simul.fits')
broad = Table.read('RUBIES/Results/REH-simul_broad.fits')
cauchy = Table.read('RUBIES/Results/REH-simul_cauchy.fits')

# Join the tables
results = join(narrow, broad, keys=['root', 'srcid'], table_names=('narrow', 'broad'))
with_cauchy = join(results, cauchy, keys=['root', 'srcid'], table_names=('', 'cauchy'))

# Compute the probabilities
results['PBroad'] = 1 / (
    1 + np.exp((results['WAIC_broad'] - results['WAIC_narrow']) / 2)
)
results['sigma'] = norm.ppf((results['PBroad'] + 1) / 2)
results['PCauchy'] = 1 / (
    1 + np.exp((with_cauchy['WAIC'] - with_cauchy['WAIC_broad']) / 2)
)

# Sort by field, uid, Pbroad
results = results.to_pandas().sort_values(
    ['PBroad', 'root', 'srcid'], ascending=[False, True, True]
)
results.insert(
    0,
    'field',
    results['root'].astype('string').apply(lambda x: x.split('-')[1][0:3].upper()),
)

# Define aggregation: max for 'srcid', unique list for everything else
ignore_cols = ['field', 'srcid']
agg_fs = {
    col: 'max' if col == 'id' else lambda x: list(x)
    for col in results.columns
    if col not in ignore_cols
}

# Group by field and uid
tab_grouped = results.groupby(ignore_cols).agg(agg_fs).reset_index()
# for col in tab_grouped.columns:
#     if col not in ['field','srcid']:
#         tab_grouped[col] = tab_grouped[col].apply(clean_list)

# Determine max number of duplicates per (field, uid)
max_dupes = max(
    len(x)
    for col in tab_grouped.columns
    if col not in ignore_cols
    for x in tab_grouped[col]
)

# Expand lists into separate columns
expanded_cols = {col: tab_grouped[col] for col in ignore_cols}
for col in tab_grouped.columns:
    if col not in ignore_cols:  # Expand only list columns
        for i in range(max_dupes):
            expanded_cols[f'{col}-{i}' if i else f'{col}'] = tab_grouped[col].apply(
                lambda x: x[i] if i < len(x) else None
            )

# Create expanded dataframe
results = pd.DataFrame(expanded_cols)
results.sort_values(
    ['PBroad', 'root', 'srcid'], ascending=[False, True, True], inplace=True
)
results['root'] = results['root'].astype(str)
results['root-1'] = results['root-1'].astype(str)
# results['root-2'] = results['root-2'].astype(str)
Table.from_pandas(results).write('fitting-results.fits', overwrite=True)

# results['root'] = results['root'].apply(lambda x: x.replace('v4','nod-v4'))

# Limit to FWHM > 1000
results['fwhm_snr'] = (
    results['HI_broad_6564.61_fwhm'] / results['HI_broad_6564.61_fwhm_std']
)

# Add in old Grades
grades = pd.read_csv('/Users/hviding/Projects/LRD-Selection/data/Grading - Final.csv')
grades = grades[
    ['srcid', 'field', 'REH', 'AdG', 'JEG', 'Old Comments', 'New Comments', 'final']
]

# Merge with grades
results = results.merge(grades, how='left', on=['srcid', 'field'])

# Save to summary
summary = results[
    [
        'srcid',
        'field',
        'root',
        'root-1',
        'sigma',
        'fwhm_snr',
        'PBroad',
        'PCauchy',
        'REH',
        'AdG',
        'JEG',
        'Old Comments',
        'New Comments',
        'final',
    ]
]

# Restrict to PBroad()

summary.to_csv('summary.csv')
