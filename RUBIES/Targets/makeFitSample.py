#! /usr/bin/env python

# Astropy Packages
from astropy.table import Table

# Data manipulation
import numpy as np
import pandas as pd

# Load data
dja = Table.read('dja.fits').to_pandas()

# Consistent srcids
dja['srcid'] = dja.groupby(['mask', 'uid'])['srcid'].transform('max')

# Sort by grade and reduction
dja.sort_values(
    by=['mask', 'srcid', 'grating', 'grade', 'reduction'],
    ascending=[True, True, True, False, True],
    inplace=True,
)

# Drop duplicates
dja.drop_duplicates(subset=['mask', 'srcid', 'grating'], keep='first', inplace=True)

# Only use nod-v3 files
for col in ['root', 'file']:
    dja[col] = dja[col].astype('string')
    dja[col] = dja[col].apply(
        lambda x: x.replace('v3', 'nod-v3') if 'nod' not in x else x
    )


# Only keep those with at least one grade 3
dja = dja.groupby(['mask', 'srcid']).filter(lambda x: (x['grade'] == 3).any())


# Function to get the best redshift
def get_best_redshift(group):
    # Sort the group by grade (descending) and grating (alphabetically ascending)
    group_sorted = group.sort_values(by=['grade', 'grating'], ascending=[False, True])
    # Select the first row (best redshift)
    best_row = group_sorted.iloc[0]
    return best_row


# Apply the function to get the best redshift for each mask, srcid combination
best_redshifts = (
    dja.groupby(['mask', 'srcid'])
    .apply(get_best_redshift, include_groups=False)
    .reset_index(drop=True)
)

# Compute best redshift for each object
best_z = dja.groupby(['mask', 'srcid']).apply(
    lambda g: g.sort_values(['grade', 'grating'], ascending=[False, True])['z'].iloc[0],
    include_groups=False,
)

# Determine which object to use based on the redshift range
use_v3 = best_z.between(3.4, 6.9)
use_v4 = best_z.between(6.9, 7.3) | best_z.between(3.1, 3.4)

# Use v3 for nominal redshift range
dja_v3 = dja[dja.set_index(['mask', 'srcid']).index.isin(best_z[use_v3].index)]

# Now we need to add in objects without a grade
all_v3 = Table.read('nod-v3.fits').to_pandas()

# Consistent srcids
all_v3['srcid'] = all_v3.groupby(['mask', 'uid'])['srcid'].transform('max')

# Find those missing ungraded spectra
srcids = []
for i, group in dja_v3.groupby(['mask', 'srcid']):
    # Find match
    subset_v3 = all_v3[np.logical_and(all_v3['mask'] == i[0], all_v3['srcid'] == i[1])]

    # This occurs when the subset from all nod v3 is larger
    if len(group) < len(subset_v3):
        srcids.append(i)

# Get the ones we need to add in
add_in = pd.DataFrame(srcids, columns=('mask', 'srcid')).merge(all_v3)

# Columns to keep
cols = [
    c for c in all_v3.columns if c in np.intersect1d(add_in.columns, dja_v3.columns)
]
new_v3 = pd.concat([add_in[cols], dja_v3[cols]])
new_v3.drop_duplicates(subset=['mask', 'srcid', 'grating'], keep='first', inplace=True)

# Use v4 for extended redshift range
dja_v4 = dja[dja.set_index(['mask', 'srcid']).index.isin(best_z[use_v4].index)]

# Get the v4 objects
v4 = Table.read('prelim-v4.fits').to_pandas()
v4[['grating', 'filter']] = v4['grating'].astype('string').str.split('_', expand=True)

# Consistent srcids
v4['srcid'] = v4.groupby(['mask', 'slitid'])['srcid'].transform('max')

# Remove all those with 'bkg' in file
v4['file'] = v4['file'].astype(str)
v4 = v4[~v4['file'].str.contains('bkg')]

# Merge
rubies_v4 = dja_v4[['srcid', 'mask']].drop_duplicates().merge(v4, how='inner')

# Keep v3 grades unless v4 is better
for i, row in rubies_v4.iterrows():
    good = (
        (dja_v4['srcid'] == row['srcid'])
        & (dja_v4['mask'] == row['mask'])
        & (dja_v4['grating'] == row['grating'])
    )
    if good.any():
        # Get the v4 grade
        grade_v4 = rubies_v4.loc[i, 'grade']

        # Get the v3 grade
        grade_v3 = dja_v4[good].iloc[0]['grade']

        # If v4 grade is better, replace z, zfit, and grade
        if pd.isna(grade_v4) or grade_v3 >= grade_v4:
            rubies_v4.loc[i, 'z'] = dja_v4[good].iloc[0]['z']
            rubies_v4.loc[i, 'zfit'] = dja_v4[good].iloc[0]['zfit']
            rubies_v4.loc[i, 'grade'] = dja_v4[good].iloc[0]['grade']


# Columns to keep
cols = [
    c for c in dja_v4.columns if c in np.intersect1d(rubies_v4.columns, new_v3.columns)
]

# Concat
rubies = pd.concat([new_v3[cols], rubies_v4[cols]], ignore_index=True)
Table.from_pandas(rubies).write('rubies.fits', overwrite=True)
