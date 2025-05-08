#! /usr/bin/env python

# Astropy Packages
import pandas as pd
from astropy.table import Table, join

# Spectroscopic Sample
spec = Table.read('rubies.fits')

# Photometric Sample
phot = Table.read(
    '/Users/hviding/Projects/LRD-Selection/data/photometry/photometry.fits'
)

# Restrict to those with PRISM
spec_prism = spec[spec['grating'] == 'PRISM']

# Make field column
spec_prism['field'] = [m[0:3].upper() for m in spec_prism['mask']]

# Join
out = join(spec_prism, phot, keys=['field', 'srcid'], join_type='left')

# Load data
v3 = Table.read('nod-v3.fits').to_pandas()
v3['grating'] = v3['grating'].astype('string')
v3['file'] = v3['file'].astype('string')
v3['mask'] = v3['mask'].astype('string')

# Consistent srcids
v3['srcid'] = v3.groupby(['mask', 'uid'])['srcid'].transform('max')

# Fix specific srcids
remap = [
    (65077, 65078),  # v3 vs nod-v3
    (146412, 930581),  # v3 vs nod-v3
    (162168, 983090),  # v3 vs nod-v3
    (164279, 980032),  # v3 vs nod-v3
    (10576, 10577),  # v3 vs v4
    (26388, 26389),  # v3 vs v4
    (31061, 31062),  # v3 vs v4
    (34219, 839123),  # v3 vs v4
    (38060, 843430),  # v3 vs v4
    (45849, 45921),  # v3 vs v4
    (46135, 46136),  # v3 vs v4
    (46141, 46199),  # v3 vs v4
    (75619, 75709),  # v3 vs v4
    (151027, 947582),  # v3 vs v4
    (154267, 943340),  # v3 vs v4
    (165166, 978699),  # v3 vs v4
    (169961, 972343),  # v3 vs v4
    (819800, 819846),  # v3 vs v4
    (926480, 926642),  # v3 vs v4
    (834181, 29954),  # Keep as v3 for phot
    (842741, 37427),  # Keep as v3 for phot
    (819846, 819800),  # Keep as v3 for phot
]
for old, new in remap:
    v3.loc[v3['srcid'] == old, 'srcid'] = new

# Get prism
v3_prism = Table.from_pandas(v3[v3['grating'] == 'PRISM'])

# Iterate over v4 rows
for o in out[out['reduction'] == 'v4']:
    # Get srcid
    srcid = o['srcid']

    # Skip bonus LRD
    if (srcid == 57040) or (srcid == 60935):
        continue

    if srcid == 902297:
        out['file'][out['srcid'] == srcid] = (
            'rubies-egs51-nod-v3_prism-clear_4233_902297.spec.fits'
        )
        continue

    # Get rows in v3
    matched = v3[
        (v3['srcid'] == srcid) & (v3['mask'] == o['mask']) & (v3['grating'] == 'PRISM')
    ]

    if matched.empty:
        print(f'No match for {srcid} in v3')
        # continue

    out['file'][(out['srcid'] == srcid) & (out['mask'] == o['mask'])] = matched['file']

# Filter names
fs = sorted(
    [
        c
        for c in out.colnames
        if any([c.startswith(f) for f in ['e_f', 'f_f', 'flag_f']])
    ]
)
out = out[['mask', 'srcid', 'best_z', 'file'] + fs]
out['mask'] = out['mask'].astype(str)

# Merge in extra columns
results = Table.from_pandas(
    pd.read_parquet('/Users/hviding/Projects/LRD-Selection/data/merged.parquet')[
        ['srcid', 'mask', 'spec.bl', 'morph.is_rsv']
    ]
)
results = results[~results['mask'].mask]
results['mask'] = results['mask'].astype(str)
results['morph.is_rsv'].data.data[results['morph.is_rsv'].data.mask] = False
results['morph.is_rsv'] = results['morph.is_rsv'].astype(bool)
out = join(out, results, keys=['srcid', 'mask'], join_type='left')

# Save
out.write('setton.fits', overwrite=True)
