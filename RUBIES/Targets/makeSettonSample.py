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

# Filter names
fs = sorted(
    [
        c
        for c in out.colnames
        if any([c.startswith(f) for f in ['e_f', 'f_f', 'flag_f']])
    ]
)
out = out[['mask', 'srcid', 'best_z', 'file'] + fs]
# out['mask'] = out['mask'].astype(str)

# # Merge in extra columns
# results = Table.from_pandas(
#     pd.read_parquet('/Users/hviding/Projects/LRD-Selection/data/merged.parquet')[
#         ['srcid', 'mask', 'spec.bl', 'morph.is_rsv']
#     ]
# )
# results = results[~results['mask'].mask]
# results['mask'] = results['mask'].astype(str)
# results['morph.is_rsv'] = results['mask'].astype(str)
# out = join(out, results, keys=['srcid', 'mask'], join_type='left')

# Save
out.write('setton.fits', overwrite=True)
