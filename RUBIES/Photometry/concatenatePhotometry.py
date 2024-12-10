#! /usr/bin/env python

# Import packages
from astropy.table import Table, vstack

# Fields
fields = ['egs_obs56', 'uds_obs1']
versions = ['v74', 'v72']

# Loop over fields
cats = []
for field, version in zip(fields, versions):
    # Read in the photometry
    cat = Table.read(f'{field}_prism_psfmatch_phot_AW{version}.fits')

    # Keep track of field
    cat.add_column(field, name='field', index=0)

    # Append to list
    cats.append(cat)

# Concatenate and save
vstack(cats, metadata_conflicts='silent').write('photometry.fits', overwrite=True)
