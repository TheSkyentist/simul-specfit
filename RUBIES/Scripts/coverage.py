#! /usr/bin/env python

import tqdm
import numpy as np
import astropy.units as u
from astropy.table import Table
import astropy.constants as const
from concurrent.futures import ProcessPoolExecutor

# Load catalog
rubies = Table.read('RUBIES/Targets/rubies.fits')

# Lines to check
lines = {'Ha': 0.656461, 'Hb': 0.486268}

# Dimensionless resolution
lineres = ((1_000 * u.km / u.s) / (const.c)).to(u.dimensionless_unscaled).value

def check_lines(ruby):
    result = {}
    spectrum = Table.read(f'RUBIES/Spectra/{ruby["file"]}', 'SPEC1D')

    for line, restwave in lines.items():
        obswave = restwave * (1 + ruby['best_z'])
        linewidth = obswave * lineres
        linemask = np.logical_and(
            spectrum['wave'] > obswave - linewidth,
            spectrum['wave'] < obswave + linewidth,
        )
        detected = np.any(~spectrum['flux'][linemask].mask)
        result[f'cover_{line.lower()}'] = detected
    return result

# Run in parallel
with ProcessPoolExecutor() as executor:
    results = list(tqdm.tqdm(executor.map(check_lines, rubies), total=len(rubies)))

# Add results to original table
for key in results[0]:
    rubies[key] = [r[key] for r in results]

# Save updated table (optional)
rubies.write('rubies_cover.fits', overwrite=True)
