#! /usr/bin/env python

# Import packages
import numpy as np
from tqdm import tqdm
from sedpy import observate
from astropy.io import fits
from astropy import units as u
from astropy.table import Table, vstack, hstack

# Open the targets
rubies = Table.read('RUBIES/Targets/rubies.fits').to_pandas()
good = rubies.select_dtypes(include='object')
rubies[good.columns] = good.astype('string')

# For now, just limit to 154183
# rubies = rubies[rubies['srcid'] == 180520]

# Get the relevant filters
fnames = ['f277w', 'f356w', 'f410m', 'f444w']
filters = observate.load_filters([f'jwst_{f.lower()}' for f in fnames])


# Create output table
out = {'srcid': [], 'root': [], 'mags': [], '': [], '_broad': []}


# Group by root and srcid
for (root, srcid), data in tqdm(rubies.groupby(['root', 'srcid'])):
    print(root, srcid)
    # Make sure PRISM is in dispersers
    summary = Table.read(f'RUBIES/Results/{root}-{srcid}_summary.fits')
    if not np.any(['PRISM' in x for x in summary.colnames]):
        continue

    # Make sure at least two prism datapoints
    if np.load(f'RUBIES/Results/{root}-{srcid}_full.npz')['PRISM_wavelength'].size < 2:
        continue

    file = data['file'][data['grating'] == 'PRISM'].values[0]

    # Load the PRISM spectrum
    prism = Table.read(f'RUBIES/Spectra/{file}', 'SPEC1D')
    prism_flam = (
        prism['flux']
        .to(
            u.erg / u.s / u.cm**2 / u.AA,
            equivalencies=u.spectral_density(u.Quantity(prism['wave'])),
        )
        .value
    )
    prism_wave = prism['wave'].value * 10000

    # Compute prism magnitudes
    prism_mags = observate.getSED(prism_wave, prism_flam, filterlist=filters)

    # Iterate over narrow and broad results
    for ext in ['', '_broad']:
        # Sumamry results
        summary = Table.read(f'RUBIES/Results/{root}-{srcid}{ext}_summary.fits')

        # Get line names
        line_names = [x[:-3] for x in summary.colnames if x.endswith('_ew')]

        # All results
        results = np.load(f'RUBIES/Results/{root}-{srcid}{ext}_full.npz')

        # Compute corrected fluxes
        line_fluxes = results['PRISM_flux'][:, np.newaxis, np.newaxis] * results[
            'PRISM_lines'
        ].transpose(0, 2, 1)

        # Now calculate the emission line fluxes
        line_mags = observate.getSED(
            results['PRISM_wavelength'] * 10000,
            line_fluxes * 1e-20,
            filterlist=filters,
        )

        # Calculate delta
        dm = -2.5 * np.log10(1 - 10 ** (-0.4 * (line_mags - prism_mags)))

        # Compute medians
        med = np.median(dm, axis=0)
        std = np.std(dm, axis=0)

        # Get names of product of linenames and filters
        names = [f'{ln}_d{f}' for ln in line_names for f in fnames]
        row = Table(
            np.concatenate([med.flatten(), std.flatten()]),
            names=names + [n + '_std' for n in names],
        )
        out[ext].append(row)

    out['srcid'].append(srcid)
    out['root'].append(root)
    out['mags'].append(Table(prism_mags, names=fnames))


# Construct the total output
left = Table([out['srcid'], out['root']], names=['srcid', 'root'])

hdul = [fits.PrimaryHDU()]
hdul += [fits.BinTableHDU(hstack([left, vstack(out['mags'])]), name='PRISM_MAG')]
hdul += [fits.BinTableHDU(hstack([left, vstack(out[''])]), name='NARROW')]
hdul += [fits.BinTableHDU(hstack([left, vstack(out['_broad'])]), name='BROAD')]

fits.HDUList(hdul).writeto('synmag.fits', overwrite=True)
