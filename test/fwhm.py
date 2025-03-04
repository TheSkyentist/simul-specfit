#! /usr/bin/env python

# Packages
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot
from astropy.table import Table, join

# Load grades
grades = Table.read('/Users/hviding/Downloads/Grading - Grades.csv')

# Load Broad line data
broad = Table.read(
    '/Users/hviding/Projects/RUBIES-BL/RUBIES/Results/REH-simul_broad.fits'
)

# Photometry
phot = Table.read(
    '/Users/hviding/Projects/LRD-Selection/data/psf_matched_phot_AW_alltargets.fits'
).to_pandas()
phot.rename(columns={'id': 'srcid'}, inplace=True)
phot['field'] = phot['field'].astype('string')

# Joined results
rubies = join(grades, broad, join_type='left').to_pandas()
rubies = phot.merge(rubies, on=['field', 'srcid'])

# Mean grade
good_grade = rubies[['REH', 'AdG', 'JEG']].fillna(0).mean(axis=1) >= 2
sigma = 2
good_prob = rubies['PBroad'] > (norm.cdf(sigma) - norm.cdf(-sigma))

# FWHMs and errs
fwhm, fwhm_err = rubies['HI_broad_6564.61_fwhm'], rubies['HI_broad_6564.61_fwhm_std']

# Create figure
fig, axes = pyplot.subplots(
    2, 2, figsize=(15, 15), sharex=False, constrained_layout=True
)

# Enumerate over axes
for i, ax in enumerate(axes.flatten()):
    # Plot x sigma upper limit
    if i:
        bins = np.arange(0, 2500, 100)
        X = fwhm - (i - 1) * fwhm_err
    else:
        bins = np.logspace(0, 2)
        ax.set(xscale='log')
        X = fwhm / fwhm_err
    ax.hist(
        X[good_prob & ~good_grade],
        bins=bins,
        histtype='step',
        label=r'P(B) $>$ 0.95 \& Grade $<$ 2',
        linewidth=2,
        density=False,
    )
    ax.hist(
        X[good_prob & good_grade],
        bins=bins,
        histtype='step',
        label=r'P(B) $>$ 0.95 \& Grade $\geq$ 2',
        linewidth=2,
        density=False,
    )
    xlabel = 'FWHM '
    if i:
        xlabel += rf'$-$ {i - 1}$\sigma$ [km/s]'
    else:
        xlabel += 'SNR'
    ax.set(ylabel='Density')  # , yticklabels=[])
    ax.set(xlabel=f'{xlabel}', xlim=(bins.min(), bins.max()))

axes[0, 0].legend()

# Save figure
fig.savefig('fwhm.pdf')
pyplot.close(fig)

# Create figure by phtometry
fcols = [col for col in rubies.columns if 'f_f' in col]
fig, axes = pyplot.subplots(4, 5, figsize=(20, 20), constrained_layout=True)

bins = np.arange(21, 35, 0.5)
for fcol, ax in zip(fcols, axes.flatten()):
    mag = -2.5 * np.log10(rubies[fcol] / 1e9) + 8.9
    ax.hist(mag, bins=bins, histtype='step', label='All', linewidth=2, density=True)
    ax.hist(
        mag[good_grade & good_prob],
        bins=bins,
        histtype='step',
        label='Broad',
        linewidth=2,
        density=True,
    )

    ax.set_title(fcol.split('_')[1].upper())
    ax.set(yticklabels=[],xlim=(bins.max(), bins.min()))

fig.savefig('mag.pdf')
