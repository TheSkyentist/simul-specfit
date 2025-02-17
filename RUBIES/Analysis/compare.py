#! /usr/bin/env python

# Import packages
from matplotlib import pyplot
from astropy.table import Table, join

# Read Zach's Fluxes
zf = Table.read('/Users/hviding/Downloads/zach_fluxes.csv')
zf = zf[zf.colnames[9:]]

# Read REH Fluxes
rf = Table.read('RUBIES/Results/REH-simul.fits')
rf['uid'] = rf['uid'].astype(int)

# Join tables on uid and root
joined = join(rf, zf, keys=['uid', 'root'])

# Plot Ha Fluxes
fig, axes = pyplot.subplots(
    2,
    2,
    figsize=(15, 10),
    gridspec_kw={'height_ratios': [5, 1]},
    sharex=True,
    sharey='row',
)
fig.subplots_adjust(hspace=0.05, wspace=0.05)

# Limits
lim = [1e1, 1e4]

# HA G395M
axes[0, 0].plot(lim, lim, ls='--', color='black')
x, y = joined['Narrow-HI-6564.61_flux'], joined['Ha_g395m']
axes[0, 0].scatter(x, y, edgecolor='none', alpha=0.5)
axes[0, 0].set(xlim=lim, xscale='log', xlabel='REH')
axes[0, 0].set(ylim=lim, yscale='log', ylabel='ZL')
axes[0, 0].set(title=r'G395M', aspect='equal')

axes[1, 0].plot(lim, [1, 1], ls='--', color='black')
axes[1, 0].scatter(x, y / x, edgecolor='none', alpha=0.5)
axes[1, 0].set(xlim=lim, xscale='log', xlabel='REH')
axes[1, 0].set(ylim=[0, 2], yscale='linear', ylabel='ZL/REH')


# HA PRISM
axes[0, 1].plot(lim, lim, ls='--', color='black')
x = x * joined['PRISM_flux']
y = joined['Ha_prism']
axes[0, 1].scatter(x, y, edgecolor='none', alpha=0.5)
axes[0, 1].set(xlim=lim, xscale='log', xlabel='REH')
axes[0, 1].set(ylim=lim, yscale='log')
axes[0, 1].set(title=r'PRISM', aspect='equal')

axes[1, 1].plot(lim, [1, 1], ls='--', color='black')
axes[1, 1].scatter(x, y / x, edgecolor='none', alpha=0.5)
axes[1, 1].set(xlim=lim, xscale='log', xlabel='REH')
axes[1, 1].set(ylim=[0, 2], yscale='linear')


fig.suptitle(r'H$\alpha$ Flux')

# Save and close figure
fig.savefig('Hacompare.pdf')
pyplot.close(fig)

# Plot Ha Fluxes
fig, axes = pyplot.subplots(
    2,
    2,
    figsize=(15, 10),
    gridspec_kw={'height_ratios': [5, 1]},
    sharex=True,
    sharey='row',
)
fig.subplots_adjust(hspace=0.05, wspace=0.05)

lim = [1e0, 1e2]


# HA SNR G395M
axes[0, 0].plot(lim, lim, ls='--', color='black')
x, y = joined['Narrow-HI-6564.61_flux'], joined['Ha_g395m']
xerr, yerr = joined['Narrow-HI-6564.61_flux_std'], (joined['Ha_g395m_p84_err'] + joined['Ha_g395m_p16_err'])/2
axes[0, 0].scatter(x / xerr, y / yerr, edgecolor='none', alpha=0.5)
axes[0, 0].set(xlim=lim, xscale='log', xlabel='REH')
axes[0, 0].set(ylim=lim, yscale='log', ylabel='ZL')
axes[0, 0].set(title=r'G395M', aspect='equal')

axes[1, 0].plot(lim, [1, 1], ls='--', color='black')
axes[1, 0].scatter(x/xerr, (y/yerr) / (x/xerr), edgecolor='none', alpha=0.5)
axes[1, 0].set(xlim=lim, xscale='log', xlabel='REH')
axes[1, 0].set(ylim=[0, 2], yscale='linear', ylabel='ZL/REH')


# HA PRISM
axes[0, 1].plot(lim, lim, ls='--', color='black')
x, xerr = x * joined['PRISM_flux'], xerr * joined['PRISM_flux']
y, yerr = joined['Ha_prism'], (joined['Ha_prism_p84_err'] + joined['Ha_prism_p16_err'])/2
axes[0, 1].scatter(x/xerr, y/yerr, edgecolor='none', alpha=0.5)
axes[0, 1].set(xlim=lim, xscale='log', xlabel='REH')
axes[0, 1].set(ylim=lim, yscale='log')
axes[0, 1].set(title=r'PRISM', aspect='equal')

axes[1, 1].plot(lim, [1, 1], ls='--', color='black')
axes[1, 1].scatter(x/xerr, (y/yerr) / (x/xerr), edgecolor='none', alpha=0.5)
axes[1, 1].set(xlim=lim, xscale='log', xlabel='REH')
axes[1, 1].set(ylim=[0, 2], yscale='linear')


fig.suptitle(r'H$\alpha$ SNR')

# Save and close figure
fig.savefig('HaSNRcompare.pdf')
pyplot.close(fig)


# Plot Ha Fluxes
fig, axes = pyplot.subplots(
    2,
    2,
    figsize=(15, 10),
    gridspec_kw={'height_ratios': [5, 1]},
    sharex=True,
    sharey='row',
)
fig.subplots_adjust(hspace=0.05, wspace=0.05)

# Limits
lim = [1e1, 1e4]

# HA G395M
axes[0, 0].plot(lim, lim, ls='--', color='black')
x, y = joined['Narrow-[OIII]-5008.24_flux'], joined['Oiii_5007_g395m']
axes[0, 0].scatter(x, y, edgecolor='none', alpha=0.5)
axes[0, 0].set(xlim=lim, xscale='log', xlabel='REH')
axes[0, 0].set(ylim=lim, yscale='log', ylabel='ZL')
axes[0, 0].set(title=r'G395M', aspect='equal')

axes[1, 0].plot(lim, [1, 1], ls='--', color='black')
axes[1, 0].scatter(x, y / x, edgecolor='none', alpha=0.5)
axes[1, 0].set(xlim=lim, xscale='log', xlabel='REH')
axes[1, 0].set(ylim=[0, 2], yscale='linear', ylabel='ZL/REH')


# HA PRISM
axes[0, 1].plot(lim, lim, ls='--', color='black')
x = x * joined['PRISM_flux']
y = joined['Oiii_5007_prism']
axes[0, 1].scatter(x, y, edgecolor='none', alpha=0.5)
axes[0, 1].set(xlim=lim, xscale='log', xlabel='REH')
axes[0, 1].set(ylim=lim, yscale='log')
axes[0, 1].set(title=r'PRISM', aspect='equal')

axes[1, 1].plot(lim, [1, 1], ls='--', color='black')
axes[1, 1].scatter(x, y / x, edgecolor='none', alpha=0.5)
axes[1, 1].set(xlim=lim, xscale='log', xlabel='REH')
axes[1, 1].set(ylim=[0, 2], yscale='linear')


fig.suptitle(r'[O III] Flux')

# Save and close figure
fig.savefig('Oiiicompare.pdf')
pyplot.close(fig)

lim = [1e0, 1e2]

# Plot Ha Fluxes
fig, axes = pyplot.subplots(
    2,
    2,
    figsize=(15, 10),
    gridspec_kw={'height_ratios': [5, 1]},
    sharex=True,
    sharey='row',
)
fig.subplots_adjust(hspace=0.05, wspace=0.05)

# Limits
lim = [1e1, 1e4]


# Plot Ha Fluxes
fig, axes = pyplot.subplots(
    2,
    2,
    figsize=(15, 10),
    gridspec_kw={'height_ratios': [5, 1]},
    sharex=True,
    sharey='row',
)
fig.subplots_adjust(hspace=0.05, wspace=0.05)

# Limits
lim = [1e0, 1e2]

# OIII SNR G395M
axes[0, 0].plot(lim, lim, ls='--', color='black')
x, xerr = joined['Narrow-[OIII]-5008.24_flux'], joined['Narrow-[OIII]-5008.24_flux_std']
y, yerr = joined['Oiii_5007_g395m'], (joined['Oiii_5007_g395m_p84_err'] + joined['Oiii_5007_g395m_p16_err'])/2
axes[0, 0].scatter(x / xerr, y / yerr, edgecolor='none', alpha=0.5)
axes[0, 0].set(xlim=lim, xscale='log', xlabel='REH')
axes[0, 0].set(ylim=lim, yscale='log', ylabel='ZL')
axes[0, 0].set(title=r'G395M', aspect='equal')

axes[1, 0].plot(lim, [1, 1], ls='--', color='black')
axes[1, 0].scatter(x/xerr, (y/yerr) / (x/xerr), edgecolor='none', alpha=0.5)
axes[1, 0].set(xlim=lim, xscale='log', xlabel='REH')
axes[1, 0].set(ylim=[0, 2], yscale='linear', ylabel='ZL/REH')


# OIII SNR PRISM
axes[0, 1].plot(lim, lim, ls='--', color='black')
x, xerr = x * joined['PRISM_flux'], xerr * joined['PRISM_flux']
y, yerr = joined['Oiii_5007_prism'], (joined['Oiii_5007_prism_p84_err'] + joined['Oiii_5007_prism_p16_err'])/2
axes[0, 1].scatter(x / xerr, y / yerr, edgecolor='none', alpha=0.5)
axes[0, 1].set(xlim=lim, xscale='log', xlabel='REH')
axes[0, 1].set(ylim=lim, yscale='log')
axes[0, 1].set(title=r'PRISM', aspect='equal')

axes[1, 1].plot(lim, [1, 1], ls='--', color='black')
axes[1, 1].scatter(x/xerr, (y/yerr) / (x/xerr), edgecolor='none', alpha=0.5)
axes[1, 1].set(xlim=lim, xscale='log', xlabel='REH')
axes[1, 1].set(ylim=[0, 2], yscale='linear')



fig.suptitle(r'[O III] SNR')

# Save and close figure
fig.savefig('OiiiSNRcompare.pdf')
pyplot.close(fig)
