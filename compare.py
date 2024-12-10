#! /usr/bin/env python

# Import packages
from matplotlib import pyplot
from astropy.table import Table, join

# Read Zach's Fluxes
zf = Table.read('/Users/hviding/Downloads/zach_fluxes.csv')
zf = zf[zf.colnames[9:]]

# Read REH Fluxes
rf = Table.read('RUBIES/Results/REH-simul.fits')
rf['srcid'] = rf['srcid'].astype(int)

# Join tables on srcid and root
joined = join(rf, zf, keys=['srcid', 'root'])

# Plot Ha Fluxes
fig,ax = pyplot.subplots()


ax.scatter(joined['Narrow-HI-6564.61_flux'],joined['Ha_g395m'])
ax.set(xlim=[1,1e5],xscale='log')
ax.set(ylim=[1,1e5],yscale='log')

pyplot.show()