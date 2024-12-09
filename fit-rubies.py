# Import packages
import json
from multiprocessing import Pool

# Numpy packages
import numpy as np

# Astronomy packages
from astropy.table import Table

from simul_specfit.fitting import RubiesMCMCFit

# Load config from JSON file
with open('config.json', 'r') as f:
    config = json.load(f)

# Get testing targets
# srcid = 154183  # 154183
# rows = targets[targets['srcid'] == srcid]

def main():

    # Load targets
    targets = Table.read('RUBIES/Targets/targets.fits')

    # Get unique targets
    unique_targets = np.unique(targets['root', 'srcid'][targets['srcid'] > 0])

    # Get rows
    allrows = []
    for u in unique_targets:
        good = np.logical_and(targets['srcid'] == u['srcid'], targets['root'] == u['root'])

        # Get the rows
        goodrows = targets[good]
        allrows.append(goodrows)

    # Multiprocess 
    for rows in allrows[0:10]:
        process(rows)
    # with Pool(10) as pool:
    #     pool.map(process, allrows)

def process(rows):

    # Get MCMC
    try:
        RubiesMCMCFit(config, rows)
    except Exception as e:
        print(rows[0]['root','srcid'],e)

if __name__ == '__main__':
    main()
