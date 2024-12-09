# Import packages
import json
import argparse

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
    # Parse arguements
    parser = argparse.ArgumentParser(description='Fit Ruby')
    parser.add_argument('root', type=str, help='Root')
    parser.add_argument('srcid', type=int, help='Source ID')
    args = parser.parse_args()

    # Load targets
    targets = Table.read('RUBIES/Targets/targets.fits')
    rows = targets[
        np.logical_and(targets['root'] == args.root, targets['srcid'] == args.srcid)
    ]

    # Process
    if len(rows) > 0:
        process(rows)

def process(rows):
    # Get MCMC
    RubiesMCMCFit(config, rows)



if __name__ == '__main__':
    main()
