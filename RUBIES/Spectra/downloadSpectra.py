#! /usr/bin/env python

# Import packages
import argparse
import requests
from os import path
from astropy.table import Table
from multiprocessing.pool import ThreadPool

# Default URL
default_url = 'https://s3.amazonaws.com/msaexp-nirspec/extractions/'


# Define download function
def downloadSpec(
    url: str,
    row: Table.Row,
    rd: bool,
) -> None:
    """
    Download a spectrum from DJA

    Parameters
    ----------
    url : str
        DJA URL
    row : Table.Row
        Relevant target row
    rd : bool
        Redownload if true
    """

    # Get URL
    root, file = row['root','file']

    # Check if file exists
    if path.exists(file) and not rd:
        # print(f'{file} already exists')
        return

    # Download URL
    remote = path.join(url, root, file)
    r = requests.get(remote)
    if r.status_code == 200:
        with open(file, 'wb') as f:
            f.write(r.content)
        # print(f'Downloaded {file}')
    else:
        print(f'Error downloading {file}')

# Define main function
def main() -> None:
    """
    Parse arguements and multithread download

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # Set up parser
    parser = argparse.ArgumentParser(description='Download spectra from DJA')
    parser.add_argument('--url', type=str, default=default_url)
    parser.add_argument('--ncpu', type=int, default=1)
    parser.add_argument('--redownload', action='store_true')
    parser.add_argument('--targets', type=str, default='../Targets/targets.fits')

    # Parse arguments
    args = parser.parse_args()
    ncpu = args.ncpu
    url = args.url
    rd = args.redownload
    targets = Table.read(args.targets)

    # Restrict to valid targets
    # targets = targets[targets['srcid'] > 0]

    # Single threading
    if ncpu == 1:
        for row in targets:
            downloadSpec(url, row, rd)
    # Multithreading
    else:
        with ThreadPool(args.ncpu) as executor:
            executor.starmap(downloadSpec, [(url, row, rd) for row in targets])


# Run main function
if __name__ == '__main__':
    main()
