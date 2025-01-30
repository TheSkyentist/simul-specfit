#! /usr/bin/env python

# Import packages
import argparse
import requests
import numpy as np
from bs4 import BeautifulSoup
from astropy.table import Table


def main() -> None:
    """
    Parse arguements and download

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Download targets from DJA')
    parser.add_argument(
        '--url',
        type=str,
        help='DJA URL',
        default='https://s3.amazonaws.com/msaexp-nirspec/extractions/rubies_prelim_v4',
    )
    parser.add_argument(
        '--prepend-version',
        type=str,
        default='',
        help='String to prepend to version',
    )
    parser.add_argument(
        '--pid',
        type=int,
        default=4233,  # RUBIES PID
        help='Program ID',
    )
    args = parser.parse_args()
    url = args.url
    pv = args.prepend_version
    pid = args.pid

    # Download
    download(url, pv, pid)


def download(url: str, pv: str, pid: int) -> None:
    """
    Download targets from DJA

    Parameters
    ----------
    url : str
        URL to download from
    pv : str
        Prepend version string (reduction type)
    pid : int
        Program ID

    Returns
    -------
    None
    """

    # Get targets
    r = requests.get(f'{url}.json')
    if r.status_code == 200:
        # Get the data
        data = r.json()['data']
    else:
        raise ValueError('Failed to download targets from DJA')

    # Get labels
    r = requests.get(f'{url}.html')
    if r.status_code == 200:
        # Get the labels
        soup = BeautifulSoup(r.text, 'html.parser')

        # Find the only thead block
        thead = soup.find('thead')

        # Extract column headers into a list
        labels = [th.text.strip() for th in thead.find_all('th')]
    else:
        raise ValueError('Failed to download labels from DJA')

    # Get targets
    r = requests.get(f'{url}.json')
    if r.status_code == 200:
        # Get the data
        data = r.json()['data']
    else:
        raise ValueError('Failed to download targets from DJA')

    # Create table
    delim = '||'
    table = Table(
        np.genfromtxt(
            [delim.join([str(i) for i in d]) for d in data],
            delimiter=delim,
            dtype=None,
            names=labels,
            missing_values='None',
        )
    )
    table.rename_column('file_','file')

    # Add program ID
    table.add_column(pid, name='pid', index=1)

    # Fix comment column
    bad_comment = np.logical_or(table['comment'] == 'None', table['comment'] == ';')
    table['comment'][bad_comment] = ''

    # Get rid of all columns after comment
    table.remove_columns(table.colnames[table.colnames.index('comment') + 1 :])

    # Add columns
    table.add_columns(
        np.array([r.split('-') for r in table['root']]).T,
        names=['survey', 'field', 'reduction'],
        indexes=[1, 1, 1],
    )

    # Add version info to file
    version = table['reduction']
    for key in ['root', 'file', 'reduction']:
        table.replace_column(
            key, [val.replace(ver, pv + ver) for val, ver in zip(table[key], version)]
        )

    # Save table
    table.write('targets.fits', overwrite=True)


# Call main function
if __name__ == '__main__':
    main()
