#! /usr/bin/env python

# Import packages
import argparse
import requests
import numpy as np
from astropy.table import Table

# Column Labels
labels = (
    'root',
    'jname',
    'uid',
    'ndup',
    'file',
    'ra',
    'dec',
    'srcid',
    'grating',
    'filter',
    'grade',
    'zfit',
    'z',
    'comment',
    'sn50',
    'wmin',
    'wmax',
    'HST',
    'NIRCam',
    'slit',
    'FITS',
    'Fnu',
    'Flam',
)


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
        default='https://s3.amazonaws.com/msaexp-nirspec/extractions/rubies_all_extractions_v3.json',
    )
    parser.add_argument(
        '--prepend-version',
        type=str,
        default='nod-',
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
    r = requests.get(url)
    if r.status_code == 200:
        # Get the data
        data = r.json()['data']

        # Create table
        table = Table(np.array(data), names=labels, dtype=[type(d) for d in data[0]])

        # Add program ID
        table.add_column(pid, name='pid', index=1)

        # Convert comment column to string
        table['comment'] = table['comment'].astype(str)

        # Fix comment column
        bad_comment = np.logical_or(table['comment'] == 'None', table['comment'] == ';')
        table['comment'][bad_comment] = ''

        # Remove unneeded columns
        table.remove_columns(['HST', 'NIRCam', 'slit', 'FITS', 'Fnu', 'Flam'])

        # Add columns
        table.add_columns(
            np.array([r.split('-') for r in table['root']]).T,
            names=['survey', 'field', 'version'],
            indexes=[1, 1, 1],
        )

        # Add version info to file
        version = table['version']
        for key in ['root', 'file', 'version']:
            table[key] = [
                val.replace(ver, pv + ver) for val, ver in zip(table[key], version)
            ]

        # Save table
        table.write('targets.fits', overwrite=True)

    else:
        raise ValueError('Failed to download targets from DJA')


# Call main function
if __name__ == '__main__':
    main()
