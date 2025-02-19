#! /usr/bin/env python

# Import packages

# Standard library
import argparse
import requests

# Data manipulation
import numpy as np
import pandas as pd

# HTML parsing
from bs4 import BeautifulSoup

# Astropy
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
        default='https://s3.amazonaws.com/msaexp-nirspec/extractions/rubies_all_extractions_v3',
    )
    parser.add_argument(
        '--prepend-version',
        type=str,
        default='nod-',
        help='String to prepend to version',
    )
    parser.add_argument(
        '--output', type=str, default='targets.fits', help='Output filename'
    )
    args = parser.parse_args()
    url = args.url
    pv = args.prepend_version
    output = args.output

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

    # Create table
    df = pd.DataFrame(data, columns=labels, dtype=None)
    df = df.apply(lambda col: col.astype('string') if col.dtype == 'object' else col)

    # If Grating in Table fix it
    if 'Grating' in df.columns:
        df.insert(
            df.columns.get_loc('Grating'),
            'grating',
            np.where(df['file'].str.contains('prism'), 'PRISM', 'G395M'),
        )
        df.drop('Grating', axis=1, inplace=True)

    # Get rid of all columns after comment
    df.drop(df.columns[df.columns.get_loc('comment') + 1 :], axis=1, inplace=True)

    # Restrict to rubies
    df = df[df['root'].str.startswith('rubies')]

    # Add columns
    split_columns = df['root'].str.split('-', expand=True, n=2)
    for idx, col in enumerate(['survey', 'mask', 'reduction']):
        df.insert(df.columns.get_loc('root'), col, split_columns[idx])

    # Add version info to file
    ver = df['reduction']
    for key in ['root', 'file', 'reduction']:
        df[key] = df.apply(
            lambda row, key=key, ver=ver: row[key].replace(
                ver[row.name], pv + ver[row.name]
            ),
            axis=1,
        )
        df[key] = df[key].astype('string')

    # Fill comment with empty string
    df['comment'] = df['comment'].fillna('')

    # Save table
    Table.from_pandas(df).write(output, format='fits', overwrite=True)


# Call main function
if __name__ == '__main__':
    main()
