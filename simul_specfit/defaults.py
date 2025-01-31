"""
Default Values
"""

from astropy import units as u

LINEDETECT: u.Quantity = 1_000 * (u.km / u.s)
LINEPAD: u.Quantity = 3_000 * (u.km / u.s)
CONTINUUM: u.Quantity = 10_000 * (u.km / u.s)