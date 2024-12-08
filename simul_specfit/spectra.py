"""
Module for Loading Spectra Data
"""

# Import packages
from os import path
from importlib import resources

# Astronomy packages
from astropy.table import Table
from astropy import units as u, constants as consts

# Numerical packages
from jax import numpy as jnp

# Calibration
from simul_specfit import calibration, defaults


# Spectra class
class Spectra:
    """
    Generic Collection of Spectra
    """

    def __init__(
        self,
        spectra: list,
        redshift_initial: float,
        λ_unit: u.Unit,
        fλ_unit: u.Unit,
        continuum_regions: list = [],
    ) -> None:
        """
        Initialize the spectra

        Parameters
        ----------
        spectra : list
            List of Spectrum objects
        redshift_initial : float
            Initial redshift
        λ_unit : u.Unit
            Wavelength target unit
        fλ_unit : u.Unit
            Spectral flux target unit (fλ)
        continuum_regions : list, optional
            Continuum regions to use, if not specified they will be computed

        Returns
        -------
        None
        """

        # Keep track
        self.redshift_initial = redshift_initial

        # Keep track of units
        self.λ_unit = λ_unit
        self.fλ_unit = fλ_unit

        # Store the spectra
        self.spectra = spectra

        # Keep track of names
        self.names = [spectrum.name for spectrum in spectra]

    def restrict(self, continuum_regions: list) -> None:
        """
        Restrict the spectra to the continuum regions

        Parameters
        ----------
        continuum_regions : list
            List of continuum regions

        Returns
        -------
        None
        """

        # Loop over the spectra
        for spectrum in self.spectra:
            spectrum.restrict(continuum_regions)

    def rescale(
        self, config: dict, continuum_regions: list, linepad: u.Quantity
    ) -> None:
        """
        Rescale the errorbars in each region

        Parameters
        ----------
        config : dict
            Configuration dictionary
        continuum_regions : list
            List of continuum regions
        linepad : u.Quantity
            Padding to mask emission lines

        Returns
        -------
        None
        """

        for spectrum in self.spectra:
            spectrum.rescale(config, continuum_regions, linepad)

    def restrictAndRescale(
        self,
        config: dict,
        continuum_regions: list,
        linepad: u.Quantity = defaults.LINEPAD,
    ) -> None:
        """
        Restrict the spectra to the continuum regions and rescale the errorbars

        Parameters
        ----------
        config : dict
            Configuration dictionary
        continuum_regions : list
            List of continuum regions

        Returns
        -------
        None
        """

        self.restrict(continuum_regions)
        self.rescale(config, continuum_regions, linepad)


# RUBIES Spectra
class RubiesSpectra(Spectra):
    """
    Collection of RUBIES spectra for a given source ID and mask
    """

    def __init__(
        self,
        rows: Table,
        spectra_directory: str,
        λ_unit: u.Unit = u.micron,
        fλ_unit: u.Unit = u.Unit(1e-20 * u.erg / u.s / u.cm**2 / u.angstrom),
    ) -> None:
        """
        Initialize the spectra

        Parameters
        ----------
        rows : Table
            Table of rows for the source
        spectra_directory : str
            Path to directory containing the spectra
        instrument_directory : str
            Path to directory containing the lsf curves
        λ_unit : u.Unit
            Wavelength target unit
        fλ_unit : u.Unit
            Spectral flux target unit (fλ)

        Returns
        -------
        None
        """

        # Compute iniital redshift
        redshift_initial = sum(rows['z']) / len(rows)

        # Compute the spectrum files
        spectrum_files = [path.join(spectra_directory, row['file']) for row in rows]

        # If there is only one spectrum, it is fixed, otherwise set PRISM to be free
        if len(spectrum_files) == 1:
            fixed = [True]
        else:
            fixed = [False if 'PRISM' in row['grating'] else True for row in rows]
        self.fixed = fixed

        # Load the spectra
        spectra = [
            RubiesSpectrum(row['grating'], sf, redshift_initial, λ_unit, fλ_unit, fix)
            for row, sf, fix in zip(rows, spectrum_files, fixed)
        ]

        # Initialize
        super().__init__(
            spectra=spectra,
            redshift_initial=redshift_initial,
            λ_unit=λ_unit,
            fλ_unit=fλ_unit,
        )


# Spectrum class
class Spectrum:
    """
    Spectrum from a given disperser
    """

    def __init__(
        self,
        name: str,
        low: jnp.ndarray,
        wave: jnp.ndarray,
        high: jnp.ndarray,
        flux: jnp.ndarray,
        err: jnp.ndarray,
        redshift_initial: float,
        λ_unit: u.Unit,
        fλ_unit: u.Unit,
    ) -> None:
        """
        Initialize the spectrum

        Parameters
        ----------
        name : str
            Name of the spectrum
        low : jnp.ndarray
            Low edge of the bins
        wave : jnp.ndarray
            Central wavelength of the bins
        high : jnp.ndarray
            High edge of the bins
        flux : jnp.ndarray
            Observed flux values
        err : jnp.ndarray
            Error in the flux values
        redshift_initial : float
            Initial redshift
        λ_unit : u.Unit
            Wavelength target unit
        fλ_unit : u.Unit
            Spectral flux target unit (fλ)

        Returns
        -------
        None
        """

        # Keep track of the name:
        self.name = name

        # Keep track of redshift
        self.redshift_initial = redshift_initial

        # Keep track of units
        self.λ_unit = λ_unit
        self.fλ_unit = fλ_unit

        # Mask NaN values and store
        mask = jnp.invert(jnp.isnan(err))
        for key, array in zip(
            ['wave', 'low', 'high', 'flux', 'err'],
            [wave, low, high, flux, err],
        ):
            setattr(self, key, array[mask])

    def __call__(self):
        """
        Return the attributes we care about
        """

        return (getattr(self, key) for key in ['low', 'wave', 'high', 'flux', 'err'])

    # Calculate if range is covered
    def coverage(self, low: float, high: float, partial: bool = True) -> jnp.ndarray:
        """
        Check if a given range is covered by the spectrum

        Parameters
        ----------
        low : float
            Low edge of the range
        high : float
            High edge of the range
        halfok : bool, optional
            Whether partial coverage is enough, defaults to True

        Returns
        -------
        jnp.ndarray
           Boolean array of spectral coverage
        """

        # Check if the range is covered
        if partial:
            return jnp.logical_and(low < self.high, self.low < high)
        else:
            return jnp.logical_and(low <= self.low, self.high <= high)

    # Restrict to continuum regions
    def restrict(self, continuum_regions: list) -> None:
        """
        Restrict the spectrum to the continuum regions

        Parameters
        ----------
        continuum_regions : list
            List of continuum regions

        Returns
        -------
        None
        """

        # Compute the mask
        opz = 1 + self.redshift_initial
        mask = jnp.logical_or.reduce(
            jnp.array(
                [
                    self.coverage(region[0] * opz, region[1] * opz, partial=False)
                    for region in continuum_regions
                ]
            )
        )

        # Apply the mask
        for key in ['wave', 'low', 'high', 'flux', 'err']:
            setattr(self, key, getattr(self, key)[mask])

    # Mask lines in continuum regions
    def maskLines(
        self,
        config: list,
        continuum_region: jnp.ndarray,
        linepad: u.Quantity,
    ) -> jnp.ndarray:
        """
        Mask the lines in the continuum region

        Parameters
        ----------
        continuum_region : jnp.ndarray
            Boundary of the continuum region
        config : dict
            Configuration of emission lines
        spectrum : Spectrum
            Spectrum
        linepad : u.Quantity
            Padding to mask emission lines

        Returns
        -------
        jnp.ndarray
            Masked region
        """

        # Grow by redshift
        opz = 1 + self.redshift_initial
        continuum_region = continuum_region * opz
        pad = (linepad / consts.c).to(u.dimensionless_unscaled).value

        # Extract the region
        low, high = continuum_region
        mask = jnp.logical_and(low < self.wave, self.wave < high)

        # Mask each line
        λ_unit = u.Unit(config['Unit'])
        for group in config['Groups']:
            for species in group['Species']:
                for line in species['Lines']:
                    # Compute the line wavelength
                    linewav = (line['Wavelength'] * λ_unit).to(self.λ_unit).value * opz

                    # Get the effective padding
                    linepad = linewav * pad

                    # Compute the boundaries
                    low, high = linewav - linepad, linewav + linepad

                    # Mask the line
                    linemask = jnp.logical_and(low < self.wave, self.wave < high)
                    mask = jnp.logical_and(mask, jnp.invert(linemask))

        return mask

    # Rescale errorbars based on linear continuum
    def scaleErrorbars(self, region: jnp.ndarray) -> float:
        """
        Rescale errorbars in a region assuming a linear continuum
        Do a least squares fit to the region and scale the errorbars to have unit variance

        Parameters
        ----------
        region : jnp.ndarray
            Boolean array defining the region of interest

        Returns
        -------
        float
            Scale factor for the errorbars
        """
        # Scale the error based on ratio of σ(flux) / median(err)
        # return jnp.std(flux[continuum_region]) / jnp.median(err[continuum_region])

        # Compute least squares fit
        N = region.sum()
        X = jnp.vstack([self.wave[region], jnp.ones(N)]).T
        W = jnp.diag(1 / jnp.square(self.err[region]))
        y = jnp.atleast_2d(self.flux[region]).T
        β = jnp.linalg.inv(X.T @ W @ X) @ X.T @ W @ y

        # Compute χ²/ν
        resid = X @ β - y
        χ2_ν = (resid.T @ W @ resid)[0][0] / (N - β.size)

        # Return scale that makes residuals have unit variance
        return jnp.sqrt(χ2_ν)

    def rescale(
        self, config: dict, continuum_regions: list, linepad: u.Quantity
    ) -> None:
        """
        Rescale the errorbars in each region

        Parameters
        ----------
        config : dict
            Configuration dictionary
        continuum_regions : list
            List of continuum regions
        linepad : u.Quantity
            Padding to mask emission lines

        Returns
        -------
        None
        """

        # Loop over the continuum regions
        newerr = jnp.zeros_like(self.err)
        for region in continuum_regions:
            # Compute the mask
            opz = 1 + self.redshift_initial
            mask = self.coverage(region[0] * opz, region[1] * opz, partial=False)

            # Scale the errorbars
            scale = self.scaleErrorbars(self.maskLines(config, region, linepad))

            # Apply the scaling
            newerr = jnp.where(mask, self.err * scale, newerr)

        # Store the new errorbars
        self.err = newerr


# RUBIES Spectrum
class RubiesSpectrum(Spectrum):
    def __init__(
        self,
        disperser: str,
        spec_file: str,
        redshift_initial: float,
        λ_unit: u.Unit,
        fλ_unit: u.Unit,
        fixed: bool,
    ) -> None:
        """
        Load the spectrum from a file

        Parameters
        ----------
        disperser : str
            Name of the spectrum
        spec_file : str
            File containing the spectrum
        redshift_initial : float
            Initial redshift
        λ_unit : u.Unit
            Wavelength target unit
        fλ_unit : u.Unit
            Spectral flux target unit (fλ)
        fixed : bool
            Whether flux/pixel offset are fixed

        Returns
        -------
        None
        """

        # Keep track if fixed
        self.fixed = fixed

        # Compute resolution
        lsf_dir = resources.files('simul_specfit.data.resolution')
        lsf_file = f'jwst_nirspec_{disperser.lower()}_lsf.fits'
        self.lsf = calibration.InterpLSFCurve(lsf_dir.joinpath(lsf_file), λ_unit)

        # Compute pixel offset
        disp_dir = resources.files('simul_specfit.data.resolution')
        disp_file = f'jwst_nirspec_{disperser.lower()}_disp.fits'
        self.offset = calibration.PixelOffset(disp_dir.joinpath(disp_file), λ_unit)

        # Load the spectrum from file
        spec = Table.read(spec_file, 'SPEC1D')

        # Unpack relevant columns, convert
        wave = spec['wave'].to(λ_unit)
        flux = spec['flux'].to(fλ_unit, equivalencies=u.spectral_density(wave))
        err = spec['err'].to(fλ_unit, equivalencies=u.spectral_density(wave))

        # Convert to JAX arrays
        wave, flux, err = jnp.array(wave), jnp.array(flux), jnp.array(err)

        # Calculate bin edges
        δλ = jnp.diff(wave) / 2
        mid = wave[:-1] + δλ
        edges = jnp.concat([wave[0:1] - δλ[0:1], mid, wave[-2:-1] + δλ[-2:-1]])
        low = edges[:-1]
        high = edges[1:]

        # Initialize the spectrum
        super().__init__(
            name=disperser,
            low=low,
            wave=wave,
            high=high,
            flux=flux,
            err=err,
            redshift_initial=redshift_initial,
            λ_unit=λ_unit,
            fλ_unit=fλ_unit,
        )
