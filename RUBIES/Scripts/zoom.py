#! /usr/bin/env python

# Standard library
import argparse
from multiprocessing import Pool

# Numerical packages
import numpy as np

# Astropy packages
from astropy import units as u
from astropy.table import Table, join

# Plotting
from matplotlib import pyplot

# Hard coded continuum regions:
cont_regs = [[0.4775, 0.4925], [0.65, 0.665]]


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Broad Compare')
    parser.add_argument('--ncpu', type=int, help='Number of CPUs', default=200)
    args = parser.parse_args()

    # Load all targets
    targets = Table.read('RUBIES/Targets/rubies.fits')

    # Get fitting results
    narrow = Table.read('RUBIES/Results/REH-simul.fits')
    broad = Table.read('RUBIES/Results/REH-simul_broad.fits')
    # cauchy = Table.read('RUBIES/Results/REH-simul_cauchy.fits')
    results = join(
        narrow,
        broad,
        keys=['root', 'srcid'],
        table_names=('narrow', 'broad'),
        join_type='inner',
    )

    # Join with targets
    rubies = join(targets, results, keys=['root', 'srcid'])

    # Convert to pandas
    df = rubies.to_pandas()
    df = df.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # Multiprocess over unique targets and plot them
    with Pool(args.ncpu) as pool:
        pool.map(plot, df.groupby(['root', 'srcid']))


def plot(row):
    # Unpack
    (root, srcid), data = row

    # Check if output already exists
    # if os.path.exists(f'Comparison-PDF/{root}-{srcid}_comparison.pdf'):
    #     return

    # Get the rows with the highest grade
    bestrow = data[data['grade'] == data['grade'].max()]

    # If multiple rows exist, prefer the one with grating == 'G395M'
    if len(bestrow) > 1:
        bestrow = bestrow[bestrow['grating'] == 'G395M']

    # If z isn't -1, use it; otherwise, use zfit
    redshift_initial = (
        bestrow['zfit'].iloc[0] if bestrow['z'].iloc[0] == -1 else bestrow['z'].iloc[0]
    )
    opz = 1 + redshift_initial

    # Get fitting posterior
    with np.load(f'RUBIES/Results/{root}-{srcid}_full.npz', 'rb') as f:
        narrow = {key: f[key] for key in f}
    with np.load(f'RUBIES/Results/{root}-{srcid}_broad_full.npz', 'rb') as f:
        broad = {key: f[key] for key in f}

    if 'G395M_wavelength' not in broad:
        data = data[data['grating'] != 'G395M']
    if 'PRISM_wavelength' not in broad:
        data = data[data['grating'] != 'PRISM']

    Nspec = len(data)

    # Create figure
    fig, axes = pyplot.subplots(
        nrows=2 * Nspec,
        ncols=4,
        figsize=(15, 6 * Nspec),
        width_ratios=[1, 1, 1, 1],
        height_ratios=[5, 1] * Nspec,
        constrained_layout=True,
    )

    # Ensure axes is always a 2D array
    axes = np.atleast_2d(axes)

    # Iterate over each spectrum
    for i, row in enumerate(data.itertuples()):
        # Get the spectrum
        spectrum = Table.read(f'RUBIES/Spectra/{row.file}', 'SPEC1D')
        disp = row.grating
        wave = u.Quantity(spectrum['wave'])
        flux = spectrum['flux'].to(
            1e-20 * u.erg / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(wave)
        )
        err = spectrum['err'].to(
            1e-20 * u.erg / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(wave)
        )
        wave = wave.value

        # Plot insets
        for j, creg in enumerate(cont_regs):
            mask = (wave > creg[0] * opz) & (wave < creg[1] * opz)
            line = r'H$\alpha$+NII+SII' if j else r'H$\beta$+OIII'
            for k in range(2):
                ax = axes[2 * i, 2 * j + k]
                rax = axes[2 * i + 1, 2 * j + k]
                ax.plot(wave[mask], flux[mask], color='gray', lw=1, ds='steps-mid')
                ax.errorbar(
                    wave[mask], flux[mask], yerr=err[mask], fmt='none', color='gray'
                )
                ax.set(xlim=(creg[0] * opz, creg[1] * opz), xticklabels=[])
                rax.set_xlim(creg[0] * opz, creg[1] * opz)

                # Narrow:
                if k:
                    model = narrow
                    if i == 0:
                        ax.set_title(f'Narrow {line}')
                else:
                    model = broad
                    if i == 0:
                        ax.set_title(f'Broad {line}')
                model_wave = model[f'{disp}_wavelength']

                # Get best fit
                best = np.argmax(model['logL'])
                fs = model[f'{disp}_flux'][best]
                model_continuum = fs * model[f'{disp}_cont'][best]
                model_components = fs * model[f'{disp}_lines'][best]
                model_flux = model[f'{disp}_model'][best]
                model_err = np.std(model[f'{disp}_model'], axis=0)

                # Plot model
                model_mask = (model_wave > creg[0] * opz) & (model_wave < creg[1] * opz)
                ax.plot(
                    model_wave[model_mask],
                    model_flux[model_mask],
                    color='k',
                    ds='steps-mid',
                )
                ax.errorbar(
                    model_wave[model_mask],
                    model_flux[model_mask],
                    yerr=model_err[model_mask],
                    fmt='none',
                    color='k',
                )
                if model_mask.sum() == mask.sum():
                    rax.plot(
                        model_wave[model_mask],
                        (flux[mask].value - model_flux[model_mask]) / err[mask].value,
                        color='k',
                        ds='steps-mid',
                    )
                if mask.sum() and not np.all(np.isnan(flux[mask])):
                    percent = np.nanpercentile(flux[mask].value, 95)
                    ax.set_ylim(None, percent)

                # Plot components
                for component in model_components.T:
                    component_mask = np.logical_and(component != 0, model_mask)
                    if component_mask.sum() == 0:
                        continue
                    ax.plot(
                        model_wave[component_mask],
                        model_continuum[component_mask] + component[component_mask],
                        ds='steps-mid',
                    )
    # Add rest-frame axes
    for i, ax in enumerate(axes.flatten()):
        secax = ax.secondary_xaxis(
            'top', functions=(lambda x: x / opz, lambda x: x * opz)
        )
        ax.tick_params(axis='x', which='both', top=False)
        if (i // 4) % 2 == 1:
            secax.set(xticklabels=[])
        for line in [6564.61, 4862.68, 5008.24, 4960.295]:
            ax.axvline(line * opz / 1e4, color='k', ls='--', lw=1)

    # Compute FWHMs
    if np.isnan(data['HI_emission_6564.61_fwhm_narrow'].iloc[0]):
        lam = '4862.68'
    else:
        lam = '6564.61'

    fwhm = data[f'HI_emission_{lam}_fwhm_narrow'].iloc[0]
    fwhm_err = data[f'HI_emission_{lam}_fwhm_std_narrow'].iloc[0]
    fwhm_narrow = data[f'HI_emission_{lam}_fwhm_broad'].iloc[0]
    fwhm_narrow_err = data[f'HI_emission_{lam}_fwhm_std_broad'].iloc[0]
    fwhm_broad = data[f'HI_broad_{lam}_fwhm'].iloc[0]
    fwhm_broad_err = data[f'HI_broad_{lam}_fwhm_std'].iloc[0]

    # Compute BL probability
    pbroad = row.pbroad

    # Set labels
    fig.suptitle(
        rf'{root}-{srcid}: ($z = {redshift_initial:.3f}$) P(BL) = {pbroad:.2f}'
        + '\n'
        + rf'FWHW-single: {fwhm:.0f} $\pm$ {fwhm_err:.0f} FWHM-narrow: {fwhm_narrow:.0f} $\pm$ {fwhm_narrow_err:.0f}, FWHM-broad: {fwhm_broad:.0f} $\pm$ {fwhm_broad_err:.0f}, '
    )
    fig.supylabel(rf'$f_\lambda$ [{flux.unit:latex_inline}]')
    fig.supxlabel(rf'$\lambda$ [{u.um:latex_inline}]')

    # Save figure
    fig.savefig(f'RUBIES/Comparison-PDF/{root}-{srcid}_zoom.pdf')
    fig.savefig(f'RUBIES/Comparison-JPG/{root}-{srcid}_zoom.jpg', dpi=150)
    pyplot.close(fig)


if __name__ == '__main__':
    main()
