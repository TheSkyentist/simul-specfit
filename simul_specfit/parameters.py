"""
Utility functions for the simulation and fitting of spectra
"""

# Import packages
import copy

# Astronomy packages
from astropy import units as u, constants as consts

# Numerical packages
import numpy as np
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

# Spectra class
from simul_specfit import defaults
from simul_specfit.spectra import Spectra


def configToMatrices(config: dict) -> tuple[BCOO, BCOO, BCOO]:
    """
    Convert the configuration to sparse matrices for the model

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    tuple[BCOO, BCOO, BCOO]
        Redshift, dispersion, and flux transformation matrice
    """

    # Keep track of total line index
    i = 0

    # Keep track of unique index (i) and index pair (inds) between unique and total
    i_f, f_inds, fluxes = 0, [], []  # for flux also keep track of the ratio
    i_z, z_inds = 0, []
    i_σ, σ_inds = 0, []

    # Iterate over groups, species, and lines
    for group in config['Groups'].values():
        for species in group['Species']:
            # Check if any fluxes in species are tied
            if not all([line['RelStrength'] is None for line in species['Lines']]):
                # Assign special index for tied fluxes
                species_f = i_f
                i_f += 1  # Increment for not tied fluxes
            # Iterate over lines
            for line in species['Lines']:
                # Keep track of nonzero matrix elements
                z_inds.append([i_z, i])
                σ_inds.append([i_σ, i])

                # Associate line with it's total index
                line['Index'] = i

                # If the flux is not tied, increment
                if line['RelStrength'] is None:
                    fluxes.append(1)
                    f_inds.append([i_f, i])
                    i_f += 1
                else:
                    fluxes.append(line['RelStrength'])
                    f_inds.append([species_f, i])

                # Increment line index
                i += 1

            # If Group is not tied, increment
            if not group['TieRedshift']:
                i_z += 1
            if not group['TieDispersion']:
                i_σ += 1

        # Increment between groups if we didn't already and species is not empty
        if group['Species']:
            i_z += 1
        if group['Species']:
            i_σ += 1

    # Create Sparce Matrices for Z and F
    Z = BCOO((jnp.ones(i, int), z_inds), shape=(i_z, i))
    F = BCOO((fluxes, f_inds), shape=(i_f, i))

    # Iterate again to decouple additional components sigma
    add_inds = []
    for group in config['Groups'].values():
        for species in group['Species']:
            for line in species['Lines']:
                # Check if there are additional components
                if 'AdditionalComponents' in species:
                    # Iterate over additional components
                    for comp, dest in species['AdditionalComponents'].items():
                        # Iterate again to find the additional components
                        for aSpecies in config['Groups'][dest]['Species']:
                            if (
                                aSpecies['Name'] != species['Name']
                                or aSpecies['LineType'] != comp
                            ):
                                continue
                            for addLine in aSpecies['Lines']:
                                if not addLine['Wavelength'] == line['Wavelength']:
                                    continue
                                add_inds.append([line['Index'], addLine['Index']])

    # If no additional components, return
    if len(add_inds) == 0:
        # Sparse matrix for sigma
        Σ = BCOO((jnp.ones(i, int), σ_inds), shape=(i_σ, i))
        Σadd = jnp.ones((0, i))
        Σuadd = jnp.ones((0, 0))
        return F, Z, (Σ, Σadd, Σuadd)

    # Make from total to unique index
    unique_map = {i[1]: i[0] for i in σ_inds}

    # Convert first index to unique index
    uadd_inds = [[unique_map[i[0]], i[1]] for i in add_inds]

    # Check if any add_inds are in σ_inds
    translation = {}
    for add_ind in uadd_inds:
        # If a component is tied within it's group, fix it
        if add_ind in σ_inds:
            # Translate the σ index to the next available one
            # If the initial index is not in translation, add it
            if add_ind[0] not in translation:
                translation[add_ind[0]] = i_σ
                i_σ += 1
            # Get the relevant σ index
            σ_ind = σ_inds[σ_inds.index(add_ind)]
            # Update the inbound σ index
            σ_ind[0] = translation[σ_ind[0]]

    # Get new mapping from total to unique index
    unique_map = {i[1]: i[0] for i in σ_inds}

    # Get unique indices that go to unique add indices
    uadd_uinds = np.unique(
        [[i[0], unique_map[i[1]]] for i in uadd_inds], axis=0
    ).tolist()

    # Remove the indices that are in add_in
    σ_inds_nar = [
        σ_i for i, σ_i in enumerate(σ_inds) if all(i != a[1] for a in add_inds)
    ]
    σ_inds_add = [
        σ_i for i, σ_i in enumerate(σ_inds) if any(i == a[1] for a in add_inds)
    ]

    # Create sparce matrix from nonempty rows
    σ_inds_nar, n = noEmptyRows(σ_inds_nar)
    Σ = BCOO((jnp.ones(len(σ_inds_nar), int), σ_inds_nar), shape=(n, i))

    # Create sparce matrix from nonempty rows
    σ_inds_add, nadd = noEmptyRows(σ_inds_add)
    Σadd = BCOO((jnp.ones(len(σ_inds_add), int), σ_inds_add), shape=(nadd, i))

    # Create sparce matrix from nonempty rows
    uadd_uinds, _ = noEmptyRows(uadd_uinds)
    uadd_uinds, _ = noEmptyRows([[u[1], u[0]] for u in uadd_uinds])
    uadd_uinds = [[u[1], u[0]] for u in uadd_uinds]
    Σuadd = BCOO((jnp.ones(len(uadd_uinds), int), uadd_uinds), shape=(n, nadd))

    return F, Z, (Σ, Σadd, Σuadd)


def noEmptyRows(indices: list[list[int]]) -> (list[list[int]], int):
    """
    Remake the indices such that there are no empty rows in the matrix

    Get
    Parameters
    ----------
    indices : list
        List of indices

    Returns
    -------
    list
        Updated list of indices
    int
        Number of non-zero rows
    """

    # Step 1: Extract unique row indices (non-empty rows)
    non_zero_rows = sorted(
        set(row for row, _ in indices)
    )  # Unique rows with non-zero values

    # Step 2: Create a mapping from old row indices to new contiguous row indices
    row_map = {old: new for new, old in enumerate(non_zero_rows)}

    # Step 3: Remap the row indices using the mapping
    new_indices = [[row_map[row], col] for row, col in indices]

    # Step 4: Return the new indices and the number of non-zero rows
    return new_indices, len(non_zero_rows)


def restrictConfig(
    config: dict, spectra: Spectra, linedet: u.Quantity = defaults.LINEDETECT
) -> list:
    """
    Restrict the configuration to only include lines that are covered by the spectra

    Parameters
    ----------
    config : dict
        Configuration of emission lines
    linedet : u.Quantity, optional
        Padding around the lines necessary to cover the line
        In velocity space

    Returns
    -------
    list
        Updated configuration
    """

    # Set the default line type as narrow
    for group in config['Groups'].values():
        for species in group['Species']:
            if 'LineType' not in species:
                species['LineType'] = 'narrow'

    # Add additional components
    new_config = copy.deepcopy(config)
    for group in config['Groups'].values():
        for species in group['Species']:
            if 'AdditionalComponents' in species:
                # Iterate over additional components
                for comp, dest in species['AdditionalComponents'].items():
                    # Add it to the correct group
                    new_group = new_config['Groups'][dest]

                    # Get copy of species
                    new_species = copy.deepcopy(species)

                    # Remove additional component and add LineType
                    new_species.pop('AdditionalComponents')
                    new_species['LineType'] = comp

                    # Add the new species
                    new_group['Species'].append(new_species)

    # Initialize dictionary
    config = copy.deepcopy(new_config)

    # Effective resolution
    lineres = (linedet / consts.c).to(u.dimensionless_unscaled).value

    # Loop over config
    new_groups = {}
    for gname, group in config['Groups'].items():
        new_species = []
        for species in group['Species']:
            new_lines = []
            for line in species['Lines']:
                # Compute line wavelength
                linewav = (line['Wavelength'] * u.Unit(config['Unit'])).to(
                    spectra.λ_unit
                )

                # Redshift the line
                linewav = linewav * (1 + spectra.redshift_initial)
                linewidth = linewav * lineres

                # Compute boundaries
                low, high = (linewav - linewidth).value, (linewav + linewidth).value

                # Check coverage
                if jnp.logical_or.reduce(
                    jnp.array([s.coverage(low, high).any() for s in spectra.spectra])
                ):
                    new_lines.append(line)

            # Add species only if it has remaining lines
            if new_lines:
                species['Lines'] = new_lines
                new_species.append(species)

        # Add group only if it has remaining species
        if new_species:
            group['Species'] = new_species
            new_groups[gname] = group

    # Return the updated config
    return config
