import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from lbutil.constants import *
from lbutil.vertical_operators import regrid_to_pressure_grid
from lbutil.basic_operators import dmmr_to_moles, calc_air_mass, drop_non_dmmr_variables


species_mmr = xr.open_dataset('../sample_data/GCHP.SpeciesConc.20180101_1200z.nc4').isel(time=0, nf=0, Xdim=0, Ydim=0)
species_mmr = drop_non_dmmr_variables(species_mmr)
met = xr.open_dataset('../sample_data/GCHP.StateMet.20180101_1200z.nc4').isel(time=0, nf=0, Xdim=0, Ydim=0)

species_moles = dmmr_to_moles(species_mmr, mass_dry_air=met.Met_AD)

ozone = species_moles['SpeciesConc_O3'].values
water = species_moles['SpeciesConc_H2O'].values

new_pressure_grid = np.logspace(1, 4)[::-1]
new_total_mass = calc_air_mass(new_pressure_grid, met.Met_AREAM2, hPa=True)

new_ozone = regrid_to_pressure_grid(ozone, new_pressure_grid, total_surface_pressure_in=met.Met_PS1WET)
new_water = regrid_to_pressure_grid(water, new_pressure_grid, total_surface_pressure_in=met.Met_PS1WET)

new_water_mass = new_water * MOLAR_MASS_H2O
new_dry_air_mass = new_total_mass - new_water_mass
new_dry_air_moles = new_dry_air_mass / MOLAR_MASS_DRY_AIR

new_ozone_mmr = new_ozone / new_dry_air_moles

# Plot
plt.figure()
plt.scatter(species_mmr.SpeciesConc_O3.values, met.Met_PMID.values, label='Original profile')
plt.scatter(new_ozone_mmr, (new_pressure_grid[:-1] + new_pressure_grid[1:])/2, label='Regridded profile')
plt.yscale('log')
plt.gca().invert_yaxis()
plt.legend()
plt.show()



