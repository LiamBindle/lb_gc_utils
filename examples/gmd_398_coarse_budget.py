import sys
import numpy as np
import xarray as xr
from pathlib import Path

from lbutil.constants import *
from lbutil.vertical_operators import regrid_to_pressure_grid, total_below
from lbutil.basic_operators import dmmr_to_moles, calc_air_mass, drop_non_dmmr_variables

data_dir = Path(sys.argv[1])
date = sys.argv[2]
out_dir = Path(sys.argv[3])

print('Processing columns and global totals')
print(f'  data_dir={data_dir}')
print(f'  date={date}')
print(f'  out_dir={out_dir}')

print("Loading datasets ...")
spc_mmr = xr.open_dataset(data_dir.joinpath(f'GCHP.SpeciesConc.{date}_1200z.nc4'))
met = xr.open_dataset(data_dir.joinpath(f'GCHP.StateMet.{date}_1200z.nc4'))
budget = xr.open_dataset(data_dir.joinpath(f'GCHP.Budget.{date}_1200z.nc4'))

# Keep only the relevant variables
relevant_spc = [
    'SpeciesConc_O3',
    'SpeciesConc_NO', 'SpeciesConc_NO2', 'SpeciesConc_HNO3', 'SpeciesConc_PAN',
    'SpeciesConc_CO', 'SpeciesConc_CH2O',
    'SpeciesConc_SO2', 'SpeciesConc_NH3',
]
spc_mmr = spc_mmr[relevant_spc]
spc_mmr = drop_non_dmmr_variables(spc_mmr)
spc_mol = dmmr_to_moles(spc_mmr, mass_dry_air=met.Met_AD)

print("Calculating tropospheric totals ...")
tropo_total = xr.apply_ufunc(
    total_below,
    spc_mol, met.Met_TropP, met.Met_PS1WET,
    input_core_dims=[['lev'], [], []],
    vectorize=True
)
global_tropo_total = tropo_total.sum(dim=('nf', 'Ydim', 'Xdim'))
global_tropo_total = global_tropo_total.rename({old_name: f'Global{old_name}' for old_name in global_tropo_total.keys()})

# strat_total = xr.apply_ufunc(
#     total_below,
#     spc_mol, met.Met_TropP, met.Met_PS1WET, kwargs=dict(above_instead=True),
#     input_core_dims=[['lev'], [], []],
#     vectorize=True
# )

print("Subsetting budget dataset ...")
# Prepare output dataset
budget_spc_names = [s.replace("SpeciesConc_", "") for s in relevant_spc]
relevant_budget = []
budget_varnames = list(budget.keys())
for region in ['Trop', 'PBL', 'Full']:
    for s in budget_spc_names:
        for name in ['Chemistry', 'EmisDryDep', 'Mixing', 'Convection', 'WetDep']:
            varname = f'Budget{name}{region}_{s}'
            if varname in budget_varnames:
                relevant_budget.append(varname)
budget = budget[relevant_budget]

print("Calculating budget totals ...")
for region in ['Trop', 'PBL', 'Full']:
    for s in budget_spc_names:
        proc_contrib = []
        for name in ['Chemistry', 'EmisDryDep', 'Mixing', 'Convection', 'WetDep']:
            varname = f'Budget{name}{region}_{s}'
            if varname in budget_varnames:
                proc_contrib.append(budget[varname])
        budget[f'BudgetSum{region}_{s}'] = sum(proc_contrib)

global_budget = budget.sum(dim=('nf', 'Ydim', 'Xdim'))
global_budget = global_budget.rename({old_name: f'Global{old_name}' for old_name in global_budget.keys()})

print("Writing output files ...")
ds_global = xr.Dataset()
ds_global.update(global_budget)
ds_global.update(global_tropo_total)
ds_global.to_netcdf(out_dir.joinpath(f'GCHP.ProcessedGlobalSums.{date}.nc'))

ds_trop = xr.Dataset()
ds_trop.update(tropo_total)
ds_trop.update(budget)
ds_trop.to_netcdf(out_dir.joinpath(f'GCHP.ProcessedColumns.{date}.nc'))

print('Done.')




# Temporary; reduce to simple dimensions
# spc_mol = spc_mol.isel(time=0, nf=0, Ydim=0, Xdim=0).squeeze()
# spc_mmr = spc_mmr.isel(time=0, nf=0, Ydim=0, Xdim=0).squeeze()
# met = met.isel(time=0, nf=0, Ydim=0, Xdim=0).squeeze()

# ozone_mol = spc_mol['SpeciesConc_O3'].values
#
# ozone_tropo_total = total_below(ozone_mol, met.Met_TropP, total_surface_pressure_in=met.Met_PS1WET)
# ozone_strat_total = total_below(ozone_mol, met.Met_TropP, total_surface_pressure_in=met.Met_PS1WET, above_instead=True)
# ozone_total = ozone_mol.sum()

# print('Ozone totals (trusted):')
# print(f'  column total:          {ozone_total:.5e} [mol]')
# print('Calculated totals:')
# print(f'  tropospheric column:   {ozone_tropo_total.item():.5e} [mol]')
# print(f'  stratospheric column:  {ozone_strat_total.item():.5e} [mol]')
# print(f'  total columns (check): {ozone_tropo_total.item() + ozone_strat_total.item():.5e} [mol]')

# Objective:
#   - Calculate PBL and TROP mass of tracer
#

# import numpy as np
# import matplotlib.pyplot as plt
# import xarray as xr
#
#
#
#
# species_mmr = xr.open_dataset('../sample_data/GCHP.SpeciesConc.20180101_1200z.nc4').isel(time=0, nf=0, Xdim=0, Ydim=0)
# species_mmr = drop_non_dmmr_variables(species_mmr)
# met = xr.open_dataset('../sample_data/GCHP.StateMet.20180101_1200z.nc4').isel(time=0, nf=0, Xdim=0, Ydim=0)
#
# species_moles = dmmr_to_moles(species_mmr, mass_dry_air=met.Met_AD)
#
# ozone = species_moles['SpeciesConc_O3'].values
# water = species_moles['SpeciesConc_H2O'].values
#
# new_pressure_grid = np.logspace(1, 4)[::-1]
# new_total_mass = calc_air_mass(new_pressure_grid, met.Met_AREAM2, hPa=True)
#
# new_ozone = regrid_to_pressure_grid(ozone, new_pressure_grid, total_surface_pressure_in=met.Met_PS1WET)
# new_water = regrid_to_pressure_grid(water, new_pressure_grid, total_surface_pressure_in=met.Met_PS1WET)
#
# new_water_mass = new_water * MOLAR_MASS_H2O
# new_dry_air_mass = new_total_mass - new_water_mass
# new_dry_air_moles = new_dry_air_mass / MOLAR_MASS_DRY_AIR
#
# new_ozone_mmr = new_ozone / new_dry_air_moles
#
# # Plot
# plt.figure()
# plt.scatter(species_mmr.SpeciesConc_O3.values, met.Met_PMID.values, label='Original profile')
# plt.scatter(new_ozone_mmr, (new_pressure_grid[:-1] + new_pressure_grid[1:])/2, label='Regridded profile')
# plt.yscale('log')
# plt.gca().invert_yaxis()
# plt.legend()
# plt.show()



