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
spc_mmr = xr.open_dataset(data_dir.joinpath(f'GCHP.SpeciesConc.{date}.nc'))
met = xr.open_dataset(data_dir.joinpath(f'GCHP.StateMet.{date}.nc'))

# Keep only the relevant variables
relevant_spc = [
    'SpeciesConc_O3',
    'SpeciesConc_NO', 'SpeciesConc_NO2', 'SpeciesConc_HNO3', 'SpeciesConc_PAN',
    'SpeciesConc_CO', 'SpeciesConc_CH2O',
    'SpeciesConc_SO2', 'SpeciesConc_NH3',
    'SpeciesConc_H2O'
]
spc_mmr = spc_mmr[relevant_spc]
spc_mmr = drop_non_dmmr_variables(spc_mmr)
spc_mol = dmmr_to_moles(spc_mmr, mass_dry_air=met.Met_AD)

print("Calculating total mass of new grid ...")

pgrid = np.linspace(1000, 100, 37)
# pgrid = np.linspace(1000, 100, 19)

new_total_mass = xr.apply_ufunc(
    calc_air_mass,
    pgrid, met.Met_AREAM2, kwargs=dict(hPa=True),
    input_core_dims=[['dim0'], []],
    output_core_dims=[['pressure']],
    vectorize=True
)

print("Regridding to pressure grid ...")

pgrid_spc_mol = xr.apply_ufunc(
    regrid_to_pressure_grid,
    spc_mol, pgrid, met.Met_PS1WET,
    input_core_dims=[['lev'], ['dim0'], []],
    output_core_dims=[['pressure']],
    vectorize=True
)

new_water_mass = pgrid_spc_mol['SpeciesConc_H2O'] * MOLAR_MASS_H2O
new_dry_air_mass = new_total_mass - new_water_mass
new_dry_air_moles = new_dry_air_mass / MOLAR_MASS_DRY_AIR

pgrid_spc_mmr = pgrid_spc_mol / new_dry_air_moles


print("Writing output datasets ...")

pgrid_spc_mmr = pgrid_spc_mmr.transpose('time', 'pressure', 'nf', 'Ydim', 'Xdim')
pgrid_spc_mmr['dry_air_mol'] = new_dry_air_moles
pgrid_spc_mmr['dry_air_mass'] = new_dry_air_mass
pgrid_spc_mmr['water_mass'] = new_water_mass
pgrid_spc_mmr.to_netcdf(out_dir.joinpath(f'GCHP.PressureGridSpeciesConc.{date}.nc'))

pgrid_spc_mol = pgrid_spc_mol.transpose('time', 'pressure', 'nf', 'Ydim', 'Xdim')
pgrid_spc_mol = pgrid_spc_mol.rename({oldname: f'Moles{oldname}' for oldname in pgrid_spc_mol.keys()})
pgrid_spc_mol['dry_air_mol'] = new_dry_air_moles
pgrid_spc_mol['dry_air_mass'] = new_dry_air_mass
pgrid_spc_mol['water_mass'] = new_water_mass
pgrid_spc_mol.to_netcdf(out_dir.joinpath(f'GCHP.PressureGridSpeciesMoles.{date}.nc'))


# import matplotlib.pyplot as plt
# plt.figure()
#
# spc_mmr = spc_mmr.isel(time=0, nf=0, Ydim=0, Xdim=0)
# met = met.isel(time=0, nf=0, Ydim=0, Xdim=0)
#
# new_spc_mmr = pgrid_spc_mmr.isel(time=0, nf=0, Ydim=0, Xdim=0)
#
# plt.scatter(spc_mmr.SpeciesConc_O3.values, met.Met_PMID.values, label='Original profile')
# plt.scatter(new_spc_mmr.SpeciesConc_O3.values, (pgrid[:-1] + pgrid[1:])/2, label='Regridded profile')
# # plt.yscale('log')
# plt.gca().invert_yaxis()
# plt.ylim([1000, 100])
# plt.legend()
# plt.show()
#
#
#
# print('here')









