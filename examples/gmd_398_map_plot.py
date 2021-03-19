import sys
import numpy as np
import xarray as xr
from scipy.ndimage.filters import gaussian_filter
import json
import argparse
from pathlib import Path

import cartopy.crs as ccrs



import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='input file', required=True)
parser.add_argument('--mean_along', help='dimensions to take the mean across', default=None)
parser.add_argument('--isel', help='index select', nargs='+', default=None)
parser.add_argument('--compare_to', help='compare to this file', default=None)
parser.add_argument('--plot_conf', help='plot config json file', default='plot_conf.json')
parser.add_argument('--draw_ibbox', action='store_true')
parser.add_argument('-o', help='output directory', required=True)
parser.add_argument('-ofs', help='output file suffix', required=True)
parser.add_argument('--hPa500', action='store_true')
args = parser.parse_args()

ds1 = xr.open_dataset(args.i)
ds1 = ds1.squeeze().rename({n: 'pressure' for n in ds1.dims.keys() if n[:8] == 'extradim'})
pressure_edges=np.linspace(1000, 100, 37)
ds1['pressure'] = (pressure_edges[:-1] + pressure_edges[1:])/2
ds1['SpeciesConc_NOx'] = ds1['SpeciesConc_NO'] + ds1['SpeciesConc_NO2']

with open(args.plot_conf, 'r') as f:
    limits = json.load(f)

if args.mean_along is not None:
    ds1 = ds1.mean(args.mean_along)

if args.hPa500:
    level = 19
    which='500hPa'
else:
    level = 4
    which = '900hPa'

vmin = lambda name: limits[which][name]['vmin']
vmax = lambda name: limits[which][name]['vmax']

if args.compare_to is not None:
    ds2 = xr.open_dataset(args.compare_to)
    ds2 = ds2.squeeze().rename({n: 'pressure' for n in ds2.dims.keys() if n[:8] == 'extradim'})
    ds2['pressure'] = (pressure_edges[:-1] + pressure_edges[1:]) / 2
    ds2['SpeciesConc_NOx'] = ds2['SpeciesConc_NO'] + ds2['SpeciesConc_NO2']

    if args.mean_along is not None:
        ds2 = ds2.mean(args.mean_along)

    ds = (ds1-ds2)/ds2*100
    cmap = 'RdBu_r'
    is_difference_plot=True
    levels = limits[which]['comparison_levels']
else:
    cmap = 'cividis'
    ds = ds1
    is_difference_plot=False

ds = ds.drop(('lon_bnds', 'lat_bnds', 'SpeciesConc_NO', 'SpeciesConc_NO2', 'SpeciesConc_H2O'))

if args.isel is not None:
    for k, v in zip(args.isel[:-1:2], args.isel[1::2]):
        ds = ds.isel(**{k:int(v)})

ds = ds.isel(pressure=level)
ds = ds.squeeze()


for v in ds.keys():
    plt.figure(figsize=limits[which]['figsize'])
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines(linewidth=0.5)
    if is_difference_plot:
        a = ds[v].values
        sigma=limits[which]['gaussian_filter_sigma']
        a = gaussian_filter(a, sigma)
        mask_cutoff=limits[which]['mask_cutoff_rel_to_vmax']
        a = np.ma.masked_where(a < vmax(v)*mask_cutoff, a)
        plt.contourf(ds.lon, ds.lat, a, levels=levels, cmap=cmap, extend='both')
        # plt.contour(ds.lon, ds.lat, a, levels=levels, cmap=cmap, extend='both', linewidth=0.1)
    else:
        plt.pcolormesh(ds.lon, ds.lat, ds[v].values, vmin=vmin(v), vmax=vmax(v), cmap=cmap)


    if args.draw_ibbox:
        ax.plot([-170.5]*2, [10, 65], transform=ccrs.PlateCarree(), color='r', linewidth=1)
        ax.plot([-35.5]*2, [10, 65], transform=ccrs.PlateCarree(), color='r', linewidth=1)

    plt.tight_layout()
    ofile = Path(args.o).joinpath(f'{v}.{args.ofs}.png')
    plt.savefig(ofile, pad_inches=0.01,  bbox_inches='tight', dpi=300)
    # plt.show()






