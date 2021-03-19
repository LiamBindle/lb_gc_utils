import sys
import numpy as np
import xarray as xr
from scipy.ndimage.filters import gaussian_filter
import json
import argparse
from pathlib import Path
import matplotlib.patches as patches


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
args = parser.parse_args()

ds1 = xr.open_dataset(args.i)
ds1 = ds1.squeeze().rename({n: 'pressure' for n in ds1.dims.keys() if n[:8] == 'extradim'})
pressure_edges=np.linspace(1000, 100, 37)
ds1['pressure'] = (pressure_edges[:-1] + pressure_edges[1:])/2
ds1['SpeciesConc_NOx'] = ds1['SpeciesConc_NO'] + ds1['SpeciesConc_NO2']

with open(args.plot_conf, 'r') as f:
    limits = json.load(f)

ylim = limits['zonal']['ylim']

if args.mean_along is not None:
    ds1 = ds1.mean(args.mean_along)

vmin = lambda name: limits['zonal'][name]['vmin']
vmax = lambda name: limits['zonal'][name]['vmax']

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
    levels = limits['zonal']['comparison_levels']
else:
    cmap = 'cividis'
    ds = ds1
    is_difference_plot=False

ds = ds.drop(('lon_bnds', 'lat_bnds', 'SpeciesConc_NO', 'SpeciesConc_NO2', 'SpeciesConc_H2O'))

if args.isel is not None:
    for k, v in zip(args.isel[:-1:2], args.isel[1::2]):
        ds = ds.isel(**{k:int(v)})

ds = ds.squeeze()


for v in ds.keys():
    plt.figure(figsize=limits['zonal']['figsize'])
    ax = plt.axes()
    if is_difference_plot:
        a = ds[v].values
        sigma=limits['zonal']['gaussian_filter_sigma']
        a = gaussian_filter(a, sigma)
        mask_cutoff=limits['zonal']['mask_cutoff_rel_to_vmax']
        a = np.ma.masked_where(a < vmax(v)*mask_cutoff, a)
        #plt.pcolormesh(ds.lat, pressure_edges, a, vmin=levels[0], vmax=levels[-1], cmap=cmap)
        plt.contourf(ds.lat, ds.pressure, a, levels=levels, cmap=cmap, extend='both')
        # plt.contour(ds.lat, ds.pressure, a, levels=levels, cmap=cmap, extend='both', linewidth=0.01)
    else:
        plt.pcolormesh(ds.lat, pressure_edges, ds[v].values, vmin=vmin(v), vmax=vmax(v), cmap=cmap)
    plt.gca().invert_yaxis()

    if args.draw_ibbox:
        upper_tropo = patches.Rectangle((10, 100), 55, 150, linewidth=1, edgecolor='k', facecolor='none', zorder=100)
        mid_tropo = patches.Rectangle((10, 250), 55, 650, linewidth=1, edgecolor='k', facecolor='none', zorder=100)
        lower_tropo = patches.Rectangle((10, 900), 55, 100, linewidth=1, edgecolor='k', facecolor='none', zorder=100)
        ax.add_patch(upper_tropo)
        ax.add_patch(mid_tropo)
        ax.add_patch(lower_tropo)

    plt.yticks([900, 500, 250])
    ax.tick_params(labelbottom=False)
    ax.tick_params(labelleft=False)
    plt.ylim(ylim)
    plt.tight_layout()
    ofile = Path(args.o).joinpath(f'{v}.{args.ofs}.png')
    plt.savefig(ofile, pad_inches=0.01,  bbox_inches='tight', dpi=300)






