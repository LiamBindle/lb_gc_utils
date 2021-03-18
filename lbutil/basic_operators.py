import numpy as np

from lbutil.constants import *


def dmmr_to_moles(dmmr, moles_dry_air=None, mass_dry_air=None):
    if moles_dry_air is not None:
        pass
    elif mass_dry_air is not None:
        moles_dry_air = mass_dry_air / MOLAR_MASS_DRY_AIR
    else:
        raise RuntimeError("Insufficient information to calculate moles of dry air.")
    moles = dmmr*moles_dry_air
    return moles


def moles_to_dmmr(moles,):
    raise NotImplemented()


def drop_non_dmmr_variables(ds):
    return ds.filter_by_attrs(units=lambda v: v == 'mol mol-1 dry')


def calc_air_mass(pressure_edges, area, hPa=True):
    v = (pressure_edges[:-1] - pressure_edges[1:]) * np.atleast_1d(area) / GRAVITATIONAL_ACCELERATION
    if hPa:
        return v*100
    return
