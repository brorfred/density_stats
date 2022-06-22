
import numpy as np
import xarray as xr
import pandas as pd
import pyresample as pr
from fast_histogram import histogram1d

import resample

bins = 100
binrange = np.log(10**-4), np.log(10**2)
binlist = np.linspace(*binrange, bins)

def open_dataset(date="2000-01-01"):
    """Open OC-CCI dataset as xarray dataarray via opendap"""
    #dtm = np.datetime64(date) if type(date) is str else date
    url = "https://www.oceancolour.org/thredds/dodsC/CCI_ALL-v5.0-DAILY"
    ds = xr.open_dataset(url)
    return ds["chlor_a"].sel(time=date)

def setup_grid():
    """Create matrices with latitudes and longitudes for the t-coords"""
    i0t,imt,j0t,jmt = (0000 ,8640, 0, 4320)
    incr  = 360.0/imt
    jR    = np.arange(j0t, jmt)
    iR    = np.arange(i0t, imt)
    latvec = (  90 - jR*incr - incr/2)[::-1]
    lonvec = (-180 + iR*incr + incr/2)
    lons,lats = np.meshgrid(lonvec, latvec)
    grid = pr.geometry.GridDefinition(lons=lons, lats=lats)
    grid.ivec =  np.arange(grid.shape[1])
    grid.jvec =  np.arange(grid.shape[0])
    grid.iarr,grid.jarr = np.meshgrid(grid.ivec, grid.jvec)
    return grid

def setup_darwin_grid():
    """Create matrices with latitudes and longitudes for the t-coords"""
    latvec = np.arange(-79.5, 80.5)
    lonvec = np.arange(-179.5, 180.5)
    lons,lats = np.meshgrid(lonvec, latvec)
    grid = pr.geometry.GridDefinition(lons=lons, lats=lats)
    grid.ivec =  np.arange(grid.shape[1])
    grid.jvec =  np.arange(grid.shape[0])
    grid.iarr,grid.jarr = np.meshgrid(grid.ivec, grid.jvec)
    return grid

def fields_to_histograms(date1="2001-01-01", date2="2001-12-31"):
    """Read CCI fields and convert to histograms, separating month and region"""
    longh   = xr.open_dataset("indata/longhurst_darwin.nc")
    reglist = np.unique(longh.regions.data.astype(int))
    histmat = np.zeros((12, len(reglist), len(binlist)))
    griddr  = setup_darwin_grid()
    grid    = setup_grid()
    for dtm in pd.date_range(date1, date2):
        fld = resample.coarsegrid(grid, open_dataset(date=dtm).data, griddr)
        for npos,reg in enumerate(reglist):
            mask = np.isfinite(fld) & (fld>0) & (reg==longh.regions.data)
            cnt = histogram1d(np.log(fld[mask]), range=binrange, bins=bins)
            histmat[dtm.month,npos,:] = cnt
    print(dtm)
    return histmat

"""
def longhurst_nc_file():
    ds = longhurst.open_dataset()
    griddr = setup_darwin_grid()
    dsgr = xr.Dataset( coords={"lat":griddr.lats[:,0], "lon":griddr.lons[0,:]})
    for  key in ds.data_vars:
        arr = resample.coarsegrid(longhurst.setup_grid(), ds[key].data, griddr) 
        dsgr[key] = (("lat","lon"), arr)
    dsgr.to_netcdf("indata/longhurst_darwin.nc")
"""
