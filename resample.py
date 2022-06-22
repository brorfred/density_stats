
import numpy as np
from scipy.spatial import cKDTree as KDTree
import pyresample as pr

def coarsegrid(source_geo_def, data, target_geo_def, ij=None, coarse_fill=True):
    """
    Resamples data using kd-tree gaussian weighting neighbour approach.

    Parameters
    ----------
    source_geo_def : object
        Geometry definition of source
    data : numpy array
        Array of single channel data points or
        (source_geo_def.shape, k) array of k channels of datapoints
    target_geo_def : object
        Geometry definition of target
    with_stats : bool, optional
        Calculate uncertainty estimates

    Returns
    -------
    data : numpy array (default)
        Source data resampled to target geometry
    data, stddev, counts : numpy array, numpy array, numpy array (if with_uncert == True)
        Source data resampled to target geometry.
        Weighted standard devaition for all pixels having more than one source value
        Counts of number of source values used in weighting per pixel
       
    """
    if ij is None:
        dist,ij = kdquery(source_geo_def, target_geo_def)
        #dist,ij = kdquery(target_geo_def, source_geo_def)
    data_vec = data.flat
    mask = np.nonzero(np.isfinite(data_vec))[0]
    jmax,imax = target_geo_def.shape
    valsum = np.bincount(ij[mask], weights=data_vec[mask], minlength=imax*jmax)
    valcnt = np.bincount(ij[mask], minlength=imax*jmax)
    target_data = (valsum / valcnt).reshape(jmax, imax)
    if coarse_fill:
        near_data = pr.kd_tree.resample_nearest(
            source_geo_def, data, target_geo_def, radius_of_influence=1e6)
        target_data[np.isnan(target_data)]= near_data[np.isnan(target_data)]
    return target_data

def kdquery(source_geo_def, target_geo_def):
    """
    Resamples data using kd-tree gaussian weighting neighbour approach.

    Parameters
    ----------
    source_geo_def : object
        Geometry definition of source
    target_geo_def : object
        Geometry definition of target
  
    Returns
    -------
    dist : array of floats
        The distances to the nearest neighbors. If x has shape tuple+(self.m,),
        then d has shape tuple if k is one, or tuple+(k,) if k is larger than
        one. Missing neighbors are indicated with infinite distances. If k is
        None, then d is an object array of shape tuple, containing lists of
        distances. In either case the hits are sorted by distance 
        (nearest first).
    ij : array of integers
        The locations of the neighbors in self.data. i is the same shape as d.
  
       
    """
    kd = KDTree(np.vstack((target_geo_def.lons.flat, target_geo_def.lats.flat)).T)
    return kd.query(np.vstack((source_geo_def.lons.flat, source_geo_def.lats.flat)).T)
