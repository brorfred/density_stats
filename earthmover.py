from calendar import month_abbr

import numpy as np
import pandas as pd
import pyemd

import histogram

class Hist(histogram.Config):
    """Class to deal with EMD's in distribution space"""
    def __init__(self):
        super().__init__()

    @property
    def distance_mat(self):
        bins = self.binlist  
        return np.abs(np.array([bins - bins[pos] for pos in range(len(bins))]))

    @property
    def mpp(self):
        if not hasattr(self, "_mpp"):
            self._mpp = projmap.Map("glob")
        return self._mpp

    def emd_distance(self, hist1, hist2):
        """Calculate EMD from two histograms"""
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        return pyemd.emd(hist1.astype(np.float64), hist2.astype(np.float64), 
            self.distance_mat)

    def emd_hist(self, monthly=True, darwin="mod"):
        """Calculate earth mover's distances from histograms"""
        ds = histogram.open_dataset()
        if darwin=="mod":
            darfld = "dr_chl_mask"
        else:
            darfld = "dr_sat_mask"
        if monthly:
            emdarr = np.zeros(ds.cci.shape[:-1])
            for mn in range(ds.cci.shape[0]): 
                for lid in range(ds.cci.shape[1]): 
                    emdarr[mn, lid] = self.emd_distance(
                        ds.cci[mn, lid, :].data,
                        ds[darfld][mn, lid, :].data)
        else:
            emdarr = np.zeros(ds.cci.shape[:-1])
            for lid in range(ds.cci.shape[1]): 
                emdarr[lid] = self.emd_distance(
                    ds.cci[:, lid, :].data.flat,
                    ds[darfld][:, lid, :].data.flat)
        return emdarr


def to_df(darwin="mod"):
    """Export EMD data as dataframe"""
    em = Hist()
    emarr = em.emd_hist(darwin=darwin)
    df = pd.DataFrame(emarr.T, columns=list(month_abbr)[1:])
    df.loc[0,:] = np.nan
    df["name"] = pd.read_hdf("datafiles/longhurst_meta.h5")["Province"]
    return df
