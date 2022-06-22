import os

import numpy as np
import pandas as pd

def percentiles(filename="indata/insitudb_v6_chl.h5"):
    """Calculate percentiles of insitu Chl"""
    perlist = [0.01,0.05,0.1,0.33,0.5,0.67,0.9,0.95,0.99]
    df = pd.read_hdf(filename)
    df["month"] = df.index.month
    datamat = np.zeros((df.regions.max()+1, 12, len(perlist)+1))
    for dg in df.groupby(["regions","month"]):
        reg,mn = dg[0]
        vec = np.log(dg[1].chl.dropna())
        try:
            prc = np.percentile(vec, perlist)
        except IndexError:
            continue
        datamat[reg,mn-1,1:] = prc
        datamat[reg,mn-1,0]  = len(vec)
    return datamat
