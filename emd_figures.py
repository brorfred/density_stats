from calendar import month_name, month_abbr
import pathlib

import numpy as np
import pylab as pl
import pandas as pd
from matplotlib import cm

import paper_figures
import earthmover 

pathlib.Path("figs/emd").mkdir(parents=True, exist_ok=True)

def global_map(emdarr=None, month=6, darwin="mod"):
    emd = earthmover.Hist()
    emdarr = emd.emd_hist(darwin=darwin) if emdarr is None else emdarr
    emdmap = emd.regions * np.nan
    for lid in np.unique(emd.regions):
        emdmap[emd.regions == lid] = emdarr[month-1, lid] 
    emdmap[emd.regions==0] = np.nan
    pl.clf()
    emd.mpp.pcolor(emd.lon, emd.lat, emdmap,
        cmap=cm.BuPu, colorbar=True, vmin=0, vmax=2)                                              
    emd.mpp.nice()
    emd.mpp.ax.set_title(month_name[month])
    return emdmap

def monthly_maps():
    """Figure """
    emd = earthmover.Hist()
    emdarr = emd.emd_hist()
    for month in range(1,13): 
        global_map(emdarr, month=month) 
        pl.savefig(f"figs/emd/emd_map_{month_abbr[month]}.png") 

def biomes_scatter(ax=None, clf=True):
    if clf: pl.clf()
    ax = pl.gca() if ax is None else ax
    biomes = pd.read_hdf("datafiles/longhurst_meta.h5")["Biome"]
    ed = earthmover.Hist()
    mod = ed.emd_hist()
    rrs = ed.emd_hist(darwin="rrs")
    colorlist = ["w", "w", "tab:purple", "tab:red", "tab:blue"]
    for c,bio in zip(colorlist, np.unique(biomes)):
        mask = biomes == bio
        alpha = 0.25 if "green" in bio else 0.8
        ax.scatter(rrs[:,mask], mod[:,mask], 5, c=c, alpha=alpha)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 3.5)
    ax.set_xlabel("EMD Chl$_{sat}$ vs Chl$_{rrs}$")
    ax.set_ylabel("EMD Chl$_{sat}$ vs Chl$_{mod}$")
    ax.plot([0,4],[0,4], "0.5", lw=1)
    pl.savefig("figs/emd_scatter_provinces.pdf", bbox_inches="tight")


def emd_scatter_2d_hist():
    pl.close("all")
    fig,_ = pl.subplots(1,2, figsize=[10,5])
    biomes_scatter(ax=fig.axes[1], clf=False)
    fig.axes[1].text(0.9,0.1, "B", transform = fig.axes[1].transAxes)
    distributions_paper1.mod_rrs_2dhist(clf=False, ax=fig.axes[0])
    fig.axes[0].text(0.9,0.1, "A", transform = fig.axes[0].transAxes)
    pl.savefig("figs/emd_2Dhist_panels.pdf", bbox_inches="tight")

def biomes():
    """Compare Rrs and mod vs sat EMD"""
    binlist = paper_figures.binlist
    df = paper_figures.read_hdf_with_biomes()
    df = df[["chl", "dr_rad", "dr_chl", "cci_chl", "biomes"]].dropna()
    ed = earthmover.Hist()
    
    def emd(df):
        chl_sat,_ = np.histogram(df["cci_chl"], binlist)
        chl_mod,_ = np.histogram(df["dr_chl"], binlist)
        chl_obs,_ = np.histogram(df["chl"], binlist)
        chl_rrs,_ = np.histogram(df["dr_rad"], binlist)
        obs_sat = ed.emd_distance(chl_obs, chl_sat)
        obs_rrs =  ed.emd_distance(chl_obs, chl_rrs)
        obs_mod = ed.emd_distance(chl_obs, chl_mod)
        return obs_sat, obs_rrs, obs_mod

    sat,rrs,mod = emd(df)
    print(f"Global: obs_sat={sat:.2}, obs_rrs={rrs:.2}, obs_mod={mod:.2}")
    sat,rrs,mod = emd(df[df.biomes==1])
    print(f"Polar: obs_sat={sat:.2}, obs_rrs={rrs:.2}, obs_mod={mod:.2}")
    sat,rrs,mod = emd(df[df.biomes==2])
    print(f"Westerlies: obs_sat={sat:.2}, obs_rrs={rrs:.2}, obs_mod={mod:.2}")
    sat,rrs,mod = emd(df[df.biomes==3])
    print(f"Trades: obs_sat={sat:.2}, obs_rrs={rrs:.2}, obs_mod={mod:.2}")
    sat,rrs,mod = emd(df[df.biomes==4])
    print(f"Coastal: obs_sat={sat:.2}, obs_rrs={rrs:.2}, obs_mod={mod:.2}")


