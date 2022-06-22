import pathlib

import matplotlib as mpl
import numpy as np
import pylab as pl
import pandas as pd
import xarray as xr
import matplotlib.gridspec as gridspec


import njord.ecoregions
from njord.utils import yrday

import insitu_chl
import histogram
import earthmover
ed = earthmover.Hist()

import taylor

binlist = np.linspace(np.log(10**-4), np.log(10**2), 100)
xposvec = (binlist[:-1] + binlist[1:]) / 2

binlist10 = np.linspace(np.log(10**-4), np.log(10**2), 1000)
xposvec10 = (binlist10[:-1] + binlist10[1:]) / 2

cobs = '#ff7f0e'; csat = '#1f77b4'; cmod = '#2ca02c'; crrs = '#d62728'

dm = insitu_chl.percentiles() 
figkw = dict(bbox_inches="tight", dpi=300)

def xaxis(ax=None):
    ax = pl.gca() if ax is None else ax
    ax.set_xticks(np.log([0.001, 0.01, 0.1,1,10]))
    ax.set_xticklabels(["", "10$^{-2}$", "10$^{-1}$","10$^{-0}$","10$^{1}$"])
    ax.set_xlim(np.log(0.001),np.log(100))  

def yaxis(ax=None):
    ax = pl.gca() if ax is None else ax
    ax.set_yticks(np.log([0.001, 0.01, 0.1,1,10]))
    ax.set_yticklabels(["", "10$^{-2}$", "10$^{-1}$","10$^{-0}$","10$^{1}$"])
    ax.set_ylim(np.log(0.001),np.log(100))  

def create_biomIDlist():
    biomlist = pd.read_hdf("datafiles/longhurst_meta.h5")["Biome"]
    biomIDlist = np.zeros(len(biomlist))
    for id,name in enumerate(biomlist.unique()):
        mask = biomlist == name
        biomIDlist[mask] = id
    return biomIDlist

def read_hdf_with_biomes():
    df = pd.read_hdf("datafiles/cci_darwin_insitu_chl.h5")
    biomlist = pd.read_hdf("datafiles/longhurst_meta.h5")["Biome"]
    biomIDlist = create_biomIDlist()
    for id,name in enumerate(biomlist.unique()):
        mask = biomlist == name
        biomIDlist[mask] = id
    df["biomes"] = 0
    for regid,biomid in enumerate(biomIDlist):
        mask = df.regions == regid
        df.loc[mask,"biomes"] = biomid
    return df



# Histogram plots

def histograms(reg=11, mn=4, ax=None, mod=True, emd=True):
    """Draw histogram panels"""
    ax = pl.gca() if ax is None else ax
    ds = histogram.open_dataset()
    if mn is None:
        vec_rrs = np.sum(ds["dr_sat_mask"][:, reg, :], axis=0)
        vec_mod = np.sum(ds["dr_chl_mask"][:, reg, :], axis=0)
        vec_cci = np.sum(ds["cci"        ][:, reg, :], axis=0)
    else:
        vec_rrs = ds["dr_sat_mask"][mn, reg, :]
        vec_mod = ds["dr_chl_mask"][mn, reg, :]
        vec_cci = ds["cci"        ][mn, reg, :]
    maxval = np.max([np.max(vec_cci), np.max(vec_rrs)])
    kw_fill = dict(alpha=0.20)
    ax.fill_between(ds.bins, vec_cci/np.sum(vec_cci), alpha=0.25, label=r"$Chl_{sat}$", color=csat)
    ax.plot(ds.bins, vec_cci/np.sum(vec_cci), "w", lw=0.5, alpha=0.7, c=csat)

    ax.fill_between(ds.bins, vec_rrs/np.sum(vec_rrs), label=r"$Chl_{Rrs}$", color=crrs, **kw_fill)
    ax.plot(ds.bins, vec_rrs/np.sum(vec_rrs), "w", lw=0.5, alpha=0.7, c=crrs)
    if mod:
        ax.fill_between(ds.bins, vec_mod/np.sum(vec_mod), alpha=0.25, label=r"$Chl_{mod}$", color=cmod)
        ax.plot(ds.bins, vec_mod/np.sum(vec_mod), "w", lw=0.5, alpha=0.7, c=cmod)
    xaxis(ax)

    ax.set_yticks([0.10,0.20])
    ax.set_ylim(0,0.25)
    #ax.legend()
    emd_rrs = ed.emd_distance(vec_cci.values, vec_rrs.values)
    emd_mod = ed.emd_distance(vec_cci.values, vec_mod.values)
    return emd_rrs,emd_mod

def all_histograms_all_months():
    """Generate histograms for all provinces and months"""
    figdir  = "figs/darwin_modis_histograms"
    lh = njord.ecoregions.Longhurst()    
    for reg in range(54):  #54
        pl.close("all")
        fig,_ = pl.subplots(4,3, sharex=True, sharey=True)
        pl.subplots_adjust(hspace=0, wspace=0)      
        for ax,mn in zip(fig.axes, range(1,13)):
            print(mn)
            histograms(reg=reg, mn=mn-1, ax=ax)
            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
        y1,y2 = ax.get_ylim()
        [fig.axes[mn].text(.1,y2*0.8, yrday.mnstr[mn]) for mn in range(12)]
        fig.suptitle(lh.region_names[reg], y=0.93, fontsize=7)
        fig.axes[0].legend(loc='upper center', bbox_to_anchor=(0.7, 1.20),
          fancybox=False, shadow=False, ncol=5, prop={'size': 6},frameon=False)
        pl.savefig(f"{figdir}/histmap_all_year_{reg:02}.png", **figkw)



# Bar figures

def bincount_percentile(vec, percentiles, bins):
    """Calculate percentiles from bincount vector"""
    if not np.isscalar(percentiles):
        return np.array([bincount_percentile(vec, per, bins=bins) 
            for per in  percentiles])
    vec10 =  np.interp(xposvec10, bins, vec) 
    ppos = np.nonzero(np.cumsum(vec10)/np.sum(vec10) <= percentiles)[0].max()
    return binlist10[ppos]

def percentile_plot_rad_bars(reg=22, mn=4, ax=None):
    """Create plot comparing percentiles between model and satellite"""
    perlist = [0.01,0.05,0.1,0.33,0.5,0.67,0.9,0.95,0.99]
    ax = pl.gca() if ax is None else ax

    ds = histogram.open_dataset()
    vec_mod = ds["dr_chl_mask"][mn, reg, :]
    vec_rad = ds["dr_sat_mask"][mn, reg, :]
    vec_cci = ds["cci"][mn, reg, :]
    try:
        plist_mod = bincount_percentile(vec_mod, perlist, bins=ds.bins)
        plist_rad = bincount_percentile(vec_rad, perlist, bins=ds.bins)
        plist_cci = bincount_percentile(vec_cci, perlist, bins=ds.bins)
    except ValueError:
        ax.set_ylim(0,1)
        return
    def bar(pos, per1, per2, color="b"):
      y = [0+pos, 0.30+pos, 0.30+pos, 0+pos]
      x = [per1, per1, per2, per2]
      ax.fill(x, y, alpha=0.2, c=color, ec=None,linewidth=0)

    for pos, plist, col in zip(
        [0.03,0.35,0.67], [plist_mod, plist_rad, plist_cci], [cmod,crrs,csat]):
        bar(pos, plist[0], plist[-1], color=col)
        bar(pos, plist[2], plist[-3], color=col)
        bar(pos, plist[3], plist[-4], color=col)
        ax.plot([plist[4], plist[4]], [pos+0.03,pos+0.26], "k", lw=0.5)

    kw = dict(fontsize=6, ha="right")
    emd_rrs = ed.emd_distance(vec_cci.values, vec_rad.values)
    emd_mod = ed.emd_distance(vec_cci.values, vec_mod.values)
    ax.text(np.log(7e1), 0.60, f"EMD$_{{Rrs}}$: {emd_rrs:.2f}", color=crrs, **kw)
    ax.text(np.log(7e1), 0.15, f"EMD$_{{mod}}$: {emd_mod:.2f}", color=cmod, **kw)

    #xaxis(ax=ax)
    ax.set_ylim(0,1)

def percentile_plot_year(reg=22, gs0=None):
    """Create percentile plot for a full year"""
    fig = pl.gcf()
    lh = njord.ecoregions.Longhurst()
    ax0 = pl.gca()
    pl.title(lh.region_names[reg], fontsize="x-small")
    if gs0 is None:
        gs1 = gridspec.GridSpec(12, 1, figure=fig)
    else:
        gs1 = gridspec.GridSpecFromSubplotSpec(12,1, subplot_spec=gs0, 
                hspace=0, wspace=0)
    for mn,gs in enumerate(gs1):
        ax = fig.add_subplot(gs)
        ax.set_yticks([])
        if int(mn/2) != mn/2:
            ax.set_facecolor("0.95")
        percentile_plot_rad_bars(reg=reg, mn=mn, ax=ax)
        ax.xaxis.label.set_alpha(0)
        pl.setp(ax.get_xticklabels(), color="w", alpha=0)
        ax.set_xlim(np.log(0.001),np.log(100))  

    ax0.set_xlabel("Chl (mg m$^{-3}$)", fontsize="x-small")
    ax0.set_ylim(0.5,12.5)
    ax0.set_yticks([4,7,10])
    ax0.set_yticklabels(["Apr", "Jul", "Oct"], fontsize="x-small", 
                         rotation='vertical', va="center") 
    xaxis(ax=ax0)
    pl.setp(ax0.get_xticklabels(), fontsize="x-small")

def all_percentile_plots():
    datadir = f"figs/percentile_plots_bars/"
    pathlib.Path(datadir).mkdir(parents=True, exist_ok=True)
    for reg in range(58):
        pl.close("all")
        percentile_plot_year(reg=reg)
        pl.savefig(f"{datadir}/percent_darwin_sat_{reg:02}.pdf",
                dpi=300, bbox_inches="tight")
        print(reg)
    
def cumsum_plot(reg=22, mn=4, ax=pl.gca()):
    """Create plot comparing percentiles between model and satellite"""
    ds = histogram.open_dataset()
    vec_mod = ds["dr_chl_mask"][mn, reg, :]
    vec_sat = ds["cci"][mn, reg, :]
    ax.plot(ds.bins, np.cumsum(vec_sat)/np.sum(vec_sat), label="CCI")
    ax.plot(ds.bins, np.cumsum(vec_mod)/np.sum(vec_mod), label="Darwin")
    

def percentile_shading(chl, perlist = [1,10,33,50,67,90,99], 
                       ax=None, y1=None, y2=None, c="b"):
    ax = pl.gca() if ax is None else ax
    y1 = pl.ylim()[0] if y1 is None else y1
    y2 = pl.ylim()[1] if y2 is None else y2
    plist = np.percentile(chl, perlist)
    yfill = [y1, y2, y2, y1]
    xfill = [plist[0], plist[0], plist[-1], plist[-1]]
    ax.fill(xfill, yfill, alpha=0.1, c=c)
    xfill = [plist[1], plist[1], plist[-2], plist[-2]]
    ax.fill(xfill, yfill, alpha=0.1, c=c)
    xfill = [plist[2], plist[2], plist[-3], plist[-3]]
    ax.fill(xfill, yfill, alpha=0.1, c=c)
    ax.plot([plist[3],plist[3]], [y1,y2], c, lw=1, alpha=0.4)

def plot_pdf(vec, label, cumsum=True, c=None, ax=None):
    ax = pl.gca() if ax is None else ax
    y,_ = np.histogram(vec, binlist)
    if cumsum:
        ax.plot(xposvec, np.cumsum(y)/np.sum(y), label=label, c=c)
    else:
        ax.plot(xposvec, y/np.sum(y), label=label, c=c)


def biomes_hists():

    df = read_hdf_with_biomes()
    font = {'weight':'normal', 'size':18}
    mpl.rc('font', **font)
    pl.close("all")
    fig,_ = pl.subplots(4,1, sharex=True, sharey=True, figsize=(9,9))
    pl.subplots_adjust(wspace=0, hspace=0)

    def panel(ax, biomeid, panel):
        mask = (~np.isnan(df.chl) & 
            ~np.isnan(df.cci_chl) &
            ~np.isnan(df.dr_chl) &
            ~np.isnan(df.dr_rad) &
            (df.biomes == biomeid))
        kws = dict(cumsum=True, ax=ax)
        plot_pdf(np.log(df.chl[mask]),     label="Chl$_{obs}$", c=cobs, **kws)
        plot_pdf(np.log(df.cci_chl[mask]), label="Chl$_{sat}$", c=csat, **kws)
        plot_pdf(np.log(df.dr_chl[mask]),  label="Chl$_{mod}$", c=cmod, **kws)
        plot_pdf(np.log(df.dr_rad[mask]),  label="Chl$_{Rrs}$", c=crrs, **kws)
        xaxis(ax=ax)
        ax.set_ylim(-0.1,1.1)
        percentile_shading(np.log(df.chl[mask]), ax=ax, c=cobs, y1=0.8, y2=1.1)
        percentile_shading(np.log(df.cci_chl[mask]), ax=ax, c=csat, y1=0.5, y2=0.8)
        percentile_shading(np.log(df.dr_chl[mask]), ax=ax, c=cmod, y1=0.2, y2=0.5)
        percentile_shading(np.log(df.dr_rad[mask]), ax=ax, c=crrs, y1=-0.1, y2=0.2)
        ax.text(np.log(5),0.0,panel)

    panel(fig.axes[0], 1, "A, Polar")
    panel(fig.axes[1], 2, "B, Westerlies")
    panel(fig.axes[2], 3, "C, Trades")
    fig.axes[2].set_ylabel("Density")
    panel(fig.axes[3], 4, "D, Coastal")
    pl.legend(fontsize="x-small")
    pl.savefig("figs/biomes_hist_insitu_cci_darwin.pdf", dpi=300)  


def multi_hist():

    font = {'weight':'normal', 'size':18}
    mpl.rc('font', **font)

    pl.close("all")
    fig,_ = pl.subplots(4,1, sharex=True, figsize=(9,9))
    df = pd.read_hdf("datafiles/cci_darwin_insitu_chl.h5")
    df = df[~np.isnan(df.chl) & ~np.isnan(df.cci_chl)]
    ax = fig.axes[0]
    plot_pdf(np.log(df.chl), label="Chl$_{obs}$",     c=cobs, cumsum=False, ax=ax)
    plot_pdf(np.log(df.cci_chl), label="Chl$_{sat}$", c=csat, cumsum=False, ax=ax)
    xaxis(ax=ax)
    ax.set_ylim(-0.01,0.15)
    percentile_shading(np.log(df.chl), ax=ax, c=cobs, y1=0.07, y2=0.15)
    percentile_shading(np.log(df.cci_chl), ax=ax, c=csat, y1=-0.01, y2=0.07)
    ax.xaxis.set_visible(False)
    ax.text(np.log(70),0.12,"A")
    
    ax = fig.axes[1]
    plot_pdf(np.log(df.chl), label="Chl$_{obs}$",     c=cobs, cumsum=True, ax=ax)
    plot_pdf(np.log(df.cci_chl), label="Chl$_{sat}$", c=csat, cumsum=True, ax=ax)
    xaxis(ax=ax)
    ax.set_ylim(-0.1,1.1)
    percentile_shading(np.log(df.chl), ax=ax, c=cobs, y1=0.5, y2=1.1)
    percentile_shading(np.log(df.cci_chl), ax=ax, c=csat, y1=-0.1, y2=0.5)
    pl.legend(fontsize="x-small")
    ax.xaxis.set_visible(False)
    ax.text(np.log(70),0.8,"B")

    df = df[~np.isnan(df.dr_chl) & ~np.isnan(df.dr_rad)]
    ax = fig.axes[2]
    plot_pdf(np.log(df.chl),     label="Chl$_{obs}$", c=cobs, cumsum=False, ax=ax)
    plot_pdf(np.log(df.cci_chl), label="Chl$_{sat}$", c=csat, cumsum=False, ax=ax)
    plot_pdf(np.log(df.dr_chl),  label="Chl$_{mod}$", c=cmod, cumsum=False, ax=ax)
    plot_pdf(np.log(df.dr_rad),  label="Chl$_{Rrs}$", c=crrs, cumsum=False, ax=ax)
    xaxis(ax=ax)
    ax.set_ylim(-0.01,0.15)
    percentile_shading(np.log(df.chl), ax=ax, c=cobs, y1=0.11, y2=0.15)
    percentile_shading(np.log(df.cci_chl), ax=ax, c=csat, y1=0.07, y2=0.11)
    percentile_shading(np.log(df.dr_chl), ax=ax, c=cmod, y1=0.03, y2=0.07)
    percentile_shading(np.log(df.dr_rad), ax=ax, c=crrs, y1=-0.01, y2=0.03)


    ax.xaxis.set_visible(False)
    ax.set_ylabel("Density")
    ax.text(np.log(70),0.12,"C")

    df = df[~np.isnan(df.dr_chl) & ~np.isnan(df.dr_rad)]
    ax = fig.axes[3]
    plot_pdf(np.log(df.chl),     label="Chl$_{obs}$", c=cobs, cumsum=True, ax=ax)
    plot_pdf(np.log(df.cci_chl), label="Chl$_{sat}$", c=csat, cumsum=True, ax=ax)
    plot_pdf(np.log(df.dr_chl),  label="Chl$_{mod}$", c=cmod, cumsum=True, ax=ax)
    plot_pdf(np.log(df.dr_rad),  label="Chl$_{Rrs}$", c=crrs, cumsum=True, ax=ax)
    xaxis(ax=ax)
    ax.set_ylim(-0.1,1.1)
    percentile_shading(np.log(df.chl), ax=ax, c=cobs, y1=0.8, y2=1.1)
    percentile_shading(np.log(df.cci_chl), ax=ax, c=csat, y1=0.5, y2=0.8)
    percentile_shading(np.log(df.dr_chl), ax=ax, c=cmod, y1=0.2, y2=0.5)
    percentile_shading(np.log(df.dr_rad), ax=ax, c=crrs, y1=-0.1, y2=0.2)


    pl.legend(fontsize="x-small")
    ax.text(np.log(70),0.8,"D")

    pl.xlabel("Chl (mg m$^{-3}$)")
    pl.subplots_adjust(hspace=0)
    #pl.title("InSitu matchups")    
    pl.savefig("figs/multi_hist_insitu_cci_darwin.pdf", dpi=300)  

def all_histograms_biomes(sea="atl"):
    """Generate all histograms"""
    figdir  = "figs/biomes_hists"
    lh = njord.ecoregions.Longhurst()    
    if "atl" in sea.lower():
        seaname = "Atlantic"
        reglist = range(3, 20)
        searepl = "Atlantic "
        seafile = "atlantic"
        biomtxt = True
    elif "pac" in sea.lower():
        seaname = "Pacific"
        reglist = range(32, 55)
        searepl = "Pacific "
        seafile = "pacific"
        biomtxt = True
    elif "trade" in sea.lower():
        seaname = "Trades"
        reglist = np.nonzero(lh._longh_names["Biome"].values == "Trades")[0]
        searepl = "Trades"
        seafile = "Trades"
        biomtxt = False
    elif "west" in sea.lower():
        seaname = "Westerlies"
        reglist = np.nonzero(lh._longh_names["Biome"].values == "Westerlies")[0]
        reglist = list(reglist) + [25,]
        searepl = "Westerlies"
        seafile = "Westerlies"
        biomtxt = False
    elif "polar" in sea.lower():
        seaname = "Polar"
        reglist = np.nonzero(lh._longh_names["Biome"].values == "Polar")[0]
        searepl = "Polar"
        seafile = "Polar"
        biomtxt = False    
    elif "ind" in sea.lower():
        seaname = "Indian Ocean"
        reglist = range(20,29)
        searepl = "Atlantic "
        seafile = "indian"
        biomtxt = True
    else:
        seaname = "Polar"
        reglist = [0,1,2,29,30,31,50,51,52,53]
        searepl = ""
        seafile = "polar"
        biomtxt = True
        
    pl.close("all")
    fig,_ = pl.subplots(4,4, sharex=False, sharey=True, figsize=(9,6))
    pl.subplots_adjust(hspace=0, wspace=0)

    for ax,reg in zip(fig.axes, reglist):
        print(reg)
        emd_rrs,emd_mod = histograms(reg=reg, mn=None, ax=ax)
        regname = lh.region_names[reg].replace("Province", "")
        regname = regname.replace(lh._longh_names["Basin"][reg], "")
        emd = f'EMD$_{{Rrs}}$: {emd_rrs:.2f}'
        ax.text(np.log(70), 0.1, emd, fontsize=7, ha="right", color='#d62728')
        emd = f'EMD$_{{mod}}$: {emd_mod:.2f}'
        ax.text(np.log(70), 0.125, emd, fontsize=7, ha="right", color='#2ca02c')
        ax.text(np.log(70), 0.20, regname, fontsize=7, ha="right")
        ax.text(np.log(70), 0.175, lh._longh_names["Basin"][reg] + " Ocean", fontsize=7, ha="right")
        if biomtxt:
            ax.text(np.log(70), 0.10, lh._longh_names["Biome"][reg], 
            fontsize=7, ha="right")


        #ax.text(np.log(1.5e-3), 0.05, regtype, fontsize=6, alpha=0.6)
    for ax in fig.axes:
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines['bottom'].set_color('0.5')
        ax.spines['top'].set_color('0.5') 
        ax.spines['right'].set_color('0.5')
        ax.spines['left'].set_color('0.5')
        ax.tick_params(axis='x', colors='0.5')
        ax.tick_params(axis='y', colors='0.5')
        ax.set_xticklabels(
            ["", "10$^{-2}$", "10$^{-1}$","10$^{-0}$","10$^{1}$"], color = 'k')
        ax.yaxis.set_ticklabels([0.10,0.20], color = 'k')
    fig.axes[0].legend(loc='upper center', bbox_to_anchor=(0.7, 1.42),
          fancybox=False, shadow=False, ncol=5, prop={'size': 8},frameon=False)
    y1,y2 = ax.get_ylim()
    fig.suptitle(seaname, y=0.93, fontsize="medium")

    if "pol" in sea.lower():
        [fig.delaxes(fig.axes[ax]) for ax in [15,14,13,12,11,10,9,8,7,6]]
        fig.text(0.5, 0.45, "Chl (mg m$^{-3}$)", size=7)
        fig.text(0.08, 0.7, "Density", size=7, 
            verticalalignment="center", rotation=90)

    pl.savefig(f"{figdir}/biomes_hists_{seafile}.png", **figkw)
    pl.savefig(f"{figdir}/biomes_hists_{seafile}.pdf", **figkw)

def histobar_months_4_examples():
    pl.close("all")
    fig = pl.figure(1,(9,9), constrained_layout=True)
    gs0 = gridspec.GridSpec(2, 2, figure=fig)
    pl.subplots_adjust(wspace=0.05)
    for pan,reg,gs in zip(["A","B","C","D"], [2,5,8,17], gs0):
        ax = fig.add_subplot(gs)
        percentile_plot_year(reg=reg, gs0=gs)
        if reg < 8:
            ax.xaxis.set_visible(False)
        if reg in [5,17]:
            ax.yaxis.set_visible(False)
        title = ax.get_title()
        ax.set_title(f"{pan}) {title}")
    pl.savefig(f"figs/province_year_bar_examples.pdf", **figkw)

def all():
    global_hists()
    return
    biomes_hists()
    all_histograms_biomes(sea="trades")
    all_histograms_biomes(sea="westerlies")
    all_histograms_biomes(sea="polar")

def mod_rrs_2dhist(clf=True, ax=None):
    if clf:
        pl.clf()
    ax = pl.gca() if ax is None else ax
    ds = xr.open_dataset("datafiles/mod_rrs_daily_histogram.nc")
    im = ax.pcolormesh(ds.mod_bins, ds.rrs_bins, np.sum(ds.hist, axis=0).T,
                  norm=mpl.colors.LogNorm(vmin=1, vmax=1e6))
    cb = pl.colorbar(im, ax=ax)
    cb.set_label("Number of values")
    xaxis(ax=ax)
    yaxis(ax=ax)
    ax.set_ylabel("Chl$_{Rrs}$ (mg m$^{-3}$)")
    ax.set_xlabel("Chl$_{mod}$ (mg m$^{-3}$)")
    ax.plot([np.log(0.001),np.log(100)], [np.log(0.001),np.log(100)], "0.5", lw=1)  

