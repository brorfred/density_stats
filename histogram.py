
import numpy as np
import xarray as xr

class Config(object):
    def __init__(self, bins=None):
        self.bins = 100
        self.binrange = np.log(10**-4), np.log(10**2)
        self.binlist = np.linspace(*self.binrange, self.bins)
        da = xr.open_dataarray("datafiles/darwin_longhurst_provinces.nc")
        self.regions = da.values
        self.lat = da.lat.values
        self.lon = da.lon.values

class Stats(Config):
    """
    
    ref:    https://stackoverflow.com/questions/46086663/
    """
    def __init__(self, bins=None):
        super().__init__()
        self.binval = self.binlist if bins is None else bins
        self._ord = np.argsort(self.binval)

    def mean_(self, freq):
        """Compute the arithmetic mean of a dataset provided in the form of a histogram.

        Parameters
        ----------
        freq : array_like
            1D array containing frequencies of values in the dataset in bins
            defined by self.binval or the 'bins' class attribute. 

        Returns
        -------
        result : float
            the arithmetic mean of the original dataset
        """
        try:
            return np.average(self.binval, weights = freq)
        except ZeroDivisionError:
            return np.nan
    
    def median_(self, freq):
        """Compute the median of a dataset provided in the form of a histogram.

        Parameters
        ----------
        freq : array_like
            1D array containing frequencies of values in the dataset in bins
            defined by self.binval or the 'bins' class attribute. 

        Returns
        -------
        result : float
            the median of the original dataset
        """
        return self.percentile(freq, 0.5)

    def mode_(self, freq): #in the strictest sense, assuming unique mode
        """Compute the mode of a dataset provided in the form of a histogram.

        Parameters
        ----------
        freq : array_like
            1D array containing frequencies of values in the dataset in bins
            defined by self.binval or the 'bins' class attribute. 

        Returns
        -------
        result : float
            the mode of the original dataset
        """
        return self.binval[np.argmax(freq)]

    def var_(self, freq):
        avg = self.mean_(freq)
        dev = freq * (self.binval - avg) ** 2
        return (dev.sum() / (freq.sum() - 1)).item()

    def std_(self, freq):
        """Compute the standard deviation of a dataset provided in the form of a histogram.

        Parameters
        ----------
        freq : array_like
            1D array containing frequencies of values in the dataset in bins
            defined by self.binval or the 'bins' class attribute. 

        Returns
        -------
        result : float
            the standard deviation of the original dataset
        """
        return np.sqrt(self.var_(freq))

    def moment_(self, freq, moment=1):
        """Compute the moment of a dataset provided in the form of a histogram.

        Parameters
        ----------
        freq : array_like
            1D array containing frequencies of values in the dataset in bins
            defined by self.binval or the 'bins' class attribute. 
        moment : int or array_like of ints, optional
            Order of central moment that is returned. Default is 1.

        Returns
        -------
        result : float
            the moment of the original dataset
        """
        n = (freq * (self.binval - self.mean_(freq)) ** moment) / freq.sum()
        d = self.var_(freq) ** (moment / 2)
        return moment / d

    def percentile(self, freq, q):
        """Compute a given percentile of a dataset provided in the form of a histogram.

        Parameters
        ----------
        freq : array_like
            1D array containing frequencies of values in the dataset in bins
            defined by self.binval or the 'bins' class attribute.
        q : array_like of float
            Percentile to compute, which must be between
            0 and 1.0 inclusive.

        Returns
        -------
        result : float
            the 'q' percentile of the original dataset
        """
        cdf = np.cumsum(freq[self._ord])
        return self.binval[self._ord][np.searchsorted(cdf, cdf[-1] * q)]

    def histogram(self, fld, mask=None):
        """Create histogram from an nd array using the class instance's bins
        
        Parameters
        ----------
        fld : array_like
            ND array of the original dataset

        Returns
        -------
        freq : float
            the 'q' percentile of the original dataset
        """
        mask = np.full(fld.shape, True) if mask is None else mask
        mask = np.isfinite(fld) & (fld>0) & mask
        cnt,_ = np.histogram(np.log(fld[mask]),bins=self.binlist)
        return cnt

def open_dataset():
    """Open necdf dataset with precalculated histograms"""
    return xr.open_dataset("datafiles/cci_darwin_monthly_histograms_100bins.nc")
