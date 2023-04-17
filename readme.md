
# Code for Model-Data comparisons based on distributions

This repository includes code to analyze distributions of geophysical data as described in the XXX publication. We provide the following functionaly:

* Calculate statistics from histograms (histogram.py)
* Create Taylor diagrams from histograms (taylor.py)
* Calculate Earth Mover's distances from histograms (earthmover.py)
* Generate figures for the paper (paper_figures.py)
* Some simple example of how to generate histograms from netcdf files.

The repository also includes all necessary data as histograms used in the analysis presented in the paper. The raw data needed to generate the histograms are accessible from reposotories referenced in the paper.

The `figures_in_paper` contains all figures not included in the paper.

### Prerequisites

The following pypi packages are necessary for the code to work:

* xarray
* scipy
* numpy
* matplotlib
* netcdf4
* pytables
* h5py
* cartopy
* pyresample
* projmap

Cartopy can be challenging to install. Please see the [cartopy](https://scitools.org.uk/cartopy/docs/latest/installing.html) website for instructions.



### Installation

The easiest way to set up all prerequisites including cartopy is via [conda](https://conda.io):

1. Create a new conda environment
```sh
conda env create -f environment.yml
```
2. Activate the environment 
```sh
conda activate modeldata_GBC
```
3. Start the interactive ipython shell
```sh
ipython
```



<!-- USAGE EXAMPLES -->
## Usage

Thesa are some few examples of how to use the modules. The main general is to generate, perform statistics, and analysis on diferent kinds of histograms.

### Calculate histograms from opendap/netcdf fields
```python
import retrieve_cci

histmat = retrieve_cci.fields_to_histograms(date1="2001-01-01", date2="2001-12-31")
#histmat has the dimensions (months,regions,bins)
```
Calculate histograms for OC-CCI is very time consuming and doing it for Darwin requires specialized scripts. We provide pre-generated data in the netcdf file ```datafiles/cci_darwin_monthly_histograms_100bins.nc``` or via a method:

### Load pre-calculated histograms
```python
import histogram

ds = histogram.open_dataset()
# The dataset contains the following datavars:
#   cci         OC-CCI Chl
#   dr_chl      Darwin Chl, same as Chl_mod in the paper
#   dr_sat      Darwin Chl from Rrs, same as Chl_Rrs in the paper
#   dr_chl_mask Chl_mod masked by removing pixels invalid in OC-CCI
#   dr_sat_mask Chl_Rrs masked by removing pixels invalid in OC-CCI
```
 
### Calculate statistics over histograms
```python
import histogram

ds = histogram.open_dataset()
hs = histogram.Stats()

hs.median_(ds.cci[0,10,:]) #Median value of Chl_sat for Jan in province 10
hs.var_(ds.cci[7,22,:])  #Variance of Chl_Rrs for Aug in province 22
```

### Earth Mover's Distances
```python
import earthmover

em = earthmover.Hist()
em_mod_arr = em.emd_hist(darwin="mod")
em_rrs_arr = em.emd_hist(darwin="rrs")
```

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Bror Jonsson - brorfred@gmail.com




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
