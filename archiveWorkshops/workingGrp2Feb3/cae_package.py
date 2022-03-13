#cal-adapt analytics engine project temporary workaround

import numpy as np
import xarray as xr
import pandas as pd
import xesmf as xe
import param
import panel as pn
import s3fs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cycler import cycler
from itertools import cycle
import geoviews as gv

import lmoments3 as lm
from lmoments3 import distr

import matplotlib.pyplot as plt
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely import geometry
import holoviews as hv
from holoviews import opts
import hvplot.pandas
import hvplot.xarray

#connect to AWS S3 storage
fs = s3fs.S3FileSystem(anon=True)

#constants
cached_stations = ['','BAKERSFIELD MEADOWS FIELD','BLYTHE ASOS','BURBANK-GLENDALE-PASADENA AIRPORT','LOS ANGELES DOWNTOWN USC CAMPUS','NEEDLES AIRPORT','FRESNO YOSEMITE INTERNATIONAL AIRPORT','IMPERIAL COUNTY AIRPORT','LAS VEGAS MCCARRAN INTERNATIONAL AP','LOS ANGELES INTERNATIONAL AIRPORT','LONG BEACH DAUGHERTY FIELD','MERCED MUNICIPAL AIRPORT','MODESTO CITY-COUNTY AIRPORT','SAN DIEGO MIRAMAR WSCMO','OAKLAND METRO INTERNATIONAL AIRPORT','OXNARD VENTURA COUNTY AIRPORT','PALM SPRINGS REGIONAL AIRPORT','RIVERSIDE MUNICIPAL AIRPORT','RED BLUFF MUNICIPAL AIRPORT','SACRAMENTO EXECUTIVE AIRPORT','SAN DIEGO LINDBERGH FIELD','SANTA BARBARA MUNICIPAL AIRPORT','SAN LUIS OBISPO AIRPORT','GILLESPIE FIELD','SAN FRANCISCO INTERNATIONAL AIRPORT','SAN JOSE INTERNATIONAL AIRPORT','SANTA ANA JOHN WAYNE AIRPORT','THERMAL / PALM SPRINGS','UKIAH MUNICIPAL AIRPORT','LANCASTER WILLIAM J FOX FIELD']

variable_choices = ['2-m temperature','2-m specific humidity','Surface pressure','10m u-component of the wind','10m v-component of the wind','Snow water equivalent','Skin temperature','non-convective precipitation (accumulated)','convective precipitation (accumulated)','total precipitation','accumulated snowfall equivalent','diffuse downwelled solar radiation','surface upwelled solar radiation (all sky)','surface upwelled solar radiation (clear sky)','surface downwelled solar radiation (all sky)','surface downwelled solar radiation (clear sky)','Surface upwelled longwave radiation (all sky)','Surface upwelled longwave radiation (clear sky)','Surface downwelled longwave radiation (all sky)','Surface downwelled longwave radiation (clear sky)','Surface runoff','Sub-surface runoff']

scenario_choices = ["Historical -- What's Already Happened", 'SSP 2-4.5 -- Middle of the Road','SSP 3-7.0 -- Business as Usual', 'SSP 5-8.5 -- Burn it All']

warming_level_choices = ['2˚','3˚','4˚'] #DEGREES

distributions = ['Exponential','Gamma','Generalised Extreme Value','Generalised Logistic','Generalised Normal','Generalised Pareto','Gumbel','Kappa','Wakeby','Weibull']

return_periods =['2 years','5 years','10 years','20 years','50 years','100 years','200 years','500 years','1000 years']

export_formats = ['NetCDF (.nc)','.csv']

#=== Select ===================================
class DataSelector(param.Parameterized): #these choices need to be known throughout this library -- one big class?
    #LocationSelector may end up its own class to have lots more forms of selection
    cached_station  = param.Selector(objects=cached_stations) 
    timescale = param.Selector(objects=['Hourly','Daily','Monthly'])
    variable = param.Selector(objects=variable_choices)
    def view(self):
        if self.cached_station != '':
            return self.cached_station
        else:
            return 'choose a pre-calculated location'
    
def select():
    obj = DataSelector() 
    warming_levels = pn.widgets.CheckBoxGroup(name='Warming Levels',options=warming_level_choices) 
    scenario = pn.widgets.CheckBoxGroup(name='Scenarios',options=scenario_choices)
    dyn_stat = pn.widgets.CheckBoxGroup(name='Dynamical/Statistical',options=['Dynamical','Statistical'])
    return pn.Column(pn.Row(obj.param,dyn_stat), pn.Row(warming_levels, scenario))

#=== Generate ===================================                                                                                             
def generate():
    ds = xr.open_dataset("ERA5-WRF_45km_monthly.nc")
    ds['T2'].data = ds['T2'].data - 273.15
    ds['T2'].data = ds['T2'].data * (9/5) + 32
    ds['T2'].attrs['units'] = 'F'
    da = ds['T2']
    return da

# === Transform ===============================   

class ThresholdSelector(param.Parameterized):
    distributions = param.Selector(objects=distributions)

def setReturnPeriod():
    obj = ThresholdSelector()
    new_temporal_resolution = pn.widgets.RadioButtonGroup(name='New Temporal Resolution',options=['Annual'])
    return_periods = pn.widgets.IntSlider(name='Return Period',start=2,end=1000,value=10)
    date_range  = pn.widgets.IntRangeSlider(name='Reference Period Years',start=1950,end=2021,value=(1950,2021),step=1)
    return pn.Column(obj.param, new_temporal_resolution, pn.Row(return_periods), pn.Row(date_range))

def transform(da):
    da1 = da.sel(time=slice("1950-01-01", "1985-01-01"))
    da2 = da.sel(time=slice("1985-01-01", "2020-01-01"))
    return da1, da2

def setThreshold():
    obj = ThresholdSelector()
    new_temporal_resolution = pn.widgets.RadioButtonGroup(name='New Temporal Resolution',options=['Annual'])
    threshold = pn.widgets.IntSlider(name='Threshold (Degrees F)',start=0,end=120,value=80)
    date_range  = pn.widgets.IntRangeSlider(name='Reference Period Years',start=1950,end=2021,value=(1950,2021),step=1)
    return pn.Column(obj.param, new_temporal_resolution, threshold, pn.Row(date_range))  

def getReturnValue(y,return_year=10):
    ams = y.groupby('time.year').max('time')
    paras = distr.gev.lmom_fit(ams)
    fitted_gev = distr.gev(**paras)
    return_period = 1.0-(1./return_year)
    return_value = fitted_gev.ppf(return_period)
    return_value = round(return_value, 5)
    return xr.DataArray(return_value)

def calculateReturnValue(da):
    da_stacked = da.stack(allpoints=['x','y']).squeeze()
    da_stacked = da_stacked.groupby('allpoints')
    return_values = da_stacked.apply(getReturnValue)
    return_values = return_values.unstack('allpoints').transpose()
    return return_values

def getExceedance(y, exceedance=80):
    ams = y.groupby('time.year').max('time')
    paras = distr.gev.lmom_fit(ams)
    fitted_gev = distr.gev(**paras)
    exceedance_probability = 1-(fitted_gev.cdf(exceedance))
    return xr.DataArray(exceedance_probability)

def calculateExceedance(da):
    da_stacked = da.stack(allpoints=['x','y']).squeeze()
    da_stacked = da_stacked.groupby('allpoints')
    return_values = da_stacked.apply(getExceedance)
    return_values = return_values.unstack('allpoints').transpose()
    return return_values

#=== Visualize =============================
def getReturnValuePlot(x, y):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=[20,6], subplot_kw={"projection": ccrs.Orthographic(-112,42)})
    fig.suptitle('Comparison Return Values of a 1 in 10 Year Temperature Event (F) Between 1950-1985 and 1985-2020 ', fontsize='x-large')


    cf1 = ax1.pcolormesh(x.lon, x.lat, 
                                 x,
                                 transform=ccrs.PlateCarree(),
                                 cmap=cm.OrRd, vmin=40, vmax=100)
    ax1.set_extent([-130,-100,28,50])
    ax1.coastlines() 
    ax1.gridlines()
    ax1.add_feature(cfeature.BORDERS)
    ax1.add_feature(cfeature.NaturalEarthFeature(category='cultural',
                                        name='admin_1_states_provinces_lines',
                                        scale='110m',facecolor='None'),
                   edgecolor='k')
    ax1.set_title('1950-1985',
                 fontsize='x-large')


    cf2 = ax2.pcolormesh(y.lon, y.lat, 
                                 y,
                                 transform=ccrs.PlateCarree(),
                                 cmap=cm.OrRd, vmin=40, vmax=100)
    ax2.set_extent([-130,-100,28,50])
    ax2.coastlines() 
    ax2.gridlines()
    ax2.add_feature(cfeature.BORDERS)
    ax2.add_feature(cfeature.NaturalEarthFeature(category='cultural',
                                        name='admin_1_states_provinces_lines',
                                        scale='110m',facecolor='None'),
                   edgecolor='k')
    ax2.set_title('1985-2020',
                 fontsize='x-large')


    fig.subplots_adjust(bottom=0.25, wspace=-0.5)
    cbar_ax = fig.add_axes([0.22, 0.2, 0.6, 0.02])
    cbar=fig.colorbar(cf2, cax=cbar_ax, orientation='horizontal')
    cbar.ax.set_xlabel('Return Value of a 1 in 10 Year Event (F)', fontsize='large')
    
    plt.savefig('returnValuePlot.png', bbox_inches='tight')

    plt.show()

def getExceedancePlot(y):
    fig = plt.figure(figsize=[10,5])
    ax = plt.subplot(111,projection=ccrs.Orthographic(-112,42))

    plotted_data = ax.pcolormesh(y.lon, y.lat,
                             y,
                             transform=ccrs.PlateCarree(),
                             cmap=cm.Reds)

    cbar = fig.colorbar(plotted_data)

    ax.set_extent([-130,-100,28,50])
    ax.coastlines() 
    ax.gridlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.NaturalEarthFeature(category='cultural',
                                    name='admin_1_states_provinces_lines',
                                    scale='110m',facecolor='None'),
               edgecolor='k')

    cbar.ax.set_ylabel('Exceedance Probability',
                   fontsize='large',
                   labelpad=25,
                   rotation=270)
    ax.set_title('Exceedance Probability of a 80 F Temperature Event in 1950-2021',
             fontsize='x-large')
    plt.savefig('exceedancePlot.png', bbox_inches='tight')
    plt.show()


# === Export =================================
class ExportSelector(param.Parameterized):
    format_choice  = param.Selector(objects=export_formats)
    
    def view(self):
        if self.format_choice:
            return self.format_choice
        else:
            return 'Please select a data output format.'
    
def export():
    obj = ExportSelector() 
    widget = pn.widgets.TextInput(name='Output File', value='filename')
    button = pn.widgets.Button(name='Save')
    return pn.Column(obj.param, widget, button)
