#climakit temporary workaround
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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely import geometry
import holoviews as hv
from holoviews import opts
import hvplot.pandas
import hvplot.xarray

# Connect to AWS S3 storage
fs = s3fs.S3FileSystem(anon=True)

#constants
cached_stations = ['','BAKERSFIELD MEADOWS FIELD','BLYTHE ASOS','BURBANK-GLENDALE-PASADENA AIRPORT','LOS ANGELES DOWNTOWN USC CAMPUS','NEEDLES AIRPORT','FRESNO YOSEMITE INTERNATIONAL AIRPORT','IMPERIAL COUNTY AIRPORT','LAS VEGAS MCCARRAN INTERNATIONAL AP','LOS ANGELES INTERNATIONAL AIRPORT','LONG BEACH DAUGHERTY FIELD','MERCED MUNICIPAL AIRPORT','MODESTO CITY-COUNTY AIRPORT','SAN DIEGO MIRAMAR WSCMO','OAKLAND METRO INTERNATIONAL AIRPORT','OXNARD VENTURA COUNTY AIRPORT','PALM SPRINGS REGIONAL AIRPORT','RIVERSIDE MUNICIPAL AIRPORT','RED BLUFF MUNICIPAL AIRPORT','SACRAMENTO EXECUTIVE AIRPORT','SAN DIEGO LINDBERGH FIELD','SANTA BARBARA MUNICIPAL AIRPORT','SAN LUIS OBISPO AIRPORT','GILLESPIE FIELD','SAN FRANCISCO INTERNATIONAL AIRPORT','SAN JOSE INTERNATIONAL AIRPORT','SANTA ANA JOHN WAYNE AIRPORT','THERMAL / PALM SPRINGS','UKIAH MUNICIPAL AIRPORT','LANCASTER WILLIAM J FOX FIELD']

variable_choices = ['2-m temperature','2-m specific humidity','Surface pressure','10m u-component of the wind','10m v-component of the wind','Snow water equivalent','Skin temperature','non-convective precipitation (accumulated)','convective precipitation (accumulated)','total precipitation','accumulated snowfall equivalent','diffuse downwelled solar radiation','surface upwelled solar radiation (all sky)','surface upwelled solar radiation (clear sky)','surface downwelled solar radiation (all sky)','surface downwelled solar radiation (clear sky)','Surface upwelled longwave radiation (all sky)','Surface upwelled longwave radiation (clear sky)','Surface downwelled longwave radiation (all sky)','Surface downwelled longwave radiation (clear sky)','Surface runoff','Sub-surface runoff']

scenario_choices = ['SSP 2-4.5 -- Middle of the Road','SSP 3-7.0 -- Business as Usual', 'SSP 5-8.5 -- Burn it All']

warming_level_choices = ['2˚','3˚','4˚'] #DEGREES

distributions = ['Exponential','Gamma','Generalised Extreme Value','Generalised Logistic','Generalised Normal','Generalised Pareto','Gumbel','Kappa','Wakeby','Weibull']

return_periods =['2 years','5 years','10 years','20 years','50 years','100 years','200 years','500 years','1000 years']

export_formats = ['NetCDF (.nc)','.csv']

#=== Select ===================================
class DataSelector(param.Parameterized): #these choices need to be known throughout this library -- one big class?
    #LocationSelector may end up its own class to have lots more forms of selection
    cached_station  = param.Selector(objects=cached_stations) 
    timescale = param.Selector(objects=['hourly','daily'])
    variable = param.Selector(objects=variable_choices)
    def view(self):
        if self.cached_station != '':
            return self.cached_station
        else:
            return 'choose a pre-calculated location'

class ThresholdSelector(param.Parameterized):
    cached_station  = param.Selector(objects=cached_stations) 
    timescale = param.Selector(objects=['hourly','daily'])
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
    ds = xr.open_zarr('s3://cdcat/wrf/ucla/era5/historical/1hr/all/d02/')
    return ds

#=== Visualize =============================
def getGWL(smoothed,degrees):
    #assumes smoothed *as scenario mean* is global:
    GWL = smoothed.sub(degrees).abs().idxmin()
    #make sure it's not just choosing one of the final timestamps just because it's the highest warming
    #despite being nowhere close to (much less than) the target value:
    for scenario in smoothed:
        if smoothed[scenario].sub(degrees).abs().min() > 0.01:
            GWL[scenario] = np.NaN
    return GWL

def explore(dataOneModel,do_smoothing=False):
    scenarioMeans = dataOneModel.T.groupby(level='scenario').mean().T
    anom = scenarioMeans - scenarioMeans['1850':'1980'].mean()
    if do_smoothing==True:
        anom = anom.rolling(120,center=True).mean()['2000':] #defaults to window-size for min periods, and closed=right

    dfPlot = anom.hvplot(label='Temperature') #need to get the dates displaying along the x-axis
    
    warming = [2,3,4] #DEGREES
    linewidths = [0.3,0.6,1.1]
    scenarioColors = ['b','r','orange','g'] #the color order they plot in...
    tempList = [dfPlot]
    for j, degrees in enumerate(warming):
        lineHoriz = hv.Curve((anom.index,np.zeros(len(anom.index))+degrees))
        lineHoriz = lineHoriz.opts(color='black',line_width=linewidths[j])
        tempList.append(lineHoriz)
        for i, scenario in enumerate(anom):
            level = getGWL(anom,degrees)
            temp = hv.Curve(([level[scenario] for k in np.arange(10)],np.linspace(0,degrees,10)))
            temp = temp.opts(line_width=linewidths[j],color=scenarioColors[i])
            tempList.append(temp)
    linePlot = hv.Overlay(tempList) 
    return linePlot

# === Transform ===============================
class TransformSelector(param.Parameterized):
    type_of_transform = param.Selector(objects=['mean','sum','running mean','filter'])
    
    def view(self):
        if self.type_of_transform:
            return 'Historical Reference Period:'+str(self.date_range)
        else:
            return 'Please enter a date range'
    
def transform():
    obj = TransformSelector() 
    anomaly = pn.widgets.RadioButtonGroup(name='Calculate Anomaly',options=['Subtract Ref Period','Absolute Timeseries'])
    date_range  = pn.widgets.IntRangeSlider(name='Reference Period Years',start=1850,end=2015,value=(1980,2010),step=1)
    running_mean_window_months = pn.widgets.IntSlider(name='Months',start=12,end=360,value=120)
    new_temporal_resolution = pn.widgets.RadioButtonGroup(name='New Temporal Resolution',options=['seasonal','annual','decadal'])
    return pn.Column(obj.param, new_temporal_resolution, running_mean_window_months, pn.Row(anomaly,date_range))
        

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

# to do:
# -- map-based location selection
# -- dates in a range
