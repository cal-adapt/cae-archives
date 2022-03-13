#!/usr/bin/env python
# coding: utf-8

import numpy as np
import xarray as xy
import pandas as pd
import xesmf as xe
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

import os

hv.output(backend='bokeh')
hv.extension('bokeh', 'matplotlib')

directory = '/data/public/gcm/CMIP6/native/mon/historical/tas/'
theFiles = os.listdir(directory)
models = []
for oneFile in theFiles:
    if '.nc' in oneFile:
        modelName = oneFile.split('_')[2]
    if modelName not in models:
        models.append(modelName)


# #### Make a table of all of the ensemble members for each scenario for each model on local server

scenarios = ['historical','scenarioMIP/ssp585','scenarioMIP/ssp370','scenarioMIP/ssp245','scenarioMIP/ssp126']
simsOnParm = pd.DataFrame(index=models,columns=scenarios)

for model in models:
    simsOnParm.append([model])
    for scenario in scenarios:
        theFiles = os.listdir('/data/public/gcm/CMIP6/native/mon/'+scenario+'/tas/')
        fileList = [one for one in theFiles if model+'_' in one]
        ensMembers = []
        for oneFile in fileList:
            ensMem = oneFile.split('_')[4]
            if ensMem not in ensMembers:
                ensMembers.append(ensMem)
        simsOnParm[scenario][model] = ensMembers
simsOnParm        


# ### Choose two well-performing models with different climate sensitivity and multiple ensemble members for multiple scenarios

model1 = 'CNRM-ESM2-1'; model2='UKESM1-0-LL'
figureModels = [model1,model2]


# #### Build pandas dataframe with all of the timeseries of GMT

def buildDFtimeSeries(variable,model,scenarios):
    scenario = 'historical'
    dataHistorical = pd.DataFrame()
    theFiles = os.listdir('/data/public/gcm/CMIP6/native/mon/'+scenario+'/'+variable+'/')
    fileList = [one for one in theFiles if model in one]
    for oneFile in fileList:
        ensMem = oneFile.split('_')[4]
        with xy.open_dataset('/data/public/gcm/CMIP6/native/mon/'+scenario+'/'+variable+'/'+oneFile) as temp:
            weightlat = np.sqrt(np.cos(np.deg2rad(temp.lat)))
            weightlat = weightlat/np.sum(weightlat)
            timeseries = (temp[variable]*weightlat).sum('lat').mean('lon')
            dataHistorical[ensMem] = timeseries.to_pandas()

    dataOneModel = pd.DataFrame()
    for scenario in scenarios:
        theFiles = os.listdir('/data/public/gcm/CMIP6/native/mon/scenarioMIP/'+scenario+'/'+variable+'/')
        fileList = [one for one in theFiles if model in one]
        for oneFile in fileList:
            ensMem = oneFile.split('_')[4]
            with xy.open_dataset('/data/public/gcm/CMIP6/native/mon/scenarioMIP/'+scenario+'/'+variable+'/'+oneFile) as temp:
                weightlat = np.sqrt(np.cos(np.deg2rad(temp.lat)))
                weightlat = weightlat/np.sum(weightlat)
                timeseries = (temp[variable]*weightlat).sum('lat').mean('lon')
                if ensMem in dataHistorical:
                    dataOneModel[(scenario,ensMem)] = dataHistorical[ensMem].append(timeseries.to_pandas())
    #establish multiindex
    dataOneModel = dataOneModel.T
    dataOneModel.index = pd.MultiIndex.from_tuples(dataOneModel.index, names = ['scenario','ensMem'])
    dataOneModel.index.set_names(['scenario','ensMem'],inplace=True)
    dataOneModel = dataOneModel.T
    
    return dataOneModel


def getGWL(smoothed,degrees):
    #assumes smoothed *as scenario mean* is global:
    GWL = smoothed.sub(degrees).abs().idxmin()
    #make sure it's not just choosing one of the final timestamps just because it's the highest warming
    #despite being nowhere close to (much less than) the target value:
    for scenario in smoothed:
        if smoothed[scenario].sub(degrees).abs().min() > 0.01:
            GWL[scenario] = np.NaN
    return GWL

def getGWLone(timeseries,degrees):
    #assumes smoothed *as scenario mean* is global:
    GWL = timeseries.sub(degrees).abs().idxmin()
    #make sure it's not just choosing one of the final timestamps just because it's the highest warming
    #despite being nowhere close to (much less than) the target value:
    for ensMem in timeseries:
        if timeseries[ensMem].sub(degrees).abs().min() > 0.01:
            GWL[ensMem] = np.NaN
    return GWL

def GWLguidelines(timeseries,warmingLevels):
    linewidths = [0.3,0.6,1.1]
    scenarioColors = ['b','r','orange','g'] #the color order they plot in...
    tempList = []
    for j, degrees in enumerate(warmingLevels):
        lineHoriz = hv.Curve((timeseries.index,np.zeros(len(timeseries.index))+degrees))
        lineHoriz = lineHoriz.opts(color='black',line_width=linewidths[j])
        tempList.append(lineHoriz)
        for i, scenario in enumerate(timeseries):
            level = getGWL(timeseries,degrees)
            temp = hv.Curve(([level[scenario] for k in np.arange(10)],np.linspace(0,degrees,10)))
            temp = temp.opts(line_width=linewidths[j],color=scenarioColors[i])
            tempList.append(temp)
    return tempList
                                                                                                        

def getGWLmaps(variable,model,scenarios,warming,domain,GMTdf):
    scenario = 'historical'
    mapsHist = xy.Dataset()
    theFiles = os.listdir('/data/public/gcm/CMIP6/native/mon/'+scenario+'/'+variable+'/')
    fileList = [one for one in theFiles if model in one]
    for oneFile in fileList:
        ensMem = oneFile.split('_')[4]
        with xy.open_dataset('/data/public/gcm/CMIP6/native/mon/'+scenario+'/'+variable+'/'+oneFile) as temp:
            mapsHist[ensMem] = temp[variable].sel(lat=slice(domain[2],domain[3]),lon=slice(domain[0],domain[1]),
                                                                          time=slice('1850','1980')).mean('time')
    mapsHist = mapsHist.drop('height').squeeze().to_array('realization')

    maps = xy.Dataset()
    for gwl in warming:
        #level = getGWL(gwl)
        oneGWL = xy.Dataset()
        for scenario in scenarios:
            level = getGWLone(GMTdf[scenario],gwl) #it can be a different year, by ensemble member
            theFiles = os.listdir('/data/public/gcm/CMIP6/native/mon/scenarioMIP/'+scenario+'/'+variable+'/')
            fileList = [one for one in theFiles if model in one]
            oneScenario = xy.Dataset()
            for oneFile in fileList:
                ensMem = oneFile.split('_')[4]
                with xy.open_dataset('/data/public/gcm/CMIP6/native/mon/scenarioMIP/'+scenario+'/'+variable+'/'+oneFile) as temp:
                    temp = temp[variable].sel(lat=slice(domain[2],domain[3]),lon=slice(domain[0],domain[1]))
                    # need to do the 10 year rolling mean as well: 
                    temp = temp.rolling(time=120,center=True).mean()
                    temp = temp.sel(time=slice(level[ensMem],level[ensMem]+pd.Timedelta("15 days"))) #to deal with the date to id the month being offset
                    if ensMem in mapsHist.realization:
                        oneScenario[ensMem] = temp.squeeze() - mapsHist.sel(realization=ensMem).squeeze()
            oneGWL[scenario] = oneScenario.to_array('ensembleMember').squeeze()
        maps[gwl] = oneGWL.to_array('scenario').squeeze()

    maps = maps.to_array('warmingLevel').drop('height').drop('time').drop('realization')
    return maps



