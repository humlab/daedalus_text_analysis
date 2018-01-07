# -*- coding: utf-8 -*-
# %% Imports
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy.io.shapereader as shapereader
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from NER_helper import read_from_excel
from owslib.wms import WebMapService
# %%
def plot_sweden(projection=ccrs.Mercator(), extent=[5, 24, 54, 70], figsize=(6.2, 13), plot_grid=False,  cmap=None):

    #subplot_kw = dict(projection=projection)
    #fig, axes = plt.subplots(x,y, subplot_kw=subplot_kw)
    #fig.subplots_adjust(hspace=0.3)

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)

    if not extent is None:
        ax.set_extent(extent)

    if plot_grid:
        gl = ax.gridlines(draw_labels=True)
        gl.xlabels_top = gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    #ax.stock_img()
    #ax.add_feature(cf.LAND)
    #ax.add_feature(cf.OCEAN)
    #ax.add_feature(cf.COASTLINE)
    ax.add_feature(cf.BORDERS, linestyle=':')
    #ax.add_feature(cf.LAKES, alpha=0.5)
    #ax.add_feature(cf.RIVERS)
    #ax.coastlines()
    if not cmap is None:
        add_colorbar(ax, cmap)

    return fig, ax

def add_layer(ax, url, layer):
    wms = WebMapService(url)
    ax.add_wms(wms, layer) #, crs='EPSG:4326') #crs='CRS:84')

def add_colorbar(ax, cmap=plt.cm.YlGn):
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(0,1))
    sm._A = []
    plt.colorbar(sm,ax=ax,shrink=.62)

# %% Plot Sweden
geocoded_places_filename = './data/Daedalus_1931-2014_geocoded_location_tags.xlsx'
word_count_filename = "./data/Daedalus_Clean_Text_1931-2014_word_count.xlsx"

def read_stored_swedish_places(filename=geocoded_places_filename):
    df = read_from_excel(filename)
    df = df[(~pd.isnull(df).any(axis=1)) & (df.status == 'OK') & (df.country_code3 == 'SWE')]
    df['decade'] = 10 * (df['year'] / 10).astype(int)
    return df

def group_by_lon_lat(df_places, close_threshold = 25.0):
    df_places['lat_10'] =  round(df_places['latitude'] * close_threshold).astype(int)
    df_places['lon_10'] =  round(df_places['longitude'] * close_threshold).astype(int)
    df = df_places.groupby(['decade', 'lat_10', 'lon_10' ]).size().reset_index()
    df['latitude'] = df['lat_10'] / close_threshold
    df['longitude'] = df['lon_10'] / close_threshold
    df = df.drop(['lat_10', 'lon_10'], axis=1)
    df.columns = ['decade', 'number_of_places', 'latitude', 'longitude']
    return df

# %%
def plot_decade_places(df_decade, deacde, url, layer, dpi=600):
    fig, ax = plot_sweden()
    add_layer(ax, url=url, layer=layer)
    weights = df_decade['number_of_places'] if 'number_of_places' in df_decade else None
    ax.scatter(df_decade.longitude, df_decade.latitude, zorder=99, s=weights, c="red", edgecolor="face", alpha=None, transform=ccrs.Geodetic())
    ax.set_title("Places in {0}s".format(decade))
    plt.savefig('.\data\sweden_{0}.png'.format(decade), dpi=dpi)

# %%
def plot_all_decades():
    close_threshold = 25.0
    base_layer_url = 'http://geoserver.humlab.umu.se:8080/geoserver/sead/wms'
    base_layer_name = 'NE2_HR_LC_SR_W_DR'
    df_places = read_stored_swedish_places(geocoded_places_filename)
    df_grouped_places = group_by_lon_lat(df_places, close_threshold)

    for decade in range(1930,2011,10):
        df_decade = df_grouped_places.loc[(df_grouped_places.decade == decade)]
        plot_decade_places(df_decade, decade, url=base_layer_url, layer=base_layer_name)

# %%

#def test_plot_municipalities(projection=ccrs.Mercator(), figsize=(6.2, 13)):
extent=[0, 24, 36, 70]
plt.figure(figsize=(10, 10))

ax = plt.axes(projection=ccrs.Mercator())
ax.set_extent(extent)

ax.stock_img()
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
shpfilename  = '.\data\kommuner_scb\kommungränser_scb_07.shp'
#pc = ccrs.mercator()
#f = lambda x: pc.project_geometry(x, pc)
geometries = list(shapereader.Reader(shpfilename).geometries())
#shape_iterator = shapereader.Reader(shpfilename)
for geometry in geometries:
    shape_feature = cf.ShapelyFeature(geometry, ccrs.Robinson(), facecolor='blue', edgecolor='black')
    ax.add_feature(shape_feature)
    #ax.add_geometries([geometry], crs=ccrs.Mercator(), facecolor='red', edgecolor='black')
#plt.tight_layout()
plt.show()
# %%
#test_plot_municipalities()

#import numpy as np
#import matplotlib.pyplot as plt
#from scipy.stats import gaussian_kde as kde
#from matplotlib.colors import Normalize
#from matplotlib import cm
#
#df_places = read_stored_swedish_places(geocoded_places_filename)
#
#samples = [ df_places.longitude, df_places.latitude ]
## compute kernel density estimate
#densObj = kde(samples)
#
#def makeColours(vals):
#    norm = Normalize(vmin=vals.min(), vmax=vals.max())
#    #Can put any colormap you like here.
#    colours = [ cm.ScalarMappable(norm=norm, cmap=plt.cm.YlGn).to_rgba(val) for val in vals ]
#    return colours
#
#colours = makeColours( densObj.evaluate( samples ) )
#
#plt.scatter( samples[0], samples[1], color=colours )
#plt.show()