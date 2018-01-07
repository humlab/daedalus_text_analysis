# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

# %% Imports
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy.io.shapereader as shpr
import matplotlib.pyplot as plt
import matplotlib as mpl
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from NER_helper import read_from_excel
import numpy as np

# %% Setup Data
geocoded_location_tags_filename = './data/Daedalus_1931-2014_geocoded_location_tags.xlsx'
word_count_filename = "./data/Daedalus_Clean_Text_1931-2014_word_count.xlsx"

def get_stored_classified_locations(filename=geocoded_location_tags_filename):
    df = read_from_excel(filename)
    #df = df[(~pd.isnull(df).any(axis=1)) and (df.status == 'OK')]
    df = df[(~pd.isnull(df).any(axis=1)) & (df.status == 'OK')]
    return df

def get_word_count_per_decade(filename=word_count_filename):
    df = read_from_excel(filename) #.reset_index()
    df['decade'] = 10 * (df['year'] / 10).astype(int)
    df = df.loc[~df['year'].isin([1979,1990])].groupby(['decade']).word_count.sum() #.reset_index()
    df.columns = ['word_count']
    return df

def get_decade_by_country(filename=geocoded_location_tags_filename, ignore_cc3=['SWE']):
    locations = get_stored_classified_locations(filename)
    #locations['decade'] = locations['year'].apply(lambda x: 10 * int(x / 10))
    locations['decade'] = 10 * (locations['year'] / 10).astype(int)
    df = locations[~locations['country_code3'].isin(ignore_cc3)].groupby(['decade', 'country_code3' ]).size().reset_index()
    # Add decade word count:
    df_wc = get_word_count_per_decade().to_frame()
    df = pd.merge(df, df_wc, left_on='decade', right_index=True)
    # Name columns
    df.columns = ['decade', 'country_code3', 'number_of_occurences', 'word_count']
    # Compute number of occurences per kWord
    df['occ_per_kWords'] = 1000.0 * (df['number_of_occurences']  / df['word_count'] )
    # Set index
    df = df.set_index(['decade', 'country_code3'])
    # Normalize values so that highest value is 1.0
    df['n_number_of_occurences'] = df['number_of_occurences'] / df['number_of_occurences'].max()
    df['n_occ_per_kWords'] = df['occ_per_kWords'] / df['occ_per_kWords'].max()
    return df

# %% Helper functions

def get_country_shapes(shapename='admin_0_countries'):
    countries_shp = shpr.natural_earth(resolution='50m', category='cultural', name=shapename)
    return shpr.Reader(countries_shp).records()

def create_axes(projection=ccrs.Mercator(), extent=[-180, 180, -55, 75]):
    '''
    '''
    #ax = plt.axes(projection=ccrs.PlateCarree())
    ax = plt.axes(projection=projection)
    ax.set_extent(extent)
    ax.stock_img()
    #ax.add_feature(LAND)
    ax.coastlines()
    #ax.add_feature(COASTLINE)
    return ax

def setup_axes_labels(ax):
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

# %% Helper functions

def setup_countries(ax, df_decade, cmap, shapename='admin_0_countries'):
    country_codes = []
    for country in get_country_shapes():
        country_code = country.attributes['iso_a3']
        if not country_code in df_decade.index:
            #ax.add_geometries(country.geometry, ccrs.PlateCarree(),edgecolor='#000000', facecolor='#ffffff')
            continue
        country_codes.append(country_code)
        value = df_decade.loc[country_code].n_occ_per_kWords
        #scaled_value = 0.75/2.0+float(value)/4.0
        ax.add_geometries(country.geometry, ccrs.PlateCarree(),edgecolor='#000000', facecolor=cmap(value, 1))
    not_found_countries = [ x for x in df_decade.index if x not in country_codes ]
    if len(not_found_countries) > 0:
        # GUF=French Guiana
        # SJM=Svalbard and Jan Mayen MTQ=Martinique
        print("Warning! Not found and IGNORED: " + ' '.join(not_found_countries))

# %% Plot World map

df_data = get_decade_by_country()

# %% Plot World map

def plot_decade(df_decade, decade, borders=False, feature = None, dpi=300, figsize=(13,6.2)):
    plt.figure(figsize=figsize)
    ax = create_axes()
    #setup_axes_labels(ax)
    if not feature is None:
        ax.add_feature(
            cf.NaturalEarthFeature(feature['category'], feature['name'], '50m', edgecolor=feature['edgecolor'], facecolor=feature['facecolor'])
        )
    cmap = plt.cm.YlGn
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(0,1))
    sm._A = []
    plt.colorbar(sm, ax=ax, shrink=.62)
    setup_countries(ax, df_decade, cmap)
    #mm = ax.pcolormesh(lon,lat,weights,vmin=-2, vmax=30, transform=ccrs.PlateCarree()) #,cmap=cmo.balance )
    if borders == True:
        ax.add_feature(cf.BORDERS)
    plt.title(str(decade))
    plt.savefig('.\data\world_choropleth{0}{1}_{2}.png'.format(
            '_borders' if borders else '', '_physical_land' if not feature is None else '', decade), dpi=dpi)
    #plt.show()

for decade in range(1930,2011,10):
    feature = {
        'category': 'physical',
        'name': 'land',
        'edgecolor':'face',
        'facecolor':'#ffffff'  #cf.COLORS['land'])
    }
    plot_decade(df_data.loc[decade], decade, borders=True, feature=None)
    plot_decade(df_data.loc[decade], decade, borders=False, feature=None)
    plot_decade(df_data.loc[decade], decade, borders=False, feature=feature)
    plot_decade(df_data.loc[decade], decade, borders=True, feature=feature)

# %%

#n = 100
#r = 2 * np.random.rand(n)
#theta = 2 * np.pi * np.random.rand(n)
#area = 200 * r**2 * np.random.rand(n)
#colors = theta
#c = plt.scatter(theta, r, c=colors, s=area, cmap=plt.cm.Greens)
#plt.colorbar()