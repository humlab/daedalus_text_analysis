# -*- coding: utf-8 -*-
"""
This is a temporary script file.
"""
import pandas as pd
import geopandas                                    # HOWTO install: http://geoffboeing.com/2014/09/using-geopandas-windows/
from geopandas.tools import geocode                 # uses geopy
import geocoder as geocoder                         # alternative to geopy: pip install geocoder
from geopy.geocoders import GeoNames, Nominatim, GoogleV3     # if explicit use of geopy
import numpy as np

df = pd.read_fwf('Daedalus1931-99.tags.tsv', header=None, encoding='utf-8', names=['year', 'position', 'offset', 'category', 'subcategory', 'entity'])

dfloc = df.loc[df.category=='LOC',['year', 'entity']]
dfunique = dfloc['entity'].drop_duplicates().to_frame()
dfunique['processed'] = None
dfunique['latitude'] = np.nan
dfunique['longitude'] = np.nan
dfunique['reversename'] = np.nan
#dfunique = dfunique.drop('geocode',axis=1)
#df_year_count = df.groupby(['year', 'entity']).size().reset_index(name='counts')

i = 0
geolocator = GoogleV3(api_key='AIzaSyAUPl7HOuaq1rF_PmMykx1G0JMjeNJZzBQ', timeout=5)
#geolocator = GeoNames(country_bias='Sweden', username='humlab')
#geolocator = Nominatim() # OpenStreetMaps
#dfunique['processed'] = None

for index, row in dfunique.iterrows():

    if not row['processed'] is None:
        continue

    dfunique.loc[index,'processed'] = True
    location = geolocator.geocode(index) # dfunique.loc[index,'entity'])

    if not location is None:
        dfunique.loc[index,'latitude'] = location.latitude
        dfunique.loc[index,'longitude'] = location.longitude

        point = [ location.latitude, location.longitude ]

        reverseName = geolocator.reverse(point, exactly_one=True)

        if not reverseName is None:
            # print("{0} ==> {1}".format(row['entity'], reverseName[0]))
            print("{0} ==> {1}".format(index, reverseName[0]))
            dfunique.loc[index,'reversename'] = reverseName[0]

    if i > 50:
        break
    i += 1

def save_as_excel(filename, sheetname):
    writer = pd.ExcelWriter('C:\TEMP\daedalus_ner_geocoded_NEW.xlsx')
    dfunique.to_excel(writer,'Sheet1')
    writer.save()

dfu = pd.read_excel(open('Daedalus_NER_geocoded.xlsx','rb'), sheetname='Sheet1', index='entity')
dfu = dfu.set_index('entity')

map(lambda x: x == 1.0, dfu['processed'])

dfu.loc[(dfu['processed']!=1.0)]
