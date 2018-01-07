# -*- coding: utf-8 -*-

import pandas as pd
from geopy.geocoders import GeoNames, Nominatim, GoogleV3     # if explicit use of geopy
import numpy as np
import pycountry
from NER_helper import save_to_excel, read_from_excel

# %% Function definitions
def read_and_cleanup_NER_tags(filename):
    #return pd.read_fwf(filename, header=None, encoding='utf-8', names=['year', 'position', 'offset', 'category', 'subcategory', 'entity'])
    df = pd.read_csv(filename, header=None, encoding='utf-8',sep='\t',
                       names=['year', 'location', 'classifier', 'entity'])

    df_str_columns = df.select_dtypes(['object'])
    df[df_str_columns.columns] = df_str_columns.apply(lambda x: x.str.strip())

    df['category'], df['subcategory'] = df['classifier'].str.split(' ', 1).str
    df['position'], df['offset'] = df['location'].str.split(' ', 1).str

    df['position'] = df['position'].astype(int)
    df['offset'] = df['offset'].astype(int)

    #df['entity'] = df['entity'].map(lambda x: x.strip())

    df = df.drop(['location', 'classifier'],axis=1)
    return df


def setup_unique_locations_dataframe(df_tags, geocoded_filename):

    df_locations = df_tags.loc[df_tags.category.str.contains('LOC'),['year', 'entity']]
    df = df_locations['entity'].drop_duplicates().to_frame()
    df['processed'] = np.nan
    df['latitude'] = np.nan
    df['longitude'] = np.nan
    df['reversename'] = np.nan
    df['country'] = np.nan
    df = df.set_index('entity')

    df_geocoded = read_from_excel(geocoded_filename).set_index('entity')
    return df.combine_first(df_geocoded)

def get_country(x):
    aliases = {
        'UK': 'GB',
        'Russia': 'RU',
        'Vatican City': 'VA',
        'Iran': 'IR',
        'South Korea': 'KR',
        'Syria': 'SY',
        'St Barthélemy': 'BL',
        'Democratic Republic of the Congo': 'CO',
        'The Bahamas': 'BS',
        'São Tomé and Príncipe': 'ST',
        'Myanmar (Burma)' : 'MM'
    }
    if x is None or (x != x) or x == "nan": return None
    try:
        parts = list(reversed(x.split(', ')))
        for z in parts:
            try:
                if 'singapore' in z.lower(): z = 'SG'
                if 'hungary' in z.lower(): z = 'HU'
                return pycountry.countries.lookup(aliases.get(z,z)) ##.name
            except:
                pass
        print("unknown: " + str(x))
        return None
    except:
        print("failed: " + str(x))

# %% Main geocoding function
def assign_geocodes(geolocator, df_locations, count=25):

    i = 0
    for index, row in df_locations.iterrows():

        if row['processed'] > 0:
            continue

        df_locations.loc[index,'processed'] = 1.0
        location = geolocator.geocode(index) # dfunique.loc[index,'entity'])

        if not location is None:
            df_locations.loc[index,'latitude'] = location.latitude
            df_locations.loc[index,'longitude'] = location.longitude
            point = [ location.latitude, location.longitude ]
            reverseName = geolocator.reverse(point, exactly_one=True)
            if not reverseName is None:
                print("{0} ==> {1}".format(index, reverseName[0]))
                df_locations.loc[index,'reversename'] = reverseName[0]
                # df_locations.loc[index,'country'] = get_country(reverseName[0])

        if i > count:
            break
        i += 1

    return i

def assign_country_to_locations(df):
    country_info = df['reversename'].map(lambda x: get_country(str(x)))
    df['country'] = country_info.map(lambda x: x.name if not x is None else None)
    df['country_code'] = country_info.map(lambda x: x.alpha_2 if not x is None else None)
    df['country_code3'] = country_info.map(lambda x: x.alpha_3 if not x is None else None)

# %% Read input data
def process_geocoding(df_tags, geolocator, geocoded_filename, geocoded_output_filename):
    df_locations = setup_unique_locations_dataframe(df_tags, geocoded_filename)
    pending_count = len(df_locations[(df_locations['processed'] != 1.0)])
    print("Pending count: {0}".format(pending_count))
    if pending_count > 0:
        while True:
            hits = assign_geocodes(geolocator, df_locations)
            if hits == 0: break
            save_to_excel(df_locations, geocoded_output_filename)
    print("Done!")
    return df_locations

def apply_geocodes(df_loc_tags, df_locations):
    ''' Merge location tags (LOC) with geocoded data and return geocoded result set '''
    df_loc_ok = df_locations[df_locations.status == 'OK']
    return pd.merge(df_loc_tags, df_loc_ok, how='left', left_on='entity', right_index=True)

def main(do_geocoding=False, do_tags_cleanup=False, do_assign_country=False, do_merge_tags_with_geocodes=False):

    ''' Setup sources '''
    tags_filename = './data/Daedalus_1931-2014_tags.tsv'
    clean_location_tags_filename = './data/Daedalus_1931-2014_cleaned_up_location_tags.xlsx'
    geocoded_filename = './data/daedalus_ner_geocoded.xlsx'
    new_geocoded_filename = './data/daedalus_ner_geocoded_NEW.xlsx'
    geocoded_location_tags_filename = './data/Daedalus_1931-2014_geocoded_location_tags.xlsx'

    geolocator = GoogleV3(api_key='AIzaSyAUPl7HOuaq1rF_PmMykx1G0JMjeNJZzBQ', timeout=5)
    #geolocator = GeoNames(country_bias='Sweden', username='humlab')
    #geolocator = Nominatim() # OpenStreetMaps

    if do_tags_cleanup:
        df_tags = read_and_cleanup_NER_tags(tags_filename)
        df_loc_tags = df_tags[df_tags.category.str.contains('LOC')]
        save_to_excel(df_loc_tags, clean_location_tags_filename)
    else:
        df_loc_tags = read_from_excel(clean_location_tags_filename, 'Sheet1')

    if do_geocoding:
        df_locations = process_geocoding(df_loc_tags, geolocator, geocoded_filename, new_geocoded_filename)
    else:
        df_locations = read_from_excel(geocoded_filename, 'Sheet1')\
            .set_index('entity')

    if do_assign_country:
        assign_country_to_locations(df_locations)
        save_to_excel(df_locations, new_geocoded_filename)

    if do_merge_tags_with_geocodes:
        df_geocoded_result = apply_geocodes(df_loc_tags, df_locations)
        save_to_excel(df_geocoded_result, geocoded_location_tags_filename)
