# -*- coding: utf-8 -*-

import pycountry
import pandas as pd

def load_swener_tags(filename):

    df = pd.read_csv(filename, header=None, encoding='utf-8',sep='\t', names=['year', 'location', 'classifier', 'entity'])

    df_str_columns = df.select_dtypes(['object'])
    df[df_str_columns.columns] = df_str_columns.apply(lambda x: x.str.strip())

    df['category'], df['subcategory'] = df['classifier'].str.split(' ', 1).str
    df['position'], df['offset'] = df['location'].str.split(' ', 1).str

    df['position'] = df['position'].astype(int)
    df['offset'] = df['offset'].astype(int)

    df = df.drop(['location', 'classifier'],axis=1)
    return df

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

def assign_geocodes(geolocator, df_locations, count=25):
    ''' Main geocoding function '''
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

