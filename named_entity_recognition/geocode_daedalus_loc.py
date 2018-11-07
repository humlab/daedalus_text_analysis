
from geopy.geocoders import GoogleV3# GeoNames, Nominatim, GoogleV3     # if explicit use of geopy
from . geocode_loc_tags import assign_geocodes, load_swener_tags, get_country
import numpy as np
import pandas as pd
import common.file_utility as file_utility
import common.utility as utility

logger = utility.getLogger(__name__)

def setup_unique_locations_dataframe(df_tags, geocoded_filename):

    df_locations = df_tags.loc[df_tags.category.str.contains('LOC'),['year', 'entity']]
    df = df_locations['entity'].drop_duplicates().to_frame()
    df['processed'] = np.nan
    df['latitude'] = np.nan
    df['longitude'] = np.nan
    df['reversename'] = np.nan
    df['country'] = np.nan
    df = df.set_index('entity')

    df_geocoded = file_utility.FileUtility.read_excel(filename=geocoded_filename, sheet='Sheet1').set_index('entity')
    return df.combine_first(df_geocoded)

def assign_country_to_locations(df):
    country_info = df['reversename'].map(lambda x: get_country(str(x)))
    df['country'] = country_info.map(lambda x: x.name if not x is None else None)
    df['country_code'] = country_info.map(lambda x: x.alpha_2 if not x is None else None)
    df['country_code3'] = country_info.map(lambda x: x.alpha_3 if not x is None else None)

def process_geocoding(df_tags, geolocator, geocoded_filename, geocoded_output_filename):
    df_locations = setup_unique_locations_dataframe(df_tags, geocoded_filename)
    pending_count = len(df_locations[(df_locations['processed'] != 1.0)])
    logger.info("Pending count: %s", pending_count)
    if pending_count > 0:
        while True:
            hits = assign_geocodes(geolocator, df_locations)
            if hits == 0: break
            file_utility.FileUtility.save_to_excel(df_locations, geocoded_output_filename)
    logger.info("Done!")
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
        df_tags = load_swener_tags(tags_filename)
        df_loc_tags = df_tags[df_tags.category.str.contains('LOC')]
        file_utility.FileUtility.save_excel((df_loc_tags, 'Sheet1'), clean_location_tags_filename)
    else:
        df_loc_tags = file_utility.FileUtility.read_excel(clean_location_tags_filename, 'Sheet1')

    if do_geocoding:
        df_locations = process_geocoding(df_loc_tags, geolocator, geocoded_filename, new_geocoded_filename)
    else:
        df_locations = file_utility.FileUtility.read_excel(geocoded_filename, 'Sheet1')\
            .set_index('entity')

    if do_assign_country:
        assign_country_to_locations(df_locations)
        file_utility.FileUtility.save_excel((df_locations,'Sheet1'),  new_geocoded_filename)

    if do_merge_tags_with_geocodes:
        df_geocoded_result = apply_geocodes(df_loc_tags, df_locations)
        file_utility.FileUtility.save_excel((df_geocoded_result,'Sheet1'),  geocoded_location_tags_filename)

main()
