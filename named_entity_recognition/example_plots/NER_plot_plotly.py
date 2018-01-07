# -*- coding: utf-8 -*-

# %% Imports
from NER_helper import read_from_excel

import plotly
plotly.tools.set_credentials_file(username='humlab', api_key='IuJbSzEnhQIdgGV2fGHP')

# user: humlab password: Wephex4j
#

from plotly import __version__
from plotly.offline import import download_plotlyjs, init_notebook_mode, plot, iplot

# %% Main
geocoded_location_tags_filename = './data/Daedalus_1931-2014_geocoded_location_tags.xlsx'

def get_locations(filename=geocoded_location_tags_filename):
    df = read_from_excel(filename)
    df = df[~pd.isnull(df).any(axis=1)]
    return df
# %%

df_loc_tags = get_locations()

# %%


# %%
import plotly.plotly as py
import pandas as pd

df = df_loc_tags.groupby(['year', 'country_code']).agg(['count'])['entity']

data = [ dict(
        type = 'choropleth',
        locations = df['CODE'],
        z = df['GDP (BILLIONS)'],
        text = df['COUNTRY'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '$',
            title = 'GDP<br>Billions US$'),
      ) ]

layout = dict(
    title = '...',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )
