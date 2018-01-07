# -*- coding: utf-8 -*-

import pandas as pd
import gmaps
import gmaps.datasets

# %%
gmaps.configure(api_key="AIzaSyAUPl7HOuaq1rF_PmMykx1G0JMjeNJZzBQ")

# %% Setup Data
df = pd.read_excel('C:\TEMP\daedalus_ner_geocoded.xlsx')

entity_locations = df.groupby(['latitude','longitude' ]).size().reset_index()
entity_locations.columns = ['latitude', 'longitude', 'count']
locations = entity_locations[["latitude", "longitude"]]
weights = entity_locations["count"]

# %%
layer = gmaps.symbol_layer(locations, fill_color="green", stroke_color="green", scale=2)
fig = gmaps.figure()
fig.add_layer(layer)

fig

# %%
heatmap_fig = gmaps.figure()
heatmap_layer = gmaps.heatmap_layer(locations, weights=weights)
heatmap_fig.add_layer(heatmap_layer)
heatmap_fig

# %%
heatmap_layer.max_intensity = 100
heatmap_layer.point_radius = 10

# %%
import gmaps.geojson_geometries
countries_geojson = gmaps.geojson_geometries.load_geometry('countries')

fig = gmaps.figure()

gini_layer = gmaps.geojson_layer(countries_geojson)
fig.add_layer(gini_layer)
fig

# %%
import pandas as pd
import vincent
import random
vincent.core.initialize_notebook()

#Dicts of iterables
cat_1 = ['y1', 'y2', 'y3', 'y4']
index_1 = range(0, 21, 1)
multi_iter1 = {'index': index_1}
for cat in cat_1:
    multi_iter1[cat] = [random.randint(10, 100) for x in index_1]

bar = vincent.Bar(multi_iter1['y1'])
bar.axis_titles(x='Index', y='Value')
bar.display()
# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

ax = plt.axes(projection=ccrs.Mollweide())
ax.stock_img()
plt.show()

# %%

# %%

    import sys
    import os
    import subprocess
    import datetime
    import platform

    import pandas as pd
    import matplotlib.pyplot as plt

    import cartopy.crs as ccrs
    from cartopy.io.img_tiles import OSM
    import cartopy.feature as cfeature
    from cartopy.io import shapereader
    from cartopy.io.img_tiles import StamenTerrain
    from cartopy.io.img_tiles import GoogleTiles
    #from owslib.wmts import WebMapTileService

    from matplotlib.path import Path
    import matplotlib.patheffects as PathEffects
    import matplotlib.patches as mpatches

    import numpy as np

# %%
plt.figure(figsize=(13,6.2))
tiler = OSM() #GoogleTiles()
mercator = tiler.crs
ax = plt.axes(projection=mercator)

#ax.set_extent(( 153, 153.2, -26.6, -26.4))

zoom = 3
ax.add_image(tiler, zoom )

# even 1:10m are too coarse for .2 degree square
#ax.coastlines('10m')

home_lat, home_lon = 0,0
# Add a marker for home
#plt.plot(home_lon, home_lat, marker='o', color='red', markersize=5, alpha=0.7, transform=ccrs.Geodetic())

plt.show()

# %%

