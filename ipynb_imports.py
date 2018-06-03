from bokeh.core.properties import value, expr
from bokeh.io import output_file, push_notebook
from bokeh.io import push_notebook, show, output_notebook
from bokeh.io import show, output_file
from bokeh.layouts import layout
from bokeh.layouts import row
from bokeh.layouts import row, column, widgetbox
from bokeh.layouts import widgetbox
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.models.glyphs import VBar
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models import ColumnDataSource, CustomJS, Rect
from bokeh.models import ColumnDataSource, HoverTool, BoxSelectTool, LabelSet, Label, Arrow, OpenHead
from bokeh.models import ColumnDataSource, Paragraph, HoverTool, Div
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models.widgets import Slider
from bokeh.plotting import figure
from bokeh.plotting import figure, output_notebook, show
from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.plotting import output_notebook, figure, show
from bokeh.plotting import show, output_notebook, output_file
from bokeh.sampledata.glucose import data
from bokeh.sampledata.iris import flowers
from bokeh.transform import transform, jitter
from cartopy.io.img_tiles import OSM
#from compute_topic_model import MyLdaMallet
from functools import reduce
from __future__ import print_function
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.wrappers import ldamallet
from geopandas.tools import geocode                 # uses geopy
from geopy.geocoders import GeoNames, Nominatim, GoogleV3     # if explicit use of geopy
from IPython.core.display import display, HTML, clear_output
# from IPython.core.interactiveshell import InteractiveShell
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display, clear_output
from IPython.display import display, HTML, clear_output, IFrame
# from IPython.display import IFrame, display
from IPython.display import IFrame, display
from IPython.lib.display import YouTubeVideo
from ipywidgets import interact
from ipywidgets import interact, interactive, fixed, interact_manual
from itertools import product
from math import gamma
from matplotlib import pyplot as plt
from matplotlib.path import Path
#from matplotlib_venn import venn2, venn3, venn3_circles
from matplotlib_venn import venn3, venn3_circles
from matplotlib_venn_wordcloud import venn2_wordcloud
from nltk import word_tokenize
from numpy import exp
from operator import mul
from pivottablejs import pivot_ui
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.special import factorial
from scipy.special import gamma
from scipy.stats import t, zscore
from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
from sklearn.manifold import TSNE
import bokeh.io as bio
import bokeh.models as bm
import bokeh.palettes
import bokeh.plotting as bp
import cartopy.crs as ccrs
import gensim
import gensim.models
import geocoder as geocoder                         # alternative to geopy: pip install geocoder
import geopandas                                    # HOWTO install: http://geoffboeing.com/2014/09/using-geopandas-windows/
import glob
import gmaps
import gmaps.datasets
import gmaps.geojson_geometries
import IPython.display # import display, HTML
import ipywidgets as widgets
import logging
import math
import matplotlib.pyplot as plt
import matplotlib.tri as tri
#import mpld3
import numpy as np
import os
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
import pyLDAvis.gensim as gensimvis
import scipy
import types
import warnings
import wordcloud
