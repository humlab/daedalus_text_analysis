# -*- coding: utf-8 -*-
from matplotlib import pyplot
import pandas as pd
from scipy import spatial
import logging
logger = logging.getLogger(__name__)

class SimilarWordGenerater():

    def __init__(self, word_vectors):
        self.word_vectors = word_vectors

    def flatten(self, items):
        return [ item for sublist in items for item in sublist ]

    def get_similar(self, words, iter=1, topn=3, similar_threshold=0.0):
        return self.flatten([ [ y[0] for y in self.word_vectors.most_similar(x, topn=topn) if y[1] >= similar_threshold ] for x in words ])

    #def generate0(self, words, iter=1, topn=3, similar_threshold = 0.0):
    #    if isinstance(words,str): words = [ words ]
    #    if iter < 0 or len(words) == 0: return []
    #    similar_words = list(set(self.get_similar(words, iter, topn, similar_threshold)) - set(words))
    #    return list(set(words + self.generate0(similar_words, iter-1, topn)))

    def generate(self, words, iter=1, topn=3, cumulated_words=[], similar_threshold=0.5):
        # logger.info('{}: {}'.format(iter, cumulated_words))
        if isinstance(words, str): words = [ words ]
        if iter < 0 or len(words) == 0: return cumulated_words
        cumulated_words = list(set(cumulated_words + words))
        similar_words = list(set(self.get_similar(words, iter, topn, similar_threshold)) - set(cumulated_words))
        return self.generate(similar_words, iter-1, topn, cumulated_words)

# %%
def word_scale_similarity(word_vectors, scale_x_pair, scale_y_pair, word_list):

    scale_x = word_vectors[scale_x_pair[0]] - word_vectors[scale_x_pair[1]]
    scale_y = word_vectors[scale_y_pair[0]] - word_vectors[scale_y_pair[1]]

    word_x_similarity = [1 - spatial.distance.cosine(scale_x, word_vectors[x]) for x in word_list ]
    word_y_similarity = [1 - spatial.distance.cosine(scale_y, word_vectors[x]) for x in word_list ]

    df = pd.DataFrame({ 'word': word_list, 'x': word_x_similarity, 'y': word_y_similarity })

    return df

# %%

def word_pair_list_similarity(word_vectors, word_x, word_y, word_list):

    word_x_similarity = [ word_vectors.similarity(x, word_x) for x in word_list ]
    word_y_similarity = [ word_vectors.similarity(x, word_y) for x in word_list ]

    df = pd.DataFrame({ 'word': word_list, 'x': word_x_similarity, 'y': word_y_similarity })

    return df

# %%

def word_pair_toplist_similarity(word_vectors, word_x, word_y, topn=50):

    word_x_toplist = [ word_x ] + word_vectors.most_similar_cosmul(word_x, topn=topn)
    word_y_toplist = [ word_y ] + word_vectors.most_similar_cosmul(word_y, topn=topn)

    word_toplist = [ x[0] for x in word_x_toplist + word_y_toplist ]

    return word_pair_list_similarity(word_vectors, word_x, word_y, word_toplist)

## %%
#
#def word_pair_list_similarity(word_vectors, word_x, word_y, word_list, topn=50):
#
#    word_x_toplist = [ word_x ] + word_vectors.most_similar_cosmul(word_x, topn=topn)
#    word_y_toplist = [ word_y ] + word_vectors.most_similar_cosmul(word_y, topn=topn)
#
#    word_toplist = [ x[0] for x in wordlist ]
#
#    word_x_similarity = [ word_vectors.similarity(x, word_x) for x in word_toplist ]
#    word_y_similarity = [ word_vectors.similarity(x, word_y) for x in word_toplist ]
#
#    df = pd.DataFrame({ 'word': word_toplist, 'x': word_x_similarity, 'y': word_y_similarity })
#
#    return df
# %%

def plot_df(df,xlabel=None,ylabel=None):
    fig = pyplot.figure()
    #pyplot.plot([0,0.75], [0,0.75])
    if not xlabel is None: pyplot.xlabel(xlabel)
    if not ylabel is None: pyplot.ylabel(ylabel)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df['x'], df['y'],marker='o')
    for i, txt in enumerate(df['word']):
        ax.annotate(txt, xy=(df['x'].iloc[i], df['y'].iloc[i])) #, textcoords = 'offset points', ha = 'left', va = 'top', **TEXT_KW)
    pyplot.show()

# %%

def generate_plot(words, x_word, y_word, iter=5, topn=3, similar_threshold= 0.5):
    word_list = SimilarWordGenerater(word_vectors).generate(words, iter, topn, [], similar_threshold)
    df = word_pair_list_similarity(word_vectors, x_word, y_word, word_list)
    plot_df(df,'<--- {} --->'.format(x_word), '<--- {} --->'.format(y_word))


word_list = SimilarWordGenerater(word_vectors).generate(['jordbruk'], 5, topn=3)
df = word_pair_list_similarity(word_vectors, 'person', 'maskin', word_list)
plot_df(df, '<--- Person --->', '<--- Maskin --->')

# %%

df = word_pair_toplist_similarity(word_vectors, 'konsument', 'producent')
plot_df(df, 'Konsument', 'Producent')

# %%

df = word_pair_toplist_similarity(word_vectors, 'industri', 'jordbruk')
plot_df(df)
# %%

df = word_pair_toplist_similarity(word_vectors, 'krig', 'fred')
plot_df(df)
# %%

df = word_pair_toplist_similarity(word_vectors, 'industri', 'hantverk')
plot_df(df)


# %%
df_swe_loc = pd.read_excel('../data/ner_swe_loc_plc.xlsx', 'swe_loc_plc')
df_place_minus_one = pd.DataFrame(df_swe_loc.place.apply(lambda x: x[:-1]))
df_place_minus_one.columns = ['place']
swe_loc_list = list(pd.merge(df_swe_loc,df_place_minus_one,how='inner',left_on='place',right_on='place')['place'].unique())
swe_loc_list = [ x for x in swe_loc_list if x in word_vectors.vocab ]
df = word_pair_list_similarity(word_vectors, 'industri', 'jordbruk',swe_loc_list)
plot_df(df,'<--- Industri --->', '<--- Jordbruk --->')


# %%

df = word_pair_toplist_similarity(word_vectors, 'flicka', 'pojke')
plot_df(df, 'Flicka', 'Pojke')

# %%
word_list = SimilarWordGenerater(word_vectors).generate(['kärnkraft'], 5, topn=3)
df_scale = word_scale_similarity(word_vectors, ('stad', 'landsbygd'), ('jordbruk', 'industri'), word_list)
plot_df(df_scale, 'Stad ---- Landsbygd', 'jordbruk ---- industri')

# %%
word_list = generate_similar_list(word_vectors, ['kärnkraft'], 5, topn=2)
df = word_pair_list_similarity(word_vectors, 'sovjetunionen', 'england',word_list)
plot_df(df, 'Sovjet', 'England')


