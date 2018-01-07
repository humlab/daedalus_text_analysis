# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from matplotlib import pyplot
import pandas as pd
from utility import load_model
# %%


# %%
def compute_tsne_coordinates(model):
    model = load_model(filename)
    vocab = list(model.wv.vocab)
    X = model[vocab]
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    return X_tsne
# %%

def plot_df(df):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df['x'], df['y'])
    for i, txt in enumerate(df['word']):
        ax.annotate(txt, (df['x'].iloc[i], df['y'].iloc[i]))
    pyplot.show()

# %%

filename = '../data/output/w2v_model_skip_gram_win_5_dim_50_iter_20_mc_5_complete_not_segmented.dat'

model = load_model(filename)

X_tsne = compute_tsne_coordinates(model)

# df = pd.concat([pd.DataFrame(X_tsne), pd.Series(vocab)], axis=1)

# plot_df(df)
