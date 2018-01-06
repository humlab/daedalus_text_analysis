# -*- coding: utf-8 -*-
from sklearn.decomposition import PCA
from matplotlib import pyplot
from utility import load_model

# %%
# fit a 2d PCA model to the vectors

def compute_pca_coordinates(model):
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    vocab_pca_coordinates = pca.fit_transform(X)
    return vocab_pca_coordinates

def scatter_plot_words(vocab_coordinates, words):
    # create a scatter plot of the projection of dimensions PC1 and PC2
    word_indexes = [ model.wv.vocab[word].index for word in words ]
    coordinates = [ vocab_coordinates[j].tolist() for j in word_indexes ]
    xs=[x[0] for x in coordinates]
    ys = [x[1] for x in coordinates]
    pyplot.scatter(xs, ys)
    # words = list(model.wv.vocab)[0:100]
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(xs[i],ys[i]))
    pyplot.show()

# %%

if __name__ == "__main__":

    filename = '../data/output/w2v_model_skip_gram_win_5_dim_50_iter_20_mc_5_complete_not_segmented.dat'

    model = load_model(filename)

    print('Top 10 words       : ', model.wv.index2word[0:10])
    print('Size of vocabulary : ', len(model.wv.vocab))

    words = ['polhem', 'man', 'kvinna']

    vocab_coordinates = compute_pca_coordinates(model)
    scatter_plot_words(vocab_coordinates, words)

# farthest away from a word
# model.most_similar('polhem',topn=50)
# %%
# which word is to “x” as “y” is to “z”?
#model.most_similar(positive=['x','y'], negative=['z'])
#model.most_similar(positive=['karin','han'], negative=['hon'],topn=20)

# %%
''' Words closest to "karin" (which are essentially all first names) and
pick out the ones which are closer to “he” than to “she”.
'''
#[ x[0] for x in model.most_similar('karin',topn=2000)
#    if model.similarity(x[0],'han') > model.similarity(x[0], 'hon')]

#print(model.most_similar(positive=['teknik', 'producent'], negative=['teknik']))
#model.most_similar(positive=['flicka', 'pappa'], negative=['pojke'], topn=20)

# CHECK: https://stackoverflow.com/questions/43776572/visualise-word2vec-generated-from-gensim
