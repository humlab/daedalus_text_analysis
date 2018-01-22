
from matplotlib import pyplot
import pandas as pd
import warnings
from matplotlib.cbook import MatplotlibDeprecationWarning
warnings.filterwarnings(action="ignore", category=MatplotlibDeprecationWarning)

def plot_xy_word_chart(df, filename=None, xlabel=None, ylabel=None, figsize=(16,8), plot_arrow=False):
    
    fig = pyplot.figure(figsize=figsize)
    
    #pyplot.plot([0,0.75], [0,0.75])
    
    if not xlabel is None: pyplot.xlabel(xlabel)
    if not ylabel is None: pyplot.ylabel(ylabel)
        
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df['x'], df['y'],marker='o')
    
    for i, txt in enumerate(df['word']):
        ax.annotate(txt, xy=(df['x'].iloc[i], df['y'].iloc[i])) #, textcoords = 'offset points', ha = 'left', va = 'top', **TEXT_KW)
        if plot_arrow is True and (i % 2 == 1):
             ax.arrow(df['x'].iloc[i-1], df['y'].iloc[i-1], df['x'].iloc[i], df['y'].iloc[i], color='green')
    pyplot.show()
    
    # if filename is not None:
    #    pyplot.savefig(filename)

# This belongs to notebook:
def matplotlib_plot_anthologies(word_vectors, xpair, ypair, word_list, filename=None, figsize=(16,8)):
    df = ModelUtility.compute_similarity_to_anthologies(word_vectors, xpair, ypair, word_list)
    
    xlabel = '{} {} {}'.format(xpair[1], ' ' * 100, xpair[0])
    ylabel = '{} {} {}'.format(ypair[1], ' ' * 100, ypair[0])
    
    plot_xy_word_chart(df, xlabel=xlabel,ylabel=ylabel, filename=filename, figsize=figsize)
    
df_toplist = Utility.read_excel("./data/relevant_words.xlsx", "toplist1")

word_list = list(df_toplist.words.values)
word_list = [ x for x in word_list if x in word_vectors.vocab.keys() ]
    
matplotlib_plot_anthologies(word_vectors, ('good', 'evil'), ('europe', 'africa'), word_list)
