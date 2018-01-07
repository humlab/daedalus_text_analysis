import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import networkx as nx
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from topic_modelling.viz_utility import get_model_names, load_gensim_lda_model
import matplotlib as plt

def vectorize_topic_terms(lda, n_words=100):
    vec = DictVectorizer()
    topic_terms = [
        dict({ (x[1], x[0]) for x in lda.show_topic(i, n_words) })
            for i in range(0, lda.num_topics)
    ]
    X = vec.fit_transform(topic_terms)
    return X

def compute_PCA_euclidean_distances(lda, random_state=8675309):
    X = vectorize_topic_terms(lda)
    pca_norm = make_pipeline(PCA(n_components=20,random_state=None), Normalizer(copy=False))
    X_pca_norm = pca_norm.fit(X.toarray()).transform(X.toarray())
    coordinates = squareform(pdist(X_pca_norm, metric="euclidean"))
    return coordinates

def plot_PCA_euclidean_distanc_network(coordinates, min_norm_weight=0.9):
    G = nx.Graph()

    max_weight = coordinates.max()

    for i in range(0, coordinates.shape[0]):
        for j in range(0, coordinates.shape[1]):
            G.add_edge(i, j, {
                    "weight": 0 if i == j else coordinates[i, j] / max_weight # 1.0 / coordinates[i, j]
            })

    # Only display strong links i.e edges greater then min_norm_weight
    edges = [ (i, j) for i, j, w in G.edges(data=True) if w['weight'] > min_norm_weight ]
    #edge_weight=dict([((u,v,),int(d['weight'])) for u,v,d in G.edges(data=True)])
    #pos = nx.graphviz_layout(G, prog="twopi") # twopi, neato, circo
    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size = 200, alpha=.5)
    nx.draw_networkx_edges(G, pos, edgelist = edges, width=1)
    #nx.draw_networkx_edge_labels(G, pos ,edge_labels=edge_weight)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

    # plt.savefig("network")
    # plt.close()

    plt.show()

if __name__ == '__main__':

    source_folder = '/tmp'
    models_names = get_model_names(source_folder)

    basename = models_names[-1]
    data_folder = os.path.join(source_folder, basename)
    lda = load_gensim_lda_model(data_folder, basename)

    coordinates = compute_PCA_euclidean_distances(lda)
    plot_PCA_euclidean_distanc_network(coordinates)
