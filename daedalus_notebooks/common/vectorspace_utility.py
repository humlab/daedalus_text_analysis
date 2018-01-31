import math
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import distance
from scipy.cluster import hierarchy
import inspect

if 'extend' not in globals():
    extend = lambda a,b: a.update(b) or a
    
if 'filter_kwargs' not in globals():
    import inspect
    filter_kwargs = lambda f, args: { k:args[k] for k in args.keys() if k in inspect.getargspec(f).args }

class VectorSpaceHelper:
    
    #@staticmethod
    #def create_vector_space(lda, n_words = 50):
    #    X_n_space, _ = ModelUtility.compute_topic_terms_vector_space(lda, n_words)
    #    return X_n_space
    
    #@staticmethod
    #def reduce_dimensionality(X_m_space, reducer, normalize=True):
    #    if normalize is True:
    #        reducer = make_pipeline(reducer, Normalizer(copy=False))
    #    X_n_norm = reducer.fit(X_m_space.toarray()).transform(X_m_space.toarray())
    #    return X_n_norm
    
    @staticmethod
    def compute_pca(X_m_space, **kwargs):
        kwargs = filter_kwargs(PCA, kwargs)
        X_n_space = PCA(**kwargs).fit_transform(X_m_space.toarray())
        return X_n_space

    @staticmethod
    def compute_pca_norm(X_m_space, normalize=True, **kwargs):
        kwargs = filter_kwargs(PCA, kwargs)
        reducer = PCA(**kwargs)
        if normalize is True:
            reducer = make_pipeline(reducer, Normalizer(copy=False))
        X_n_norm = reducer.fit_transform(X_m_space.toarray())
        return X_n_norm
    
    @staticmethod
    def compute_tsne_norm(X_m_space, **kwargs):
        kwargs = filter_kwargs(TSNE, kwargs)
        kwargs = extend(dict(n_components=20, init='pca', random_state=55887, perplexity=30), kwargs)
        reducer = TSNE(**kwargs)
        X_n_norm = reducer.fit_transform(X_m_space.toarray())
        return X_n_norm
    
    @staticmethod
    def reduce_dimensions(X_m_space, method=None, **kwargs):
        reducer = None
        if method not in [ 'none', 'passthrough', 'pca', 'pca_norm', 'tsne']:
            raise Exception('Method unknown')
        if method in [ 'pca', 'pca_norm']:
            kwargs = filter_kwargs(PCA.__init__, kwargs)
            reducer = PCA(**kwargs)
            if method == 'pca_norm':
                reducer = make_pipeline(reducer, Normalizer(copy=False))
        if method == 'tsne':
            kwargs = filter_kwargs(TSNE.__init__, kwargs)
            reducer = TSNE(**kwargs)
        X = X_m_space.toarray() if hasattr(X_m_space, 'toarray') else X_m_space
        X_n_space = X_m_space if reducer is None else reducer.fit_transform(X)
        return X_n_space
    
    @staticmethod
    def compute_distance_matrix(X_n_space, metric='euclidean'):
        # https://se.mathworks.com/help/stats/pdist.html
        X = X_n_space.toarray() if hasattr(X_n_space, 'toarray') else X_n_space
        distances = distance.pdist(X, metric=metric.lower())
        distance_matrix = distance.squareform(distances)
        return distance_matrix

    @staticmethod
    def compute_clustering(correlation_matrix):
        # Z = hierarchy.linkage(correlation_matrix, 'single')
        clustering = hierarchy.linkage(correlation_matrix)
        return clustering
    