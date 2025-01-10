from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import *


def my_Spectral(npmat, n_clusters=2, metrics='cosine', random_state=0):
    if metrics == 'cosine':
        A = cosine_similarity(npmat)
    elif metrics == 'euclidean':
        A = euclidean_distances(npmat)
    elif metrics == 'laplacian':
        A = laplacian_kernel(npmat)
    elif metrics == 'manhattan':
        A = manhattan_distances(npmat)

    labels = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(A)
    return labels

