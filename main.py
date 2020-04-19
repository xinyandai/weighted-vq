import math
import numpy as np

from kde import all_density
from util import test_recall
from vecs_io import loader
from weighted_vq import weighted_kmeans


if __name__ == '__main__':
    dataset = 'sift1m'
    # metric = 'product'
    # bandwidth = 4

    metric = 'euclid'
    bandwidth = 2

    X, _, Q, G = loader(dataset, top_k=1000, ground_metric=metric, folder='../../data/')
    assert G is not None

    norm = np.max(np.linalg.norm(X, axis=1))
    X /= norm
    Q /= norm

    density = all_density(X, dataset=dataset, bandwidth=bandwidth)
    print(density)
    centroid, codes = weighted_kmeans(X, w=density, k=256, iter=20, minit='points')
    compressed = centroid[codes, :]
    test_recall(compressed, Q, G, metric=metric)

    centroid, codes = weighted_kmeans(X, w=None, k=256, iter=20, minit='points')
    compressed = centroid[codes, :]
    test_recall(compressed, Q, G, metric=metric)
