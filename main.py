import os
import numpy as np

from kde import all_density
from util import test_recall
from vecs_io import loader
from weighted_vq import weighted_kmeans


def run(dataset, metric, bandwidths):
    X, _, Q, G = loader(dataset, top_k=1000, ground_metric=metric, folder='../../data/')
    assert G is not None

    norm = np.max(np.linalg.norm(X, axis=1))
    X /= norm
    Q /= norm

    os.makedirs("logs/{}/{}".format(dataset, metric), exist_ok=True)
    filename = "logs/{}/{}/original_kmeans.txt".format(dataset, metric)
    print("writing to filename {}".format(filename))
    centroid, codes = weighted_kmeans(X, w=None, k=256, iter=20, minit='points')
    compressed = centroid[codes, :]
    test_recall(compressed, Q, G, metric=metric, file=open(filename, 'w'))

    for bandwidth in bandwidths:
        filename = "logs/{}/{}/weighted_kmeans_{}.txt".format(dataset, metric, bandwidth)
        print("writing to filename {}".format(filename))
        density = all_density(X, dataset=dataset, bandwidth=bandwidth)
        print(density)
        centroid, codes = weighted_kmeans(X, w=density, k=256, iter=20, minit='points')
        compressed = centroid[codes, :]
        test_recall(compressed, Q, G, metric=metric, file=open(filename, 'w') )


if __name__ == '__main__':
    datasets = ['sift1m']
    metrics = ['product']
    bandwidths = [0.02, 0.1, 0.2, 0.5, 1, 2, 4, 8]
    for dataset in datasets:
        for metric in metrics:
            run(dataset, metric, bandwidths)
