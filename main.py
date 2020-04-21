import os
import numpy as np

from vq import PQ
from kde import all_density
from util import test_recall
from vecs_io import loader



def run(dataset, metric, bandwidths, seed):
    X, _, Q, G = loader(dataset, top_k=1000, ground_metric=metric, folder='../../data/')
    assert G is not None

    norm = np.max(np.linalg.norm(X, axis=1))
    X /= norm
    Q /= norm

    dir = "logs/{}/{}/{}".format(dataset, metric, seed)
    os.makedirs(dir, exist_ok=True)
    filename = dir + "/pq8_original_kmeans.txt"
    print("writing to filename {}".format(filename))

    np.random.seed(seed)
    pq = PQ(M=8, Ks=256)
    compressed = pq.fit(X, w=None, iter=20).compress(X)
    test_recall(compressed, Q, G, metric=metric, file=open(filename, 'w'))

    for bandwidth in bandwidths:
        filename = "{}/pq8_weighted_kmeans_{}.txt".format(dir, bandwidth)
        print("writing to filename {}".format(filename))
        density = all_density(X, dataset=dataset, bandwidth=bandwidth)
        print(density)

        pq = PQ(M=8, Ks=256)
        compressed = pq.fit(X, w=density, iter=20).compress(X)
        test_recall(compressed, Q, G, metric=metric, file=open(filename, 'w'))
        del compressed
        del pq
        del density


if __name__ == '__main__':
    datasets = ['netflix', 'yahoomusic', 'sift1m', 'imagenet']
    metrics = ['euclid', 'product']
    bandwidths = [0.02, 0.1, 0.2, 0.5, 1, 2, 4, 8]

    for seed in range(10):
        for dataset in datasets:
            for metric in metrics:
                run(dataset, metric, bandwidths, seed)
