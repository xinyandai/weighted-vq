import math
import numpy as np

from tqdm import tqdm
from vecs_io import loader
from sorter import BatchSorter
from weighted_vq import weighted_kmeans


def all_density(x, bs=2**12, verbose=True):
    n, d = x.shape
    kernel = np.empty(shape=n, dtype=x.dtype)
    x = np.sum(x**2, axis=1, keepdims=True)
    iter = n // bs + 1

    for i in tqdm(range(iter)) if verbose else range(iter):
        batch = x[i * bs : (i + 1) * bs]
        d = -2.0 * np.dot(batch, x.T) + batch + x.T
        kernel[i * bs : (i + 1) * bs] =np.sum(np.exp(-d / 2.0), axis=1)
    return kernel


def execute(compressed, X, Q, G, metric):
    print("# ranking metric {}".format(metric))
    print("# sorting items")
    Ts = [2 ** i for i in range(2+int(math.log2(len(X))))]
    recalls = BatchSorter(compressed, Q, X, G, Ts, metric=metric, batch_size=200).recall()
    print("# searching!")
    print("expected items, overall time, avg recall, avg precision, avg error, avg items")
    for i, (t, recall) in enumerate(zip(Ts, recalls)):
        print("{}, {}, {}, {}, {}, {}".format(
            2**i, 0, recall, recall * len(G[0]) / t, 0, t))


if __name__ == '__main__':
    dataset = 'yahoomusic'
    topk = 20
    codebook = 4
    Ks = 256
    metric = 'product'

    X, _, Q, G = loader(dataset, topk, metric, folder='../../data/')
    assert G is not None

    density = all_density(X)
    centroid, codes = weighted_kmeans(X, w=density, k=256, iter=20, minit='points')
    compressed = centroid[codes, :]
    execute(compressed, X, Q, G, metric)

    centroid, codes = weighted_kmeans(X, w=None, k=256, iter=20, minit='points')
    compressed = centroid[codes, :]
    execute(compressed, X, Q, G, metric)
