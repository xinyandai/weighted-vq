import os
import argparse
import numpy as np

from vq import PQ
from kde import all_density
from util import test_recall
from vecs_io import loader


def run(dataset, metric, bandwidths, seed):
    codebook = 4
    X, _, Q, G = loader(
        dataset, top_k=1000, ground_metric=metric, folder="../../data/"
    )
    assert G is not None

    norm = np.max(np.linalg.norm(X, axis=1))
    X /= norm
    Q /= norm

    dir = "logs/{}/{}/{}".format(dataset, metric, seed)
    os.makedirs(dir, exist_ok=True)

    filename = dir + "/pq{}_original_kmeans.txt".format(codebook)
    print("writing to filename {}".format(filename))
    np.random.seed(seed)
    pq = PQ(M=codebook, Ks=256)
    compressed = pq.fit(X, w=None, iter=20).compress(X)
    test_recall(compressed, Q, G, metric=metric, file=open(filename, 'w'))

    filename = dir + "/pq{}_balance_kmeans.txt".format(codebook)
    print("writing to filename {}".format(filename))
    np.random.seed(seed)
    pq = PQ(M=codebook, Ks=256)
    compressed = pq.fit(X, w=-1, iter=20).compress(X)
    test_recall(compressed, Q, G, metric=metric, file=open(filename, "w"))

    for bandwidth in bandwidths:
        filename = "{}/pq{}_weighted_kmeans_{}.txt".format(dir, codebook, bandwidth)
        print("writing to filename {}".format(filename))
        density = all_density(X, dataset=dataset, bandwidth=bandwidth)
        print(density)

        np.random.seed(seed)
        pq = PQ(M=codebook, Ks=256)
        compressed = pq.fit(X, w=density, iter=20).compress(X)
        with open(filename, 'w') as file:
            test_recall(compressed, Q, G, metric=metric, file=file)
        del compressed
        del pq
        del density


if __name__ == "__main__":
    datasets = ["netflix", "yahoomusic", "sift1m", "imagenet"]
    metrics = ["euclid", "product"]
    bandwidths = [0.02, 0.1, 0.2, 0.5, 1, 2, 4, 8]
    seeds = range(10)

    parser = argparse.ArgumentParser(
        description="Process input method and parameters."
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="choose data set name"
    )
    parser.add_argument(
        "--metric", type=str, default=None, help="metric of ground truth"
    )
    parser.add_argument(
        "--bandwidth", type=int, default=None, help="bandwidth for density",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="seed for vq initialization"
    )
    args = parser.parse_args()
    print(args)
    if args.dataset is not None:
        datasets = [args.dataset]
    if args.metric is not None:
        metrics = [args.metric]
    if args.bandwidth is not None:
        bandwidths = [args.bandwidth]
    if args.dataset is not None:
        seeds = [args.seed]

    for seed in seeds:
        for dataset in datasets:
            for metric in metrics:
                run(dataset, metric, bandwidths, seed)
