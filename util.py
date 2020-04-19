import math
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def l2_dist(q: np.ndarray, x: np.ndarray):
    assert len(q.shape) == 2
    assert len(x.shape) == 2
    assert q.shape[1] == q.shape[1]
    x = x.T
    sqr_q = np.sum(q ** 2, axis=1, keepdims=True)
    sqr_x = np.sum(x ** 2, axis=0, keepdims=True)
    l2 = sqr_q + sqr_x - 2 * np.dot(q, x)
    l2[ np.nonzero(l2 < 0) ] = 0.0
    return np.sqrt(l2)

def arg_sort(q, x, metric):
    if metric == "euclid":
        dists = l2_dist(q, x)
    elif metric == "product":
        dists = - np.dot(q, x.T)
    else:
        assert False
    return np.argsort(dists)

def batch_arg_sort(q, x, k , bs, metric, verbose):
    iter = len(q) // bs + 1
    idx = np.empty(shape=(len(q), k))
    for i in tqdm(range(iter)) if verbose else range(iter):
        batch_index = idx[i * bs : (i + 1) * bs, :]
        batch_query = q[i * bs : (i + 1) * bs, :]
        batch_index[:, :] = arg_sort(batch_query, x, metric)[:, :k]
    return idx

def intersect(gs, ids):
    return np.mean([len(np.intersect1d(g, list(id))) for g, id in zip(gs, ids)])

def intersect_sizes(args):
    gs, ids = args
    return np.array([len(np.intersect1d(g, list(id))) for g, id in zip(gs, ids)])


def test_recall(X, Q, G, metric, bs=100, k=2**16, verbose=True):
    ks = np.array([1, 5, 10, 20, 50, 100, 1000])
    k = np.min([len(X), k])
    Ts = [2 ** i for i in range(1 + int(math.log2(k)))]
    sort_idx = batch_arg_sort(Q, X, k, bs, metric, verbose)

    print("# Probed \t Items \t", end="")
    for top_k in ks:
        print("top-%d\t" % (top_k), end="")
    print()
    p = Pool(len(ks))
    for t in Ts:
        ids = sort_idx[:, :t]
        items = np.mean([len(id) for id in ids])
        print("%6d \t %6d \t" % (t, items), end="")
        # tps = [intersect_sizes(G[:, :top_k], ids) / float(top_k) for top_k in ks]
        tps = p.map(intersect_sizes, [(G[:, :top_k], ids) for top_k in ks])
        tps = np.array(list(tps)) / ks[:, np.newaxis]
        rcs = [np.mean(t) for t in tps]
        vrs = [np.std(t) for t in tps]
        for rc in rcs:
            print("%.4f \t" % rc, end="")
        # for vr in vrs:
        #     print("%.4f \t" % vr, end="")
        print()