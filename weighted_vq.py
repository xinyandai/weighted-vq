import warnings
import numpy as np
from tqdm import tqdm
from scipy.cluster.vq import _vq
from scipy.cluster.vq import _valid_miss_meth
from scipy.cluster.vq import _valid_init_meth
from scipy.cluster.vq import _asarray_validated


def weighted_kmeans(data, w, k, iter=10, minit='random',
                    missing='warn', check_finite=True, verbose=False):
    """
    Returns
    -------
    centroid : ndarray
        A 'k' by 'N' array of centroids found at the last iteration of
        k-means.
    label : ndarray
        label[i] is the code or index of the centroid the
        i'th observation is closest to.
    """
    n, d = data.shape
    if w is not None:
        w = w.reshape((n, 1))

    assert int(iter) > 0, "Invalid iter (%s)" % iter
    miss_meth = _valid_miss_meth[missing]

    data = _asarray_validated(data, check_finite=check_finite)


    if data.size < 1:
        raise ValueError("Empty input is not supported.")

    # If k is not a single value it should be compatible with data's shape
    if minit == 'matrix' or not np.isscalar(k):
        code_book = np.array(k, copy=True)
        if data.ndim != code_book.ndim:
            raise ValueError("k array doesn't match data rank")
        nc = len(code_book)
        if data.ndim > 1 and code_book.shape[1] != d:
            raise ValueError("k array doesn't match data dimension")
    else:
        nc = int(k)

        if nc < 1:
            raise ValueError("Cannot ask kmeans2 for %d clusters"
                             " (k was %s)" % (nc, k))
        elif nc != k:
            warnings.warn("k was not an integer, was converted.")

        try:
            init_meth = _valid_init_meth[minit]
        except KeyError:
            raise ValueError("Unknown init method %r" % (minit,))
        else:
            code_book = init_meth(data, k)

    for _ in tqdm(range(iter)) if verbose else range(iter):
        # Compute the nearest neighbor for each obs using the current code book
        label = _vq.vq(data, code_book)[0]
        # Update the code book by computing centroids
        if w is None:
            new_code_book, has_members = _vq.update_cluster_means(data, label, nc)
        else:
            new_code_book, has_members = _vq.update_cluster_means(data * w, label, nc)
            density_sum, _ = _vq.update_cluster_means(w, label, nc)
            new_code_book /= density_sum
        if not has_members.all():
            miss_meth()
            # Set the empty clusters to their previous positions
            new_code_book[~has_members] = code_book[~has_members]
        code_book = new_code_book

    return code_book, _vq.vq(data, code_book)[0]
