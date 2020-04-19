import os
import numpy as np
from tqdm import tqdm


def _gpu_all_density(x, verbose=True, bandwidth=2.0):
    import cupy as cp
    bs = 2 ** 12
    n, d = x.shape
    kernel = cp.zeros(shape=n, dtype=x.dtype)
    iter = n // bs + 1
    for i in tqdm(range(iter)) if verbose else range(iter):
        _batch_i = cp.asarray(x[i * bs: (i + 1) * bs])
        _norm_i = cp.sum(_batch_i ** 2, axis=1, keepdims=True)
        for j in  range(iter):
            _batch_j = cp.asarray(x[j * bs: (j + 1) * bs])
            _norm_j = cp.sum(_batch_j ** 2, axis=1, keepdims=True)
            d = -2.0 * cp.dot(_batch_i, _batch_j.T) + _norm_i + _norm_j.T
            kernel[i * bs: (i + 1) * bs] += cp.sum(cp.exp(-d / bandwidth), axis=1)
    return cp.asnumpy(kernel)


def _cpu_all_density(x, verbose=True, bandwidth=2.0):
    bs = 2 ** 12
    n, d = x.shape
    kernel = np.empty(shape=n, dtype=x.dtype)
    norm = np.sum(x ** 2, axis=1, keepdims=True)
    iter = n // bs + 1

    for i in tqdm(range(iter)) if verbose else range(iter):
        _batch = x[i * bs: (i + 1) * bs]
        _norm = norm[i * bs: (i + 1) * bs]
        d = -2.0 * np.dot(_batch, x.T) + _norm + norm.T
        kernel[i * bs: (i + 1) * bs] = np.sum(np.exp(-d / bandwidth), axis=1)
    return kernel


def all_density(x, dataset, verbose=True, bandwidth=2.0):
    os.makedirs("density/", exist_ok=True)
    file_name = 'density/{}_{}.npy'.format(dataset, bandwidth)
    if os.path.isfile(file_name):
        print("# loading indices from file")
        return np.load(file_name)
    try:
        kernel = _gpu_all_density(x, verbose, bandwidth)
    except:
        kernel = _cpu_all_density(x, verbose, bandwidth)
    np.save(file_name, kernel)
    return kernel
