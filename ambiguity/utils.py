"""utils.py"""

from collections import defaultdict
import itertools as it
import logging
import numpy as np

logging.getLogger().setLevel(logging.INFO)


def all_meanings_available_filter(M):
    """

    Parameters
    ----------
    M: np.array
        A boolean matrix.

    Returns
    -------
    bool
        True if all meanings can be talked about (each col as some val.)

    """
    return all(np.sum(M, axis=0) > 0.)


def is_valid_matrix(M, filter_fn):
    """Filter on boolean matrices.

    E.g. we may want to make sure that we only consider
    boolean matrices that allow all meanings to be talked about.
    """
    return filter_fn(M)


def idxs2matrix(idxs, m, n):
    """Convert a list of indices to an M x N matrix.

    Example
    -------
    m, n = 3, 3
    idxs = [0, 2, 3, 8]

    idxs2matrix(idxs, m, n) -->

            [[1., 0., 1.],
             [1., 0., 0.],
             [0., 0., 1.]]

    """
    d = []
    for row in range(m):
        curr_row = []
        for col in range(n):
            if (row * n + col) in idxs:
                curr_row.append(1.)
            else:
                curr_row.append(0.)
        d.append(curr_row)
    return np.array(d)


def generate_all_boolean_matrices(m, n, n_true):
    """Generate all M x N boolean matrices with
    n_true ones and m*n-n_true zeros.
    """
    matrices = []
    idxs_range = range(m * n)
    all_subsets = it.combinations(idxs_range, n_true)
    for idxs in all_subsets:
        matrices.append((idxs, idxs2matrix(idxs, m, n)))
    return matrices


def matrix2dict(M, utterances, meanings):
    """Rows correspond to utterances, Cols correspond to meanings."""
    assert M.shape[0] == len(utterances) and M.shape[1] == len(meanings)
    d = defaultdict(list)
    for row in range(M.shape[0]):
        for col in range(M.shape[1]):
            if M[row][col] != 0.:
                d[utterances[row]].append(meanings[col])
    return d


def get_item_idx(item, arr1, arr2):
    assert len(arr1) == len(arr2)
    try:
        idx = [i for i, x in enumerate(arr1) if x == item][0]
    except:
        raise Exception("Problem indexing into item array.")
    return arr2[idx]


def largest_denotation(idxs, n, m):
    m = idxs2matrix(idxs, n, m)
    return np.max(np.sum(m, axis=0))

def contains_ambiguities(idxs, n, m):
    return largest_denotation(idxs, n, m) != 1.