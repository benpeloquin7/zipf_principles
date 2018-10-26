"""matrix.py

Define all experiments in terms of matrix operations.

"""

import numpy as np
from .language_constructs import idxs2matrix
from .objectives import *


def is_literal(m):
    pass


def row_normalize(m, transform=lambda x: x):
    """All rows sum to 1"""
    m = transform(m)
    totals = np.sum(m, axis=1)
    return (m.transpose() / totals).transpose()


def col_normalize(m, transform=lambda x: x):
    """All cols sum to 1"""
    m = transform(m)
    totals = np.sum(m, axis=0)
    return (m / totals)


def row_multiply(m, k):
    assert m.shape[0] == len(k)
    return (m.transpose() * k).transpose()


def col_multiply(m, k):
    assert m.shape[1] == len(k)
    return m * k


def rsa_speaker(m):
    return col_normalize(m, np.exp)


def rsa_listener(m):
    return row_normalize(m, np.exp)


def run_rsa_speaker_foward(m_listener, k):
    m = m_listener
    for i in range(k):
        m = rsa_listener(m)
        m = rsa_speaker(m)
    return m


def run_rsa_listener_foward(m_speaker, k):
    m = m_speaker
    for i in range(k):
        m = rsa_speaker(m)
        m = rsa_listener(m)
    return m


if __name__ == '__main__':
    N_UTTERANCES = 4
    N_MEANINGS = 4
    idxs = (0, 1, 5, 10, 15)
    m = idxs2matrix(idxs, N_UTTERANCES, N_MEANINGS)
    k = np.array([0.1, 00.1, 0.3, 0.5])

    ce = CrossEntropy()

    print(run_rsa_speaker_foward(rsa_listener(m), 1))
    print(run_rsa_speaker_foward(rsa_listener(m), 10))

    print(run_rsa_speaker_foward(rsa_listener(m), 1))
    print(run_rsa_speaker_foward(rsa_listener(m), 10))
