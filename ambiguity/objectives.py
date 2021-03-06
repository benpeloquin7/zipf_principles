"""objectives.py"""

import numpy as np

from .utils import *


class ObjectiveError(Exception):
    pass


class Objective:
    def __init__(self):
        pass

    def compute_cost(self, m1, m2):
        raise NotImplementedError()


def inf2zero(log_M):
    """Convert -np.Inf entries to 0.

    Slightly hacky, but useful for information-theoretic
    costs and divergences like KL and CE.
    """
    for i, _ in enumerate(log_M):
        for j, _ in enumerate(log_M[i]):
            if log_M[i][j] == -np.Inf:
                log_M[i][j] = 0
    return log_M


class CrossEntropy(Objective):
    def compute_cost(self, m1, m2):
        return -np.sum(m1 * inf2zero(np.log(m2)))

    def get_speaker_effort(self, m1, m2):
        pass

    def get_listener_effort(self, m1, m2):
        pass


class KL(Objective):
    def compute_cost(self, m1, m2):
        n_rows, n_cols = m1.shape
        total = 0
        for i in range(n_rows):
            for j in range(n_cols):
                total += m1[i][j] * np.log(m1[i][j] / m2[i][j]) \
                    if m2[i][j] != 0 and m1[i][j] != 0 else 0.
        return total


class FerrerObjective(Objective):
    def compute_cost(self, m1, m2):
        return -np.sum(m1 * inf2zero(np.log(m1))) - np.sum(m2 * inf2zero(np.log(m2)))


class SymmetricCrossEntropy(Objective):
    def compute_cost(self, m1, m2):
        return -np.sum(m1 * inf2zero(np.log(m2))) - np.sum(m2 * inf2zero(np.log(m1)))


class SingleEntropy(Objective):
    def compute_cost(self, m):
        return -np.sum(m * inf2zero(np.log(m)))


if __name__ == '__main__':
    N_UTTERANCES = 4
    N_MEANINGS = 4
    idxs = (0, 1, 5, 10, 15)
    m = idxs2matrix(idxs, N_UTTERANCES, N_MEANINGS)
    ce = CrossEntropy()
    kl = KL()
    ferrer = FerrerObjective()
    symmetric_ce = SymmetricCrossEntropy()

   # print(ce.compute_cost(m, m))
   # print(kl.compute_cost(m, m))
   # print(ferrer.compute_cost(m, m))
   # print(symmetric_ce.compute_cost(m, m))
