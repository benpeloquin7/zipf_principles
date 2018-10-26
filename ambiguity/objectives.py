"""objectives.py"""

import numpy as np

from .utils import *


class ObjectiveError(Exception):
    pass


class Objective:
    def __init__(self):
        pass

    @staticmethod
    def _check_matrix(m):
        return np.sum(m) == 1

    def compute_cost(self, m1, m2):
        raise NotImplementedError()


class CrossEntropy(Objective):
    def compute_cost(self, m1, m2):
        assert (self._check_matrix(m1) and self._check_matrix(m2))
        return np.sum(m1 * np.log(m2))


class KL(Objective):
    def compute_cost(self, m1, m2):
        assert (self._check_matrix(m1) and self._check_matrix(m2))
        return np.sum(m1 * np.log(m2 / m1))


class FerrerObjective(Objective):
    def compute_cost(self, m1, m2):
        assert (self._check_matrix(m1) and self._check_matrix(m2))
        return np.sum(m1 * np.log(m1)) + np.sum(m2 * np.log(m2))


class SymmetricCrossEntropy(Objective):
    def compute_cost(self, m1, m2):
        assert (self._check_matrix(m1) and self._check_matrix(m2))
        return np.sum(m1 * np.log(m2)) + np.sum(m2 * np.log(m1))


if __name__ == '__main__':
    N_UTTERANCES = 4
    N_MEANINGS = 4
    idxs = (0, 1, 5, 10, 15)
    m = idxs2matrix(idxs, N_UTTERANCES, N_MEANINGS)
    ce = CrossEntropy()
    kl = KL()
    ferrer = FerrerObjective()
    symmetric_ce = SymmetricCrossEntropy()

    print(ce.compute_cost(m, m))
    print(kl.compute_cost(m, m))
    print(ferrer.compute_cost(m, m))
    print(symmetric_ce.compute_cost(m, m))
