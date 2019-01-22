"""tests.py"""

import numpy as np
import unittest

from .language_constructs import Semantics, Language, RSAAgent, CrossEntropy
from .matrix import rsa_speaker_forward, rsa_listener_forward, row_multiply, \
    col_multiply, \
    rsa_listener, rsa_speaker, col_normalize, row_normalize
from .objectives import CrossEntropy as CE
from .utils import idxs2matrix, matrix2dict


def test_language2matrix():
    pass


def test_rsa_speaker():
    N_UTTERANCES = 4
    N_MEANINGS = 4
    idxs = (0, 5, 10, 15)
    m = idxs2matrix(idxs, N_UTTERANCES, N_MEANINGS)
    assert rsa_speaker_forward(m, 1) == np.array(m)

def test_matrix_formalism():

    p_utterances = [0.1, 0.3, 0.6]
    p_meanings = [0.1, 0.3, 0.6]
    l0 = idxs2matrix([0, 3, 4, 8], 3, 3)

    # Non-matrix formulation
    utterances = ['a', 'b', 'c']
    meanings = [1, 2, 3]
    d = matrix2dict(l0, utterances, meanings)
    sems = Semantics(d)
    language = Language(utterances, meanings, p_utterances, p_meanings, sems)
    agent = RSAAgent(language)
    cost1 = CrossEntropy(agent, agent, language).compute_cost()

    import pdb;
    pdb.set_trace();

    listener = rsa_listener(l0, p_meanings, p_utterances)
    speaker = rsa_speaker(l0, p_meanings)
    # Matrix formulation
    m_l0 = row_multiply(l0, p_meanings)
    m_listener = col_multiply(rsa_listener_forward(m_l0, 1), p_utterances)
    m_speaker = row_multiply(rsa_speaker_forward(m_l0, 1), p_meanings)
    ce_matrix_cost = CE()
    cost2 = ce_matrix_cost.compute_cost(m_listener, m_speaker)






if __name__ == '__main__':
    test_matrix_formalism()




