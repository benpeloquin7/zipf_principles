"""language_constructs.py"""

import itertools as it
from utils import *


def get_utterance_prob(utterance, utterances, utterance_probs):
    return get_item_idx(utterance, utterances, utterance_probs)


def get_meaning_prob(meaning, meanings, meaning_probs):
    return get_item_idx(meaning, meanings, meaning_probs)


class Objective:
    """Base klass for objective (cost) functions."""

    def __init__(self, speaker, listener, language):
        self.speaker = speaker
        self.listener = listener
        self.language = language
        self.events = list(it.product(language.meanings, language.utterances))

    def compute_cost(self):
        raise NotImplementedError()


class SpeakerListenerCrossEntropy(Objective):
    """H_c(P_s(u, m), P_l(u, m))"""

    def __init__(self, speaker, listener, language):
        self.speaker = speaker
        self.listener = listener
        self.language = language
        self.events = list(it.product(language.meanings, language.utterances))

    def compute_cost(self):
        raise NotImplementedError()


class FerrerCost(Objective):
    """Ferrer-i-cancho entropy objective.

    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC298679/pdf/pq0303000788.pdf

    """

    def __init__(self, speaker, listener, language):
        self.super(Objective, self).__init__(speaker, listener, language)
        self.speaker = speaker
        self.listener = listener
        self.language = language
        self.events = list(it.product(language.meanings, language.utterances))

    def compute_cost(self):
        raise NotImplementedError()


class Semantics:
    """Literal semantics (adjacency matrix)"""

    def __init__(self, mapper):
        self.mapper = mapper

    def delta(self, utterance, meaning):
        if utterance not in self.mapper:
            return False
        return meaning in self.mapper[utterance]


class Language:
    """Language."""

    def __init__(self, utterances, meanings,
                 p_utterances, p_meanings, semantics):
        self.utterances = utterances
        self.meanings = meanings
        self.p_utterances = p_utterances
        self.p_meanings = p_meanings
        self.semantics = semantics


class RSAAgent:
    """RSA interlocutor."""

    def __init__(self, language, alpha=1.):
        self.language = language
        self.alpha = alpha

    def cost(self, utterance):
        """Default cost is utterance surprisal."""
        return -np.log(get_utterance_prob(utterance,
                                          self.language.utterances,
                                          self.language.p_utterances))

    def speak(self, meaning):
        """u ~ p(u|m)"""
        p_u_scores = \
            np.array([self.speaker_score(u, meaning) for u in
                      self.language.utterances])
        p_u_scores = p_u_scores / np.sum(p_u_scores)
        return p_u_scores

    def listen(self, utterance):
        """m ~ p(m|u)"""
        pass

    def speaker_score(self, utterance, meaning):
        listener_score = self.p_listen(utterance, meaning)
        if listener_score == 0.:
            return 0.
        return \
            np.exp(
                self.alpha * (np.log(listener_score) - self.cost(utterance)))

    def listener_score(self, utterance, meaning):
        def score_meaning(u, m):
            return int(self.language.semantics.delta(u, m)) * \
                   get_meaning_prob(m, self.language.meanings,
                                    self.language.p_meanings)

        score = score_meaning(utterance, meaning)
        return score

    def p_speak(self, utterance, meaning):
        """p(u|m) \propto exp(-alpha * (log(L(m|u)-cost))"""
        Z = np.sum(self.speaker_score(u, meaning) \
                   for u in self.language.utterances)

        if Z == 0.:
            return 0.
        else:
            return self.speaker_score(utterance, meaning) / Z

    def p_listen(self, utterance, meaning):
        """p(m|u) = delta(u, m) * p(m)"""
        listener_score_ = self.listener_score(utterance, meaning)
        Z = np.sum([self.listener_score(utterance, m) \
                    for m in self.language.meanings])
        if Z == 0.:
            return 0.
        return listener_score_ / float(Z)
