"""language_constructs.py"""

from .utils import *


def get_utterance_prob(utterance, utterances, utterance_probs):
    return get_item_idx(utterance, utterances, utterance_probs)


def get_meaning_prob(meaning, meanings, meaning_probs):
    return get_item_idx(meaning, meanings, meaning_probs)


##############
# Objectives #
##############

class Objective:
    """Base klass for objective (cost) functions."""

    def __init__(self, speaker, listener, language):
        self.speaker = speaker
        self.listener = listener
        self.language = language
        self.events = list(it.product(self.language.utterances,
                                      self.language.meanings))

    def compute_cost(self):
        raise NotImplementedError()


class CrossEntropy(Objective):
    """H_c(P_s(u, m), P_l(u, m))"""

    def compute_cost(self):
        """Compute H_c(P_s(u, m), P_l(u, m))

        = \sum_{u, m}P_s(u, m)log(P_l(u, m))
        = \sum_{u, m}P_s(u, m)log(p(u)) + \sum_{u, m}P_s(u, m)log(L(m|u))
        = E[speaker effort] + E[listener effort]

        Returns
        -------
        tuple of floats
            (Cross entropy score, avg speaker effort, avg listener effort)

        """
        expected_speaker_effort = 0.
        expected_listener_effort = 0.
        for u, m in self.events:
            # meanings prior
            p_m = get_meaning_prob(m,
                                   self.language.meanings,
                                   self.language.p_meanings)
            # utterances prior
            p_u = get_utterance_prob(u,
                                     self.language.utterances,
                                     self.language.p_utterances)
            # S(u|m)
            s_u_given_m = self.speaker.p_speak(u, m)
            # L(m|u)
            l_m_given_u = self.listener.p_listen(u, m)
            # P_s(u, m)
            p_joint_speaker = p_m * s_u_given_m
            expected_speaker_effort += p_joint_speaker * -np.log(p_u)
            expected_listener_effort += \
                (p_joint_speaker * -np.log(l_m_given_u)) \
                    if l_m_given_u != 0. else 0.

        ce_score = (expected_speaker_effort + expected_listener_effort)

        return ce_score, expected_speaker_effort, expected_listener_effort


class SymmetricCrossEntropy(Objective):
    """(speaker -> listener CE) + (listener -> speaker CE)

    \lambda * H_c(P_s(u, m), P_l(u, m)) + \
            (1-\lambda) H_c(P_l(u, m), P_s(u, m)

    """

    def compute_cost(self, eta=0.5):
        """

        Returns
        -------
        tuple of floats
            (Cross entropy score, avg speaker effort, avg listener effort)

        """
        expected_speaker_effort_1 = 0.
        expected_listener_effort_1 = 0.
        expected_speaker_effort_2 = 0.
        expected_listener_effort_2 = 0.
        for u, m in self.events:
            # meanings prior
            p_m = get_meaning_prob(m,
                                   self.language.meanings,
                                   self.language.p_meanings)
            # utterances prior
            p_u = get_utterance_prob(u,
                                     self.language.utterances,
                                     self.language.p_utterances)
            # S(u|m)
            s_u_given_m = self.speaker.p_speak(u, m)
            # L(m|u)
            l_m_given_u = self.listener.p_listen(u, m)
            # P_s(u, m)
            p_joint_speaker = p_m * s_u_given_m
            # P_l(m, u)
            p_joint_listener = p_u * l_m_given_u

            expected_speaker_effort_1 += p_joint_speaker * -np.log(p_u)
            expected_listener_effort_1 += \
                (p_joint_speaker * -np.log(l_m_given_u)) \
                    if l_m_given_u != 0. else 0.
            expected_speaker_effort_2 += p_joint_listener * -np.log(p_m)
            expected_listener_effort_2 += \
                (p_joint_listener * -np.log(s_u_given_m)) \
                    if s_u_given_m != 0. else 0.

        H_s2l = (expected_speaker_effort_1 + expected_listener_effort_1)
        H_l2s = (expected_speaker_effort_2 + expected_listener_effort_2)
        interplotated_total = eta * H_s2l + (1 - eta) * H_l2s

        return interplotated_total, H_s2l, H_l2s


class FerrerObjective(Objective):
    """Ferrer-i-cancho entropy objective.

    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC298679/pdf/pq0303000788.pdf

    """

    def compute_cost(self, eta=0.5):
        """Compute \eta * H_s(u, m) + (1 - \eta) * H_s(u, m)

        Returns
        -------
        tuple of floats
            (total cost, H_s, H_l)

        """
        H_s = 0.
        H_l = 0.
        for u, m in self.events:
            # meanings prior
            p_m = get_meaning_prob(m, self.language.meanings,
                                   self.language.p_meanings)
            # utterances prior
            p_u = get_utterance_prob(u, self.language.utterances,
                                     self.language.p_utterances)
            # S(u|m)
            s_u_given_m = self.speaker.p_speak(u, m)
            # L(m|u)
            l_m_given_u = self.listener.p_listen(u, m)
            # P_s(u, m)
            p_joint_speaker = p_m * s_u_given_m
            # P_l(u, m)
            p_joint_listener = p_u * l_m_given_u
            H_s += p_joint_speaker * np.log(p_joint_speaker) \
                if p_joint_speaker != 0. else 0.
            H_l += p_joint_listener * np.log(p_joint_listener) \
                if p_joint_speaker != 0. else 0.
        H_s *= -1
        H_l *= -1
        total = eta * H_s + (1 - eta) * H_l
        return total, H_s, H_l


class KL(Objective):
    """Kullback-Leibler Divergence.

    """
    def compute_cost(self):
        """Compute \eta * H_s(u, m) + (1 - \eta) * H_s(u, m)

        Returns
        -------
        float
            KL divergence.

        """
        KL = 0.
        for u, m in self.events:
            # meanings prior
            p_m = get_meaning_prob(m, self.language.meanings,
                                   self.language.p_meanings)
            # utterances prior
            p_u = get_utterance_prob(u, self.language.utterances,
                                     self.language.p_utterances)
            # S(u|m)
            s_u_given_m = self.speaker.p_speak(u, m)
            # L(m|u)
            l_m_given_u = self.listener.p_listen(u, m)
            # P_s(u, m)
            p_joint_speaker = p_m * s_u_given_m
            # P_l(u, m)
            p_joint_listener = p_u * l_m_given_u

            KL += p_joint_speaker * \
                  (np.log(p_joint_speaker) - np.log(p_joint_listener)) \
                if p_joint_listener != 0. else 0.

        return KL


############
# Language #
############

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


#######
# RSA #
#######

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


if __name__ == '__main__':
    p_utterances = [0.6, 0.25, 0.1, 0.05]
    p_meanings = [0.6, 0.25, 0.1, 0.05]
    sems = idxs2matrix([0, 4, 5, 10], 4, 4)
    utterances = ['a', 'b', 'c', 'd']
    meanings = [1, 2, 3, 4]
    d = matrix2dict(sems, utterances, meanings)
    sems = Semantics(d)
    language = Language(utterances, meanings, p_utterances, p_meanings, sems)
    agent = RSAAgent(language)
    obj1 = CrossEntropy(agent, agent, language)
    obj2 = FerrerObjective(agent, agent, language)
    obj3 = KL(agent, agent, language)
    print(obj1.compute_cost())
    print(obj2.compute_cost(eta=0.4))
    print(obj3.compute_cost())
