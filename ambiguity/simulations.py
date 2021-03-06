"""simulations.py"""

import pandas as pd
import tqdm

from .agents import *
from .objectives import *
from .utils import *

logging.getLogger().setLevel(logging.INFO)


class Simulation:
    def __init__(self):
        self.name = 'BaseSimulation'

    def run(self, n_sims,
            utterance_order_fn,
            utterance_order_fn_name,
            meaning_order_fn,
            meaning_order_fn_name,
            speaker_k,
            listener_k,
            speaker_alpha,
            listener_alpha,
            verbose=False):
        NotImplementedError()


class SimpleSimulation(Simulation):
    def __init__(self):
        self.name = 'SimpleSimulation'

    def run(self, n_sims=1):
        N_UTTERANCES = 4
        N_MEANINGS = 4
        idxs = (0, 1, 5, 10, 15)
        m = idxs2matrix(idxs, N_UTTERANCES, N_MEANINGS)
        p_utterances = np.array([0.1, 0.1, 0.3, 0.5])
        p_meanings = np.array([0.1, 0.1, 0.3, 0.5])

        ce = CrossEntropy()
        kl = KL()
        ferrer = FerrerObjective()
        symmetric_ce = SymmetricCrossEntropy()
        single_entropy = SingleEntropy()

        m_listener = col_multiply(listener(m, p_utterances, p_meanings, alpha=1., k=0), p_utterances)
        m_speaker= row_multiply(speaker(m, p_utterances, p_meanings, alpha=1., k=1), p_meanings)
        return \
            {
                'CE': ce.compute_cost(m_speaker, m_listener),
                'KL': kl.compute_cost(m_speaker, m_listener),
                'ferrer': ferrer.compute_cost(m_speaker, m_listener),
                'symmetric_ce': symmetric_ce.compute_cost(m_speaker, m_listener),
                'speaker_entropy': single_entropy.compute(m_speaker),
                'listener_entropy': single_entropy.compute(m_listener)
            }


class ContextSimulation(Simulation):
    """Basic context simulation."""
    def run(self,
            n_sims,
            n_contexts,
            utterance_order_fn,
            utterance_order_fn_name,
            meaning_order_fn,
            meaning_order_fn_name,
            speaker_k=1,
            listener_k=1,
            speaker_alpha=1,
            listener_alpha=1,
            verbose=False,
            matrix_density=1):

        all_results = []
        pbar = tqdm.tqdm(total=n_sims * verbose)
        for sim_id in range(n_sims):
            # Costs
            ce = CrossEntropy()
            kl = KL()
            ferrer = FerrerObjective()
            sym_ce = SymmetricCrossEntropy()
            # Set-up
            N_UTTERANCES = 4
            N_MEANINGS = 4
            p_utterances = np.random.dirichlet([utterance_order_fn(i) for i in range(N_UTTERANCES)])
            p_meanings = np.random.dirichlet([meaning_order_fn(i) for i in range(N_MEANINGS)])
            p_cs = np.random.dirichlet(
                [1. for _ in range(n_contexts)])
            # Uniform prior over contexts (NOT USING)
            # p_cs = np.array([1. for _ in range(n_contexts)])
            # p_cs = p_cs / np.sum(p_cs)
            p_cs = p_cs.reshape((n_contexts, 1))
            # Populate contexts
            contexts = [p_meanings]

            for i in range(1, n_contexts):
                # Sample new context from Dirichlet
                # new_p_m populates a totally new prior over meanings.
                new_p_m = np.random.dirichlet([meaning_order_fn(k) for k in range(N_MEANINGS)])
                # or
                # New context is simply a permutation
                # contexts.append(np.roll(contexts[0], i+1))
                contexts.append(new_p_m)
            # Baseline model
            # (context isn't informative so we take)
            # p(m) = \sum_c p(m|c)p(c)
            p_m_no_context = np.array(contexts)
            p_m_no_context *= p_cs
            p_m_no_context = \
                np.sum(p_m_no_context, axis=0) / np.sum(p_m_no_context)

            # Generate all valid matrices (as defined by filter)
            all_matrices = \
                [(idxs, m) for (idxs, m) in
                 generate_all_boolean_matrices_upto(N_UTTERANCES,
                                                    N_MEANINGS,
                                                    N_MEANINGS+matrix_density) \
                 if all_utterances_meanings_used_filter(m)]

            d_results = defaultdict(dict)
            for idxs, m in all_matrices:
                for context_id, context in enumerate(contexts):
                    p_meanings_ = context

                    # P_{speaker}(u, m | c)
                    m_speaker = col_multiply(speaker(m, p_utterances,
                                                     p_meanings_,
                                                     speaker_alpha,
                                                     speaker_k),
                                             p_meanings_)

                    # P_{listener}(u, m | c)
                    m_listener = row_multiply(listener(m, p_utterances,
                                                       p_meanings_,
                                                       listener_alpha,
                                                       listener_k),
                                              p_utterances)
                    # p(u) term
                    # in \sum_{u, m}P_s(u, m)*log(p(u))
                    m_simple_speaker_costs = row_multiply(m, p_utterances)

                    # L(m|u, c)p(c) term
                    # in \sum_{u, m}P_s(u, m)*log(L(m|u))
                    m_simple_listener_costs = \
                        listener(m, p_utterances,
                                 p_meanings_,
                                 listener_alpha,
                                 listener_k)

                    # Safe-keeping for clean-use below
                    args = [('ce', m_speaker, m_listener, ce),
                            ('kl', m_speaker, m_listener, kl),
                            ('ferrer', m_speaker, m_listener, ferrer),
                            ('sym_ce', m_speaker, m_listener, sym_ce),
                            ('speaker_entropy', m_speaker,
                             m_simple_speaker_costs, ce),
                            ('listener_entropy', m_speaker,
                             m_simple_listener_costs, ce)]

                    # Record objectives
                    for arg in args:
                        typ, s, l, objective_fn = arg
                        if (idxs in d_results) and (typ in d_results[idxs]):
                            d_results[idxs][typ] += \
                                objective_fn.compute_cost(s, l) * \
                                p_cs.flatten()[context_id]
                        else:
                            d_results[idxs][typ] = \
                                objective_fn.compute_cost(s, l) * \
                                p_cs.flatten()[context_id]

                # Track meta-data
                d_results[idxs]['sim_id'] = sim_id
                d_results[idxs]['contains_ambiguity'] = \
                    contains_ambiguities(idxs, N_UTTERANCES, N_MEANINGS)
                d_results[idxs]['n_contexts'] = len(contexts)
                d_results[idxs]['utterance_order_fn_name'] = \
                    utterance_order_fn_name
                d_results[idxs]['meaning_order_fn_name'] = \
                    meaning_order_fn_name
                d_results[idxs]['speaker_k'] = speaker_k
                d_results[idxs]['listener_k'] = listener_k
                d_results[idxs]['speaker_alpha'] = speaker_alpha
                d_results[idxs]['listener_alpha'] = listener_alpha
                d_results[idxs]['n_utterances'] = N_UTTERANCES
                d_results[idxs]['n_meanings'] = N_MEANINGS
                d_results[idxs]['p_contexts'] = [c.tolist() for c in contexts]
                d_results[idxs]['p_utterances'] = p_utterances.tolist()

                # Non-pragmatic S and L's
                # they don't have access to the conditionals so
                # p(m|c) is just p(m) = \sum_c p(m|c)
                # non-pragmatic speaker / listener
                m_speaker_no_context = \
                    col_multiply(speaker(m, p_utterances, p_m_no_context,
                                         speaker_alpha, speaker_k),
                                 p_m_no_context)
                m_listener_no_context = \
                    row_multiply(listener(m, p_utterances,
                                          p_m_no_context, listener_alpha,
                                          listener_k), p_utterances)
                d_results[idxs]['base_speaker_listener_ce'] = \
                    ce.compute_cost(m_speaker_no_context,
                                    m_listener_no_context)

            # Track objectives
            min_ce = np.Inf
            min_kl = np.Inf
            min_ferrer = np.Inf
            min_sym_ce = np.Inf
            min_speaker_entropy = np.Inf
            min_listener_entropy = np.Inf
            min_base_ce = np.Inf
            for idxs, results in d_results.items():
                if results['ce'] < min_ce:
                    min_ce = results['ce']
                if results['kl'] < min_kl:
                    min_kl = results['kl']
                if results['ferrer'] < min_ferrer:
                    min_ferrer = results['ferrer']
                if results['sym_ce'] < min_sym_ce:
                    min_sym_ce = results['sym_ce']
                if results['speaker_entropy'] < min_speaker_entropy:
                    min_speaker_entropy = results['speaker_entropy']
                if results['listener_entropy'] < min_listener_entropy:
                    min_listener_entropy = results['listener_entropy']
                if results['base_speaker_listener_ce'] < min_base_ce:
                    min_base_ce = results['base_speaker_listener_ce']

            # Record minimums
            for idxs, results in d_results.items():
                results['is_min_ce'] = results['ce'] == min_ce
                results['is_min_kl'] = results['kl'] == min_kl
                results['is_min_ferrer'] = results['ferrer'] == min_ferrer
                results['is_min_sym_ce'] = results['sym_ce'] == min_sym_ce
                results['is_min_speaker_entropy'] = \
                    results['speaker_entropy'] == min_speaker_entropy
                results['is_min_listener_entropy'] = \
                    results['listener_entropy'] == min_listener_entropy
                results['is_min_base_speaker_listener_ce'] = \
                    results['base_speaker_listener_ce'] == min_base_ce
            all_results.append(d_results)
            pbar.update()

        pbar.close()
        return all_results


class VaryContextSimulations(Simulation):
    """Simulation focuses on benefits of pragmatic recursion.

    Evidence backing claim 1 in Peloquin, Goodman & Frank (2018)

    Each context uses the same meaning distribution, but shifts
    the probabilities.

    E.g.
    meanigs = [1, 2, 3
    p_meanings_1 = [0.2, 0.3, 0.5]
    p_meanings_2 = [0.5, 0.2, 0.3]
    ...

    """

    def run(self,
            n_sims,
            utterance_order_fn,
            utterance_order_fn_name,
            meaning_order_fn,
            meaning_order_fn_name,
            speaker_k=1,
            listener_k=1,
            speaker_alpha=1,
            listener_alpha=1,
            verbose=False,
            matrix_density=1):
        simulator = ContextSimulation()

        # Note we examine 1-4 contexts
        context_1_params = [n_sims, 1,
                            utterance_order_fn,
                            utterance_order_fn_name,
                            meaning_order_fn,
                            meaning_order_fn_name,
                            speaker_k, listener_k,
                            speaker_alpha, listener_alpha,
                            verbose, matrix_density]
        context_2_params = [n_sims, 2,
                            utterance_order_fn,
                            utterance_order_fn_name,
                            meaning_order_fn,
                            meaning_order_fn_name,
                            speaker_k, listener_k,
                            speaker_alpha, listener_alpha,
                            verbose, matrix_density]
        context_3_params = [n_sims, 3,
                            utterance_order_fn,
                            utterance_order_fn_name,
                            meaning_order_fn,
                            meaning_order_fn_name,
                            speaker_k, listener_k,
                            speaker_alpha, listener_alpha,
                            verbose, matrix_density]
        context_4_params = [n_sims, 4,
                            utterance_order_fn,
                            utterance_order_fn_name,
                            meaning_order_fn,
                            meaning_order_fn_name,
                            speaker_k, listener_k,
                            speaker_alpha, listener_alpha,
                            verbose, matrix_density]
        params = [
            context_1_params,
            context_2_params,
            context_3_params,
            context_4_params
        ]
        results = []
        pbar = tqdm.tqdm(total=len(params)*verbose)
        for param in params:
            logging.info("Running context sim with {} contexts".format(param[1]))
            d = simulator.run(*param)
            d_processed = self.process_results(d)
            results.append(pd.DataFrame(d_processed))
            pbar.update()
        pbar.close()

        return pd.concat(results)

    def process_results(self, d):
        d_results = []
        for x in d:
            for idxs, vals in x.items():
                curr_d = vals
                curr_d['idxs'] = idxs
                d_results.append(curr_d)
        return d_results


# Below is old code that needs updating and is not used in the
# paper "Pragmatic interactions lead to efficient language structure and use"
class RecursiveSimulation(Simulation):
    """Simulation focuses on benefits of pragmatic recursion.

    Evidence backing claim 3 in Peloquin, Goodman & Frank (2018)

    """

    def run(self,
            n_sims,
            utterance_order_fn,
            utterance_order_fn_name,
            meaning_order_fn,
            meaning_order_fn_name,
            speaker_k_range=range(1, 10),
            listener_k_range=range(1, 10),
            speaker_alpha=1,
            listener_alpha=1,
            verbose=False):
        simulator = ContextSimulation()

        # All combinations of s/l recursion
        all_recursive_k_combs = \
            list(it.product(speaker_k_range, listener_k_range))

        # Todo (BP): Just setting single context for now,
        # but this is likely something that we want to vary.
        n_contexts = 1

        results = []
        pbar = tqdm.tqdm(total=len(all_recursive_k_combs) * verbose)
        for (s_k, l_k) in all_recursive_k_combs:
            run_params = [n_sims, n_contexts,
                          utterance_order_fn, utterance_order_fn_name,
                          meaning_order_fn, meaning_order_fn_name,
                          s_k, l_k, speaker_alpha, listener_alpha, True]
            logging.info("Running recursive sim with "
                         "depths {}\{} ".format(s_k, l_k))
            out = simulator.run(*run_params)





