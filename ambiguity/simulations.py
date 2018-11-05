"""simulations.py"""

from collections import defaultdict
import pandas as pd
import numpy as np
import tqdm

from .matrix import *
from .objectives import *
from .utils import *


class Simulation:
    def __init__(self):
        pass

    def run(self, n_sims):
        NotImplementedError()


class SimpleSimulation(Simulation):

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

        m_listener = col_multiply(listener(m, p_utterances, p_meanings, alpha=1., k=0), p_utterances)
        m_speaker= row_multiply(speaker(m, p_utterances, p_meanings, alpha=1., k=1), p_meanings)
        return \
            {
                'CE': ce.compute_cost(m_speaker, m_listener),
                'KL': kl.compute_cost(m_speaker, m_listener),
                'ferrer': ferrer.compute_cost(m_speaker, m_listener),
                'symmetric_ce': symmetric_ce.compute_cost(m_speaker, m_listener)
            }


class OrderedSimulation(Simulation):
    def run(self, n_sims, n_contexts, order_fn, order_fn_name, verbose=False):
        all_results = []
        pbar = tqdm.tqdm(total=n_sims*verbose)
        for sim_id in range(n_sims):
            # Costs
            ce = CrossEntropy()
            kl = KL()
            ferrer = FerrerObjective()
            sym_ce = SymmetricCrossEntropy()
            # Set-up
            N_UTTERANCES = 5
            N_MEANINGS = 5
            utterances = ['a', 'b', 'c', 'd']
            meanings = [1, 2, 3, 4]
            p_utterances = np.random.dirichlet([order_fn(i) for i in range(N_UTTERANCES)])
            p_meanings = np.random.dirichlet([order_fn(i) for i in range(N_MEANINGS)])
            # Populate contexts
            contexts = [p_meanings]
            for i in range(1, n_contexts):
                contexts.append(np.roll(contexts[0], i+1))

            all_matrices = \
                [(idxs, m) for (idxs, m) in generate_all_boolean_matrices(N_UTTERANCES, N_MEANINGS, N_MEANINGS) if all_meanings_available_filter(m)]

            d_results = defaultdict(dict)
            for idxs, m in all_matrices:
                for context_id, context in enumerate(contexts):
                    p_meanings_ = context
                    m_speaker = col_multiply(speaker(m, p_utterances, p_meanings_, 1., 1), p_meanings_)
                    m_listener = row_multiply(listener(m, p_utterances, p_meanings_, 1., 0), p_utterances)
                    if idxs in d_results:
                        d_results[idxs]['ce'] += ce.compute_cost(m_speaker, m_listener)
                        d_results[idxs]['kl'] += kl.compute_cost(m_speaker, m_listener)
                        d_results[idxs]['ferrer'] += ferrer.compute_cost(m_speaker, m_listener)
                        d_results[idxs]['sum_ce'] += sym_ce.compute_cost(m_speaker, m_listener)
                    else:
                        d_results[idxs]['ce'] = ce.compute_cost(m_speaker, m_listener)
                        d_results[idxs]['kl'] = kl.compute_cost(m_speaker, m_listener)
                        d_results[idxs]['ferrer'] = ferrer.compute_cost(m_speaker, m_listener)
                        d_results[idxs]['sum_ce'] = sym_ce.compute_cost(m_speaker, m_listener)
                        d_results[idxs]['sim_id'] = sim_id
                        d_results[idxs]['contains_ambiguity'] = contains_ambiguities(idxs, N_UTTERANCES, N_MEANINGS)
                        d_results[idxs]['type'] = order_fn_name
                        d_results[idxs]['n_contexts'] = len(contexts)
                        d_results[idxs]['order_fn_name'] = order_fn_name

            min_ce = np.Inf
            for idxs, results in d_results.items():
                if results['ce'] < min_ce:
                    min_ce = results['ce']
            for idxs, results in d_results.items():
                results['is_min'] = True if results['ce'] == min_ce else False
            all_results.append(d_results)
            pbar.update()
        pbar.close()
        return all_results


class VaryContextSimulations(Simulation):
    """Run simulations varying number of contets.

    Each context uses the same meaning distribution, but shifts
    the probabilies.

    E.g.
    meanigs = [1, 2, 3
    p_meanings_1 = [0.2, 0.3, 0.5]
    p_meanings+2 = [0.5, 0.2, 0.3]

    """
    def run(self, n_sims, order_fn, order_fn_name, verbose=False):
        simulator = OrderedSimulation()
        context_1_params = [n_sims, 1, order_fn, order_fn_name]
        context_2_params = [n_sims, 2, order_fn, order_fn_name]
        context_3_params = [n_sims, 3, order_fn, order_fn_name]
        context_4_params = [n_sims, 4, order_fn, order_fn_name]
        params = [('context_1', context_1_params), ('context_2', context_2_params), ('context_3', context_3_params), ('context_4', context_4_params)]
        results = []
        pbar = tqdm.tqdm(total=len(params)*verbose)
        for param_type, param in params:
            d = simulator.run(*param)
            d_processed = self.process_results(d, param_type)
            results.append(pd.DataFrame(d_processed))
            pbar.update()
        pbar.close()

        return pd.concat(results)

    def process_results(self, d, type_name):
        d_results = []
        for x in d:
            for idxs, vals in x.items():
                curr_d = vals
                curr_d['idxs'] = idxs
                curr_d['type'] = type_name
                d_results.append(curr_d)
        return d_results




































































