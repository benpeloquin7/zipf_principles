"""config.py"""

from .simulations import VaryContextSimulations, RecursiveSimulation

CONTEXT_RUN_PARAMS = {
    'simulator': VaryContextSimulations,
    'run_params': {
        'n_sims': 30,
        'utterance_order_fn': lambda x: 1.,
        'utterance_order_fn_name': 'uniform',
        'meaning_order_fn': lambda x: 1.,
        'meaning_order_fn_name': 'uniform',
        'speaker_alpha': 1.,
        'listener_alpha': 1.,
        'speaker_k': 1,
        'listener_k': 1,
        'verbose': False,
        'matrix_density': 2
    },
    # fp pattern below
    # ----------------
    # utterance_order_fn_name-meaning_order_fn_name-speaker_k-listener_k
    'out_file_name': 'uniform_uniform_1_1_500.csv'
}


RECURSIVE_RUN_PARAMS = {
    'simulator': RecursiveSimulation,
    'run_params': {
        'n_sims': 1,
        'utterance_order_fn': lambda x: 1.,
        'utterance_order_fn_name': 'uniform',
        'meaning_order_fn': lambda x: 1.,
        'meaning_order_fn_name': 'uniform',
        'speaker_k_range': list(range(1,3)),
        'listener_k_range': list(range(1,3)),
        'speaker_alpha': 1,
        'listener_alpha': 1,
        'verbose': False
    },
    # utterance_order_fn_name-meaning_order_fn_name-speaker_k-listener_k
    'out_file_name': 'recursive_10.csv'
}

SIMULATIONS = {
    'recursive': RECURSIVE_RUN_PARAMS,
    'context': CONTEXT_RUN_PARAMS
}