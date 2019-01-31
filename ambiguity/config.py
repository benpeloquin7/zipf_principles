"""config.py"""

from simulations import VaryContextSimulations, RecursiveSimulation

CONTEXT_RUN_PARAMS = {
    'simulator': VaryContextSimulations,
    'run_params': {
        'n_sims': 500,
        'utterance_order_fn': lambda x: 1.,
        'utterance_order_fn_name': 'uniform',
        'meaning_order_fn': lambda x: 1.,
        'meaning_order_fn_name': 'uniform',
        'speaker_alpha': 1.,
        'listener_alpha': 1.,
        'speaker_k': 1,
        'listener_k': 1,
        'verbose': False
    },
    # fp pattern below
    # ----------------
    # utterance_order_fn_name-meaning_order_fn_name-speaker_k-listener_k
    'out_file_name': 'uniform_uniform_1_1_500.csv'
}


RECURSIVE_RUN_PARAMS = {
    'simulator': RecursiveSimulation,
    'run_params': {
        'n_sims': 10,
        'n_contexts': 1,
        'utterance_order_fn': lambda x: 1.,
        'utterance_order_fn_name': 'uniform',
        'meaning_order_fn': lambda x: 1.,
        'meaning_order_fn_name': 'uniform',
        'ks': list(range(30)),
        'verbose': False
    },
    # utterance_order_fn_name-meaning_order_fn_name-speaker_k-listener_k
    'out_file_name': 'recursive_10.csv'
}

SIMULATIONS = {
    'recursive': RECURSIVE_RUN_PARAMS,
    'context': CONTEXT_RUN_PARAMS
}