"""run.py

Run a single simulation specified in config.py

Example
-------

>>> python -m ambiguity.run --sim-type context --out-dir

"""

import os

from config import SIMULATIONS

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sim-type', type=str, default='context',
                        help='Must be one of context|recursive [Default: context]')
    parser.add_argument('--out-dir', type=str, default='sims',
                        help='Output directory [Default: sims/]')

    args = parser.parse_args()

    params = SIMULATIONS[args.sim_type]
    simulator = params['simulator']()
    run_params = params['run_params']
    data = simulator.run(**run_params)
    fp = os.path.join(args.out_dir, params['out_file_name'])
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    data.to_csv(fp, index=False)
