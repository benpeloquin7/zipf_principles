"""run.py

Run a single simulation specified in config.py

Example
-------

>>> python -m ambiguity.run context --out-dir sims

"""

import logging
import os

from config import SIMULATIONS

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('sim_type', type=str, default='context',
                        help='Must be one of context|recursive '
                             '[Default: context]')
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

    logging.info("Cacheing data to {}".format(fp))
    data.to_csv(fp, index=False)
    with open(os.path.join(args.out_dir, "run_settings.pickle"), "wb") as fp:
        run_params = {k: v for k, v in params["run_params"].items() if
                      not callable(v)}
        env_params = {k: v for k, v in params.items() \
                      if not callable(v) and k != "run_params"}
        x = run_params.copy()
        x.update(env_params)
        pickle.dump(x, fp)
