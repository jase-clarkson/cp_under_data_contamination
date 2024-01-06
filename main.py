import argparse
import os
import pandas as pd

from sklearn.model_selection import ParameterGrid
from itertools import product

from config_utils import get_config
from run_experiment import run_experiment
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config_file',
        type=str,
        help="Name of config file within the experiment_configs folder",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    config_file = args.config_file
    config = get_config(os.path.join('experiment_configs', config_file))
    ds_args, cp_args, corr_args, models = (config.pop(s) for s in ['dataset', 'cp', 'corruption', 'model'])

    ds_name = ds_args.dataset_name
    ds_grid = ParameterGrid(ds_args['params'])
    cp_grid = ParameterGrid(cp_args)
    corr_grid = ParameterGrid(corr_args)
    models_grid = ParameterGrid(models)

    ix_cols = []
    for dict in (ds_grid[0], cp_grid[0], corr_grid[0], models_grid[0]):
        ix_cols += [k for k, _ in sorted(dict.items(), key=lambda x: x[0])]

    ix_values = []
    results = []
    exp_args = config
    exp_args['dataset_name'] = ds_name

    for ds_params, cp_params, corr_params, model_params in product(ds_grid, cp_grid, corr_grid, models_grid):
        ix = []
        for dict in (ds_params, cp_params, corr_params, model_params):
            ix += [v for _, v in sorted(dict.items(), key=lambda x: x[0])]
        ix_values.append(tuple(ix))
        reps = []
        for seed in tqdm(range(config.seed_base, config.seed_base + config.n_reps)):
            res = run_experiment(seed, exp_args, ds_params, cp_params, corr_params, model_params)
            reps.append(res)
        exp_frame = pd.concat(reps, axis=0)
        exp_frame = (pd.concat([exp_frame.mean(axis=0), exp_frame.std(axis=0)], axis=1).
                     rename({0: 'mean', 1: 'std'}, axis=1)).stack()
        results.append(exp_frame)
    results = pd.concat(results, axis=1).T
    results.index = pd.MultiIndex.from_tuples(ix_values, names=ix_cols)
    print(results)
    results.to_pickle(os.path.join(config.save_dir, 'results.pkl'))


if __name__ == '__main__':
    main()