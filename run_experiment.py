import numpy as np
import pandas as pd
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from label_noise import sample_noisy_labels, make_random_flip
from robust_cp import compute_adjusted_i, ScoreECDF
from aps import ProbabilityAccumulator, make_aps_sets
from dataset.synthetic_datasets import generate_logistic, generate_hypercube

_synthetic_datasets = ['logistic', 'hypercube']
_real_datasets = ['CIFAR10N']


def get_dataset(name, n, **kwargs):
    if name == 'logistic':
        return generate_logistic(n, **kwargs)
    elif name == 'hypercube':
        return generate_hypercube(n, **kwargs)
    else:
        raise NotImplementedError(f'Unknown dataset: {name}')


def get_noise_matrix(K, noise_type, eps):
    if noise_type == 'random_flip':
        return make_random_flip(eps, K)
    else:
        raise NotImplementedError(f'Unknown noise type: {noise_type}')


def get_model(name, **kwargs):
    if name == 'log_reg':
        return LogisticRegression
    elif name == 'rf':
        return RandomForestClassifier
    elif name == 'mlp':
        return MLPClassifier
    elif name == 'gbt':
        return HistGradientBoostingClassifier
    else:
        raise NotImplementedError(f'Unknown model type: {name}')

def run_experiment(seed,
                   exp_params,
                   ds_params,
                   cp_params,
                   corr_params,
                   model_params):

    np.random.seed(seed)
    alpha = cp_params['alpha']
    dataset = exp_params['dataset_name']

    if dataset in _synthetic_datasets:
        # Setup some parameters that are used a lot.
        n_train, n_cal, n_test = ds_params['n_train'], ds_params['n_cal'], ds_params['n_test']
        n = n_train + n_cal + n_test
        K = ds_params['K']
        eps = corr_params['eps']

        X, y = get_dataset(dataset, n, **ds_params)

        P = get_noise_matrix(K, corr_params['noise_type'], eps)
        # TODO: estimate if we have a synthetic model without uniform.
        p = p_tilde = np.ones(K) / K

        X_tc, X_test, y_tc, y_test = train_test_split(X, y, test_size=n_test)
        y_tc = sample_noisy_labels(y_tc, P, p, p_tilde)
        X_tr, X_cal, y_tr, y_cal = train_test_split(X_tc, y_tc, test_size=n_cal)

        # TODO: allow model args
        model_type = get_model(model_params['model_name'])
        model = model_type().fit(X_tr, y_tr)

        cal_probs = model.predict_proba(X_cal)
        test_probs = model.predict_proba(X_test)

    else:
        n_cal = ds_params['n_cal']
        n_test = ds_params['n_test']

        estimates_dir = os.path.join(exp_params['corruption_dir'], corr_params['noise_type'])
        P = pd.read_csv(os.path.join(estimates_dir, 'P.csv'), index_col=0).values
        p = pd.read_csv(os.path.join(estimates_dir, 'p.csv'), index_col=0).values.flatten()
        p_tilde = pd.read_csv(os.path.join(estimates_dir, 'p_tilde.csv'), index_col=0).values.flatten()

        K = P.shape[0]

        trained_model_dir = os.path.join(exp_params['models_dir'], corr_params['noise_type'])
        cal_probs_all = pd.read_csv(os.path.join(trained_model_dir, 'logits_calib.csv')).values
        test_probs_all = pd.read_csv(os.path.join(trained_model_dir, 'logits_test.csv')).values

        y_cal_all = pd.read_csv(os.path.join(trained_model_dir, 'calib.csv'))['label'].values
        y_test_all = pd.read_csv(os.path.join(trained_model_dir, 'test.csv'))['label'].values

        cal_ixs = np.random.choice(len(cal_probs_all), replace=False, size=n_cal)
        cal_probs, y_cal = cal_probs_all[cal_ixs], y_cal_all[cal_ixs]

        test_ixs = np.random.choice(len(test_probs_all), replace=False, size=n_test)
        test_probs, y_test = test_probs_all[test_ixs], y_test_all[test_ixs]

    i = int(np.ceil((n_cal + 1) * (1 - alpha)))

    calibrator = ProbabilityAccumulator(cal_probs)
    U = np.random.uniform(low=0.0, high=1.0, size=n_cal)
    fhat = ScoreECDF(y_cal, K, calibrator, U)
    scores = np.concatenate([fhat.scores[i][i] for i in range(K)])
    score_ord_stats = np.sort(scores)
    i_adjusted = compute_adjusted_i(alpha,
                                    score_ord_stats,
                                    fhat,
                                    np.linalg.inv(P), p, p_tilde)

    results = {}
    for name, ix in [('cp', i), ('cp_adjusted', i_adjusted)]:
        qhat = score_ord_stats[ix]
        prediction_sets = make_aps_sets(qhat, test_probs)
        cvg = np.mean([y_test[i] in prediction_sets[i] for i in range(len(y_test))])
        size = np.mean([len(set) for set in prediction_sets])
        results[name] = {'cvg': cvg, 'size': size}

    res = pd.DataFrame.from_dict(results, orient='index').stack().to_frame().T
    res.index = [seed]
    return res
