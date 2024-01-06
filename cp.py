import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List, Dict, Tuple

class CorruptedDistribution:
    def __init__(self, distns: List, mixture_params: List[Dict], mixture_weights: List[float], clean_idx: int = 0):
        assert len(distns) == len(mixture_weights) == len(
            mixture_params), 'Ensure number of mixture weights, params and distribution functions are the same.'
        assert sum(mixture_weights) == 1
        self.distns = distns
        self.mixture_params = mixture_params
        self.mixture_weights = mixture_weights
        self.clean_idx = clean_idx

    def sample(self, n: int, clean: bool = False) -> np.ndarray:
        if clean:
            return self.distns[self.clean_idx](**self.mixture_params[self.clean_idx], size=n)
        else:
            components = np.random.choice([i for i in range(len(self.mixture_params))], size=n, p=self.mixture_weights)
            out = []
            for c in components:
                params = self.mixture_params[c]
                out.append(self.distns[c](**params))
            return np.array(out)


def compute_coverage(sets: List[List[float]], y: list[float]) -> float:
    assert len(sets) == len(y)
    covered = 0.0
    for i in range(len(sets)):
        if sets[i][0] <= y[i] <= sets[i][1]:
            covered += 1
    return covered / float(len(sets))


def params_to_str(params: Dict) -> str:
    return '|'.join([f'{str(k)}={str(v)}' for k, v in params.items()])


def generate_regression_dataset(beta: np.ndarray,
                                n: int,
                                eps_dist: CorruptedDistribution,
                                clean: bool) -> Tuple[np.ndarray, np.ndarray]:
    p = beta.shape[0]
    X = np.random.normal(0, 1, (n, p))
    eps = eps_dist.sample(n, clean)
    y = X @ beta + eps
    return X, y


def estimate_coverage_and_width(clean_test: bool,
                                name: str,
                                alpha: float,
                                beta: np.ndarray,
                                eps_dist: CorruptedDistribution,
                                n: int) -> Dict:
    # Generate training data and fit regression
    X_tr, y_tr = generate_regression_dataset(beta, n, eps_dist, True)
    regr = LinearRegression().fit(X_tr, y_tr)

    # Generate calibration data
    X_cal, y_cal = generate_regression_dataset(beta, n, eps_dist, False)

    # Get the residuals on the calibration data.
    preds = np.array(regr.predict(X_cal))
    r = np.abs(y_cal - preds)

    # Estimate conformal quantile
    l = int(np.ceil((1 - alpha) * (n + 1)))
    q = np.sort(r)[l]

    # Construct conformal prediction sets and examine coverage
    # name = 'conditional' if clean_test else 'mixture'
    output = {}
    X_test, y_test = generate_regression_dataset(beta, n, eps_dist, clean_test)
    test_preds = regr.predict(X_test)
    pred_sets = [(f - q, f + q) for f in test_preds]
    cvg = compute_coverage(pred_sets, y_test)
    # name = params_to_str(params)
    output['cvg'] = cvg
    output['width'] = 2 * q
    return output
