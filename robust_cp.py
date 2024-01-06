import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from aps import ProbabilityAccumulator


class ScoreECDF:
    # Class to compute the ECDFs and scores as described in Section 4.4 of the paper
    def __init__(self,
                 y: np.array,
                 K: int,
                 calibrator: ProbabilityAccumulator,
                 U: np.array=None):

        self.ecdfs = [[None] * K for _ in range(K)]
        self.scores = [[None] * K for _ in range(K)]

        for j in range(K):
            # Condition on \tilde{Y} = j.
            idx = np.where(y == j)[0]
            if len(idx) > 0:
                # Compute score for each possible label i at these points and compute ECDF.
                for i in range(K):
                    dummy = i * np.ones_like(y)
                    # NOTE: The APS code computes the scores the 'other' way around to what you
                    # expect. So you need to put 1 - S to get back to normal.
                    self.scores[i][j] = 1.0 - calibrator.calibrate_scores(dummy, U)[idx]
                    self.ecdfs[i][j] = ECDF(self.scores[i][j])

    def compute(self, q: np.array, i: int, j: int) -> np.array:
        if self.ecdfs[i][j] is None:
            print(f'Warning: no ECDF for i={i}, j={j}')
        return self.ecdfs[i][j](q)


def g_hat(
        q: np.array,
        fhat: ScoreECDF,
        P_inv: np.array,
        p: np.array,
        p_tilde: np.array) -> np.array:
    # Compute the plug in estimator of the bias g() as in Equation (31) of the paper.
    K = P_inv.shape[0]
    ret = np.zeros_like(q)
    for i in range(K):
        for j in range(K):
            ret += p[i] * P_inv[j, i] * fhat.compute(q, i, j)
        ret -= p_tilde[i] * fhat.compute(q, i, i)
    return ret


def compute_b(n: int, p_tilde: np.array) -> np.array:
    # Compute the coefficients b(n, i) as described in Theorem 5.1 of the paper
    # (This function vectorizes b across i)
    return np.sqrt(np.pi / (n * p_tilde)) + (1 - p_tilde)**n


def compute_C(n: int, P_inv: np.array, p: np.array, p_tilde: np.array) -> float:
    # Compute the upper bound to E_q[\hat{g} - g(q)] as described in Theorem 5.1
    K = P_inv.shape[0]
    b = compute_b(n, p_tilde)
    ret = 0.0
    for i in range(K):
        ret += np.abs(P_inv[i, i]*p[i] - p_tilde[i]) * b[i]
        for j in range(K):
            if j != i:
                ret += np.abs(P_inv[j, i] * p[i]) * b[j]
    return ret


def compute_adjusted_i(alpha: float,
                       sorted_scores: np.array,
                       fhat: ScoreECDF,
                       P_inv: np.array,
                       p: np.array,
                       p_tilde: np.array) -> int:
    # Compute the adjusted index i as described in Equation (32) of the paper.

    n = sorted_scores.shape[0]
    # Compute the upper bound C
    C = compute_C(n, P_inv, p, p_tilde)
    # Compute plug in estimator of bias g(S_{(i)}) for each i
    ghats = g_hat(sorted_scores, fhat, P_inv, p, p_tilde)
    # Find the adjusted value of i
    ivals = (1 - alpha) - ghats + C - (np.arange(1, n + 1) / (n+1))
    candidates = np.where(ivals <= 0)[0]
    i = np.min(candidates) if candidates.size > 0 else int(np.ceil((n + 1) * (1 - alpha)))
    if candidates.size == 0: print('Warning: no valid found, falling back to standard CP.')
    return i
