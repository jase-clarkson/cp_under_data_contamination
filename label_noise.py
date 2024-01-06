import numpy as np


def sample_noisy_labels(y: np.array,
                        P: np.array,
                        p_marg: np.array,
                        p_tilde_marg: np.array) -> np.array:
    '''
    Samples the noisy labels \tilde{Y} from clean labels using the mixing matrix P by implementing \tilde{P}_{ij} = P_ji \tilde{P}_i / P_j
    '''
    K = P.shape[0]
    y_tilde = -1 * np.ones_like(y)
    for j in range(K):
        idx = np.where(y == j)[0]
        if len(idx) > 0:
            probs = P[j, :] * p_tilde_marg / p_marg[j]
            # Had to add this as sometimes had fp error so didn't quite sum to 1.
            probs = probs / np.sum(probs)
            y_tilde[idx] = np.random.choice(K, len(idx), p=probs)
    assert (y_tilde >= 0).all()
    return y_tilde


def make_random_flip(eps: float, K: int):
    return (1-eps) * np.identity(K) + (eps/K) * np.ones((K,K))
