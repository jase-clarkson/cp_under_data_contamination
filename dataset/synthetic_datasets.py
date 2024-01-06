import numpy as np

from sklearn.datasets import make_classification
from scipy.special import softmax


def generate_logistic(n, p, K, **kwargs):
    # Generate a dataset following a gaussian logistic regression model
    beta = np.random.normal(0, 1, (p, K))
    p = beta.shape[0]
    K = beta.shape[1]
    X = np.random.normal(0, 1, (n, p))
    projected = np.dot(X, beta)
    logits = softmax(projected, axis=1)
    y = np.array([np.random.choice(K, 1, p=row) for row in logits])
    return X, y.reshape(-1)


def generate_hypercube(n, p, K, sep=1.0, prop_informative=0.75, **kwargs):
    # Generate a dataset with gaussian clusters around the vertices of hypercube with side length sep
    # See https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
    return make_classification(n,
                               n_classes=K,
                               n_features=p,
                               class_sep=sep,
                               n_informative=int(p*prop_informative))