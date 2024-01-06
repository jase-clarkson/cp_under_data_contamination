import os
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

noise_types = ['clean_label', 'worse_label', 'aggre_label'] + [f'random_label{i}' for i in range(1, 4)]

if __name__ == '__main__':
    labels = np.load('CIFAR-10_human.npy', allow_pickle=True)
    clean_labels = pd.Series(labels.item().get('clean_label'))
    K = 10

    for noise_type in noise_types:
        out_dir = os.path.join('estimates', noise_type)
        os.makedirs(out_dir, exist_ok=True)

        noisy_labels = pd.Series(labels.item().get(noise_type))
        p = pd.Series(np.ones(K) / K)
        p_tilde = noisy_labels.value_counts(normalize=True).sort_index()

        P = confusion_matrix(clean_labels.values, noisy_labels.values, normalize='pred')

        pd.DataFrame(P).to_csv(os.path.join(out_dir, 'P.csv'))
        p.sort_index().to_csv(os.path.join(out_dir, 'p.csv'))
        p_tilde.sort_index().to_csv(os.path.join(out_dir, 'p_tilde.csv'))


