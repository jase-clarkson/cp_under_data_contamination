import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from resnet18 import resnet18
from dataset.cifar_10n import Cifar10N
from types import SimpleNamespace


def train_and_output_logits(noise_type, args):
    exp_dir = os.path.join(args['models_dir'], noise_type)
    os.makedirs(exp_dir, exist_ok=True)

    noisy_labels_path = os.path.join(args['data_dir'], 'CIFAR-10_human.npy')
    labels = np.load(noisy_labels_path, allow_pickle=True)
    random_label1 = labels.item().get(noise_type)
    noisy_labels = pd.DataFrame(random_label1, columns=['label'])
    # 20k train, 10k val, 20k calib
    train = noisy_labels.groupby('label').apply(lambda x: x.iloc[:2000])
    train.index = train.index.droplevel()
    train = train.sort_index()
    val = noisy_labels.groupby('label').apply(lambda x: x.iloc[2000:3000])
    val.index = val.index.droplevel()
    val = val.sort_index()
    calib = noisy_labels.drop(np.concatenate([train.index.values, val.index.values])).sort_index()
    train = train.index.values
    val = val.index.values
    cal = calib.index.values

    train_batch_size = args['train_batch_size']
    test_batch_size = args['test_batch_size']
    K = 10

    # I adapted this from https://github.com/Docta-ai/docta/blob/master/docta/datasets/cifar.py
    # This is for compatability reasons, could be re-written to not need this cfg in a future
    # version
    cfg = SimpleNamespace()
    cfg.data_root = args['data_dir']
    cfg.seed = args['seed']
    cfg.noisy_label_key = noise_type
    cfg.clean_label_key = 'clean_label'

    train_ds = Cifar10N(cfg, train=True, ixs=train)
    val_ds = Cifar10N(cfg, train=True, ixs=val, tr_transform=False)
    cal_ds = Cifar10N(cfg, train=True, ixs=cal, tr_transform=False)
    test_ds = Cifar10N(cfg, train=False, tr_transform=False)

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, num_workers=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=test_batch_size, num_workers=8, shuffle=False)
    cal_loader = DataLoader(cal_ds, batch_size=test_batch_size, num_workers=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, num_workers=8, shuffle=False)

    # Save labels for CP evaluation
    calib.to_csv(os.path.join(exp_dir, 'calib.csv'))
    test_labels = pd.Series(test_ds.label)
    test_labels.name = 'label'
    test_labels.to_csv(os.path.join(exp_dir, 'test.csv'))


    model = resnet18(K, ft=False)
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=exp_dir,
                                              filename='{epoch}-{val_acc:.2f}',
                                              monitor='val_acc',
                                              mode='max',
                                              save_top_k=1)
    trainer = pl.Trainer(max_epochs=30, callbacks=[checkpoint])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    model = resnet18.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, K=K)
    save_path = exp_dir

    for file, loader in [('calib.csv', cal_loader), ('test.csv', test_loader)]:
        preds = trainer.predict(model, loader)

        preds = torch.cat(preds).numpy()
        preds = pd.DataFrame(preds)
        preds.to_csv(os.path.join(save_path, f'logits_{file}'), index=False)


if __name__ == '__main__':
    args = {
        'data_dir': 'data/CIFAR10N',
        'models_dir': 'models',
        'seed': 123,
        'train_batch_size': 128,
        'test_batch_size': 256,
    }

    os.makedirs(args['data_dir'], exist_ok=True)
    os.makedirs(args['models_dir'], exist_ok=True)

    noise_types = ['clean_label', 'worse_label', 'aggre_label'] + [f'random_label{i}' for i in range(1, 4)]
    for noise_type in noise_types:
        train_and_output_logits(noise_type, args)



