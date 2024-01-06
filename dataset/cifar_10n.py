# Code adapted from https://github.com/Docta-ai/docta/blob/master/docta/datasets/cifar.py
import os
import torch
import torchvision.transforms as transforms
import numpy as np

from torchvision.datasets import CIFAR10
from PIL import Image


def load_label(label_path, clean_label = None, key = None, clean_key = None):
    noise_label = torch.load(label_path)
    if key is None: # default key is 'noisy_label'
        key = 'noisy_label'
    if clean_key is None: # default clean key is 'clean_label'
        clean_key = 'clean_label'

    if isinstance(noise_label, dict):
        if clean_key in noise_label.keys() and clean_label is not None: # sanity check
            clean_label = noise_label['clean_label']
            assert torch.sum(torch.tensor(clean_label) - clean_label) == 0
        return noise_label[key].reshape(-1)
    else:
        return noise_label.reshape(-1)


class Cifar10_noisy(CIFAR10):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def __init__(self, cfg, train=True, ixs=None, preprocess=None) -> None:
        assert(preprocess is not None, 'Pre-process not none.')
        super(Cifar10_noisy, self).__init__(cfg.data_root, train=train, transform=preprocess,
                                            target_transform=None, download=True)
        self.cfg = cfg
        if train:
            assert ixs is not None
            self.load_label()
            self.data = self.data[ixs]
            self.label = self.label[ixs]
        else:
            self.label = np.array(self.targets)


    def load_label(self):
        self.label = load_label(label_path=self.cfg.label_path, clean_label=self.targets,
                                      key=self.cfg.noisy_label_key, clean_key=self.cfg.clean_label_key)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img, (target, noisy_label), index).
            target: clean label
            noisy_label: loaded/synthesized noisy label
        """
        img, label = self.data[index], self.label[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label


class Cifar10N(Cifar10_noisy):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def __init__(self, cfg, train=True, tr_transform=True, ixs=None, preprocess=None, noisy_label_key=None, clean_label_key=None) -> None:
        if preprocess is None:
            preprocess = self.train_transform if tr_transform else self.test_transform
        self.cfg = cfg
        self.cfg.label_path = os.path.join(cfg.data_root, 'CIFAR-10_human.pt')
        if noisy_label_key is not None:
            self.cfg.noisy_label_key = noisy_label_key
        if clean_label_key is not None:
            self.cfg.clean_label_key = clean_label_key
        super(Cifar10N, self).__init__(cfg, train, ixs, preprocess)


class Cifar10_clean(CIFAR10):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def __init__(self, cfg, train=True, preprocess=None) -> None:
        if preprocess is None:
            preprocess = self.train_transform if train else self.test_transform
        super(Cifar10_clean, self).__init__(cfg.data_root, train=train, transform=preprocess,
                                            target_transform=None, download=True)
        self.cfg = cfg
        self.label = self.targets
        self.feature = self.data
