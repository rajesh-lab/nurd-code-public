import numpy as np
import torch

# Developer: Alejandro Debus
# Email: aledebus@gmail.com

def partitions(number, k):
    '''
    Distribution of the folds

    Args:
        number: number of patients
        k: folds count
    '''
    n_partitions = np.ones(k) * int(number/k)
    n_partitions[0:(number % k)] += 1
    return n_partitions

def get_indices(n_splits = 3, subjects = 145):
    '''
    Indices of the set val

    Args:
        n_splits: folds number
        subjects: number of patients
        frames: length of the sequence of each patient
    '''
    l = partitions(subjects, n_splits)
    fold_sizes = l
    indices = np.arange(subjects).astype(int)
    np.random.shuffle(indices)
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop =  current + fold_size
        current = stop
        yield(indices[int(start):int(stop)])

def k_folds(n_splits, subjects):
    '''
    Generates folds for cross validation

    Args:
        n_splits: folds number
        subjects: number of patients
    '''
    indices = np.arange(subjects).astype(int)
    for test_idx in get_indices(n_splits, subjects):
        train_idx_pre = np.setdiff1d(indices, test_idx)
        val_idx = np.random.choice(train_idx_pre, size=len(test_idx)//2, replace=False)
        train_idx = np.setdiff1d(train_idx_pre, val_idx)
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        assert len(np.intersect1d(val_idx, test_idx)) == 0
        assert len(np.intersect1d(val_idx, train_idx)) == 0
        yield torch.from_numpy(train_idx).view(-1), torch.from_numpy(val_idx).view(-1), torch.from_numpy(test_idx).view(-1)