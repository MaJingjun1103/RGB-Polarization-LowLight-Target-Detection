import numpy as np

def minmax_norm(x, eps=1e-8):
    x = x.astype(np.float32)
    return (x - np.min(x)) / (np.max(x) - np.min(x) + eps)


def zscore_norm(x, eps=1e-8):
    x = x.astype(np.float32)
    return (x - np.mean(x)) / (np.std(x) + eps)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / (np.sum(e) + 1e-8)