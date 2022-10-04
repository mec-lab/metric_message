import os

from functools import reduce

def load_benchmark(filepath="data/nguyen.csv"):

    if filepath is None:
        filepath="data/nguyen_test.csv"

    with open(filepath, "r") as f:
        lines = f.readlines()


    return lines


def r2_over_threshold(r2, thesholds=None):

    if thresholds is None:
        thresh = np.arange(0,1.0,0.01)

    r2_ot = [(x > threshold).sum() / x.shape[0] for threshold in thresh ]

    return r2_ot

def r2_auc(roc_curve):

    raw_result = reduce(lambda a,b: a+b, roc_curve)

    return raw_result / len(roc_curve)

