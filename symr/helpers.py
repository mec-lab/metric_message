import os

from functools import reduce

import numpy as np

import matplotlib.pyplot as plt

def load_benchmark(filepath=None):

    if filepath is None:
        filepath="data/nguyen_test.csv"

    with open(filepath, "r") as f:
        lines = f.readlines()


    return lines


def r2_over_threshold(r2, thresholds=None):

    if thresholds is None:
        thresholds = np.arange(0,1.0,0.01)

    r2_ot = [(r2 > threshold).sum() / r2.shape[0] for threshold in thresholds ]

    return r2_ot

def r2_auc(roc_curve):

    raw_result = reduce(lambda a,b: a+b, roc_curve)

    return raw_result / len(roc_curve)

def plot_r2_over_threshold(in_r2_ot, ex_r2_ot=None, title=None):

    my_cmap = plt.get_cmap("magma")
    colors = [my_cmap(192), my_cmap(32)]

    fig, ax = plt.subplots(1,1, figsize=(5,5))

    in_r2_ot.append(0.0)
    in_baseline = [0*elem for elem in in_r2_ot]
    ax.plot(in_r2_ot, lw=4, alpha=0.35, color=colors[0], label="in-$r^2$")
    ax.fill_between(np.arange(0,len(in_r2_ot),1), \
            in_baseline, in_r2_ot, color=colors[0], alpha=0.05)

    if ex_r2_ot is not None:

        ex_r2_ot.append(0.0)

        ex_baseline = [0*elem for elem in ex_r2_ot]
        ax.plot(ex_r2_ot, "-.", lw=4, alpha=0.35, color=colors[1], label="ex-$r^2$")
        ax.fill_between(np.arange(0,len(ex_r2_ot),1), \
                ex_baseline, ex_r2_ot, color=colors[1], alpha=0.05)


    #ax.plot([0, len(in_r2_ot)], [1.0, 0.0], "--", lw=4, alpha=0.125, label="y = x")

    if title is None:
        "R^2 over threshold"

    ax.set_title(title, fontsize=16)

    xticks = [0, len(in_r2_ot) / 2, len(in_r2_ot)]

    xtick_labels = [0, 0.5, 1.0]

    ax.set_xticks(xticks)

    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(xtick_labels)

    #ax.set_yticklabels(xticks, xticks)
    plt.legend()
    return fig, ax


