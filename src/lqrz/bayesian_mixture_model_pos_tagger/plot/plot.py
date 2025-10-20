"""Plotting functions for visualizing Gibbs sampler output."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
from collections import Counter

from lqrz.bayesian_mixture_model_pos_tagger.train.gibbs_sampler import GibbsSampler

_ = sns.set_style("whitegrid")


__all__ = [
    "plot_log_joint_trace",
    "plot_class_counts_trace",
    "plot_word_type_posterior",
    "plot_word_type_posterior_by_frequency",
]


def plot_log_joint_trace(log_probs_trace: List[float], figsize: Tuple[int, int] = (4, 3)) -> None:
    """Plot log-joint trace."""
    # asserts
    assert isinstance(log_probs_trace, List)
    assert all(isinstance(x, float) for x in log_probs_trace)

    df_plot = pd.DataFrame({"iteration": np.arange(len(log_probs_trace)), "log_prob": log_probs_trace})
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    _ = sns.lineplot(data=df_plot, x="iteration", y="log_prob", color="C0", ax=ax)
    _ = ax.set_xlabel("Iteration")
    _ = ax.set_ylabel("Collapsed log joint")
    _ = ax.set_title("Log-joint trace (convergence check)")
    _ = f.tight_layout()


def plot_class_counts_trace(class_counts_trace: List[np.ndarray], figsize: Tuple[int, int] = (4, 3)) -> None:
    """Plot class count trace."""
    # asserts
    assert isinstance(class_counts_trace, List)

    x_class_counts_trace = np.stack(class_counts_trace, axis=0)  # shape (T, n_classes)

    df_plot = (
        pd.DataFrame(x_class_counts_trace, columns=[f"class_{z}" for z in range(x_class_counts_trace.shape[1])])
        .assign(iteration=np.arange(len(x_class_counts_trace)))
        .melt(id_vars="iteration", var_name="class", value_name="count")
    )
    f, ax = plt.subplots(1, 1, figsize=figsize)
    _ = sns.lineplot(data=df_plot, x="iteration", y="count", hue="class", palette="husl", linewidth=1, ax=ax)
    _ = ax.set_xlabel("Iteration")
    _ = ax.set_ylabel("Cnt word types in class")
    _ = ax.set_title("Class size trace")
    _ = f.tight_layout()


def plot_word_type_posterior(
    sampler: GibbsSampler,
    word: str,
    wordtype_index: Dict[str, int],
    ax: plt.Axes,
    title: str,
    figsize: Tuple[int, int] = (4, 3),
) -> None:
    """Plot word type posterior probabilities."""
    # asserts
    assert isinstance(sampler, GibbsSampler)
    assert isinstance(word, str)
    assert isinstance(wordtype_index, Dict)

    posterior_probs: np.ndarray = sampler.compute_posterior_class_probs(wordtype_index=wordtype_index[word])

    df_plot = pd.DataFrame({"class": np.arange(sampler._n_classes).astype(int), "probability": posterior_probs})

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)
    _ = sns.barplot(data=df_plot, x="class", y="probability", ax=ax)
    title: str = f"Posterior class probabilities for word '{ix_to_wordtype[word_ix]}'" if title is None else title
    _ = ax.set_title(title)
    _ = ax.set_xlabel("")
    _ = ax.set_ylabel("")
    _ = ax.set_ylim(0, 1)


def plot_word_type_posterior_by_frequency(
    sampler: GibbsSampler,
    counter: Counter,
    wordtype_index: Dict[str, int],
    n_most_common: int = 10,
    figsize: Tuple[int, int] = (13, 2),
):
    """Plot."""
    # asserts
    assert isinstance(sampler, GibbsSampler)
    assert isinstance(n_most_common, int)
    assert n_most_common > 0
    assert isinstance(counter, Counter)
    assert isinstance(wordtype_index, Dict)

    title: str = "{FREQ} frequent words class"
    # most freq
    words: List[str] = [x for x, _ in counter.most_common(n_most_common)]
    f, axs = plt.subplots(1, n_most_common, figsize=figsize, sharex=True, sharey=True)
    for w, ax in zip(words, axs):
        _ = plot_word_type_posterior(sampler=sampler, word=w, wordtype_index=wordtype_index, ax=ax, title=w)
        _ = f.suptitle(title.format(FREQ="Most"))
        _ = f.tight_layout()

    # least freq
    words: List[str] = [x for x, _ in counter.most_common()[::-1][:n_most_common]]
    f, axs = plt.subplots(1, n_most_common, figsize=figsize, sharex=True, sharey=True)
    for w, ax in zip(words, axs):
        _ = plot_word_type_posterior(sampler=sampler, word=w, wordtype_index=wordtype_index, ax=ax, title=w)
        _ = f.suptitle(title.format(FREQ="Least"))
        _ = f.tight_layout()
