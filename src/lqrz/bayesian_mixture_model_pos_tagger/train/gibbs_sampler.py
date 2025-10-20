"""Gibbs sampler."""

import os
import numpy as np
from scipy.special import gammaln, logsumexp
from typing import Union, NoReturn, Dict
import joblib
import logging


class GibbsSampler:
    """GibbsSampler."""

    def __init__(self, x_wordtype_counts_left: np.ndarray, x_wordtype_counts_right: np.ndarray, n_classes: int):
        """Init."""
        # validate input params
        assert isinstance(x_wordtype_counts_left, np.ndarray)
        assert isinstance(x_wordtype_counts_right, np.ndarray)
        assert isinstance(n_classes, int)
        assert n_classes > 0

        self._x_wordtype_counts_left = x_wordtype_counts_left
        self._x_wordtype_counts_right = x_wordtype_counts_right
        self._n_classes = n_classes

        # initialise gibbs structures
        self._x_class_priors = np.array([1 / self._n_classes] * self._n_classes)  # shape: Z (n_classes,)
        self._n_wordtypes, self._n_features = self._x_wordtype_counts_left.shape
        self._x_wordtype_class_assignments = np.random.choice(
            range(len(self._x_class_priors)), p=self._x_class_priors, size=self._n_wordtypes
        )  # shape: M (n_wordtypes,)
        self._x_class_counts = np.bincount(self._x_wordtype_class_assignments)  # shape: Z (n_classes,)
        self._x_class_wordtype_counts_left = np.zeros((self._n_classes, self._n_features)).astype(int)
        self._x_class_wordtype_counts_right = np.zeros((self._n_classes, self._n_features)).astype(int)
        np.add.at(self._x_class_wordtype_counts_left, self._x_wordtype_class_assignments, self._x_wordtype_counts_left)
        np.add.at(
            self._x_class_wordtype_counts_right, self._x_wordtype_class_assignments, self._x_wordtype_counts_right
        )
        self._x_class_wordtype_counts_sum_left = self._x_class_wordtype_counts_left.sum(axis=1)
        self._x_class_wordtype_counts_sum_right = self._x_class_wordtype_counts_right.sum(axis=1)
        self._x_wordtype_counts_sum_left = self._x_wordtype_counts_left.sum(axis=1)  # M (n_wordtypes)
        self._x_wordtype_counts_sum_right = self._x_wordtype_counts_right.sum(axis=1)  # M (n_wordtypes)

        self._log_probs_trace = None
        self._class_counts_trace = None
        self._samples = None
        self._x_word_type_posterior_probs = None

        # validate gibbs structures
        _ = self._validate_initialisation()

    @classmethod
    def instantiate(
        cls, path_wordtype_counts_left: str, path_wordtype_counts_right: str, n_classes: int
    ) -> "GibbsSampler":
        """Instantiate GibbsSampler."""
        # asserts
        assert isinstance(path_wordtype_counts_left, str)
        assert isinstance(path_wordtype_counts_right, str)
        assert isinstance(n_classes, int)
        assert n_classes > 0

        # load
        x_wordtype_counts_left: np.ndarray = joblib.load(path_wordtype_counts_left)
        x_wordtype_counts_right: np.ndarray = joblib.load(path_wordtype_counts_right)

        # assert load
        assert isinstance(x_wordtype_counts_left, np.ndarray)
        assert isinstance(x_wordtype_counts_right, np.ndarray)

        # instantiate
        return cls(
            x_wordtype_counts_left=x_wordtype_counts_left,
            x_wordtype_counts_right=x_wordtype_counts_right,
            n_classes=n_classes,
        )

    def _validate_initialisation(self) -> Union[None, NoReturn]:
        """Validate initialisation structures."""
        assert self._x_class_priors.shape == (self._n_classes,)
        assert self._x_wordtype_class_assignments.shape == (self._n_wordtypes,)
        assert self._x_class_counts.shape == (self._n_classes,), f'{self._x_class_counts.shape} != {self._n_classes}'
        assert self._x_wordtype_class_assignments.shape == (self._n_wordtypes,)
        assert self._x_class_wordtype_counts_left.shape == (self._n_classes, self._n_features)
        assert self._x_class_wordtype_counts_right.shape == (self._n_classes, self._n_features)
        assert self._x_class_wordtype_counts_sum_left.shape == (self._n_classes,)
        assert self._x_class_wordtype_counts_sum_right.shape == (self._n_classes,)
        assert self._x_wordtype_counts_sum_left.shape == (self._n_wordtypes,)
        assert self._x_wordtype_counts_sum_right.shape == (self._n_wordtypes,)

    def _compute_log_conditional_probability(
        self, ix: int, alpha: float, beta_left: float, beta_right: float
    ) -> np.ndarray:
        """Compute Gibbs sampling log conditional probability."""
        # compute prior
        log_probs: float = np.log(self._x_class_counts + alpha)  # drop denominator since its common to all classes.

        # left context features
        log_probs += (
            gammaln(self._x_class_wordtype_counts_left + self._x_wordtype_counts_left[ix] + beta_left)
            - gammaln(self._x_class_wordtype_counts_left + beta_left)
        ).sum(axis=1) - (
            gammaln(
                self._x_class_wordtype_counts_sum_left
                + self._x_wordtype_counts_sum_left[ix]
                + self._n_features * beta_left
            )
            - gammaln(self._x_class_wordtype_counts_sum_left + self._n_features * beta_left)
        )

        # right context features
        log_probs += (
            gammaln(self._x_class_wordtype_counts_right + self._x_wordtype_counts_right[ix] + beta_right)
            - gammaln(self._x_class_wordtype_counts_right + beta_right)
        ).sum(axis=1) - (
            gammaln(
                self._x_class_wordtype_counts_sum_right
                + self._x_wordtype_counts_sum_right[ix]
                + self._n_features * beta_right
            )
            - gammaln(self._x_class_wordtype_counts_sum_right + self._n_features * beta_right)
        )

        # normalise
        log_probs -= logsumexp(log_probs)

        # convert to probs
        probs: float = np.exp(log_probs)

        return probs

    def _compute_log_joint_probability(self, alpha: float, beta_left: float, beta_right: float) -> float:
        """Compute collapsed log joint: log p(z, f | alpha, beta)."""

        # Prior over z: Dirichlet-multinomial on class counts
        log_prob: float = gammaln(self._n_classes * alpha) - gammaln(self._n_wordtypes + self._n_classes * alpha)
        log_prob += (gammaln(self._x_class_counts + alpha) - gammaln(alpha)).sum()

        # Likelihood: product over classes of Dirichlet-multinomial on feature counts
        # left context
        log_prob += (
            gammaln(self._n_features * beta_left)
            - gammaln(self._x_class_wordtype_counts_sum_left + self._n_features * beta_left)
        ).sum()
        log_prob += (gammaln(self._x_class_wordtype_counts_left + beta_left) - gammaln(beta_left)).sum()
        # right context
        log_prob += (
            gammaln(self._n_features * beta_right)
            - gammaln(self._x_class_wordtype_counts_sum_right + self._n_features * beta_right)
        ).sum()
        log_prob += (gammaln(self._x_class_wordtype_counts_right + beta_right) - gammaln(beta_right)).sum()

        return log_prob

    def _remove_class_assignment(self, ix: int) -> int:
        """Remove word type class assignment."""
        assert isinstance(ix, (int, np.integer))

        # get word type class assignment
        z: int = self._x_wordtype_class_assignments[ix]

        # decrement class count
        self._x_class_counts[z] -= 1

        # decrement class word type counts
        self._x_class_wordtype_counts_left[z] -= self._x_wordtype_counts_left[ix]
        self._x_class_wordtype_counts_right[z] -= self._x_wordtype_counts_right[ix]

        # decrement from class totals
        self._x_class_wordtype_counts_sum_left[z] -= self._x_wordtype_counts_sum_left[ix]
        self._x_class_wordtype_counts_sum_right[z] -= self._x_wordtype_counts_sum_right[ix]

        return z

    def _add_class_assignment(self, ix: int, z: int) -> None:
        """Add word type class assignment."""
        assert isinstance(ix, (int, np.integer))
        assert isinstance(z, (int, np.integer))

        # assign new class
        self._x_wordtype_class_assignments[ix] = z

        # increment class count
        self._x_class_counts[z] += 1

        # increment class word type counts
        self._x_class_wordtype_counts_left[z] += self._x_wordtype_counts_left[ix]
        self._x_class_wordtype_counts_right[z] += self._x_wordtype_counts_right[ix]

        # increment from class totals
        self._x_class_wordtype_counts_sum_left[z] += self._x_wordtype_counts_sum_left[ix]
        self._x_class_wordtype_counts_sum_right[z] += self._x_wordtype_counts_sum_right[ix]

    def _run_sweep(self, alpha: float, beta_left: float, beta_right: float) -> None:
        """Run Gibbs sweep."""
        # --- gibbs sweep
        for ix_wordtype in np.random.permutation(self._n_wordtypes):

            # --- remove word type assignment
            z_old: int = self._remove_class_assignment(ix=ix_wordtype)

            # --- recompute gibbs log conditional
            class_probs: np.ndarray = self._compute_log_conditional_probability(
                ix=ix_wordtype, alpha=alpha, beta_left=beta_left, beta_right=beta_right
            )

            # --- sample new assignment
            z_new: int = np.random.choice(self._n_classes, p=class_probs)

            # --- add word type assignment
            _ = self._add_class_assignment(ix=ix_wordtype, z=z_new)

    def compute_posterior_class_probs(self, wordtype_index: Dict[str, int]) -> np.ndarray:
        """Compute empirical posterior P(z_j = c | data) for one word type."""
        # asserts
        assert self._samples is not None

        x_counts: np.ndarray = np.zeros(self._n_classes, dtype=float)
        for z in self._samples:
            x_counts[z[wordtype_index]] += 1
        probs: np.ndarray = x_counts / x_counts.sum()
        return probs

    def compute_word_type_posterior_entropy(self, epsilon: float = 1e-12):
        """Compute word type posterior entropy."""
        # asserts
        assert self._x_word_type_posterior_probs is not None
        assert isinstance(epsilon, float)
        assert epsilon > 0

        x_word_type_posterior_probs_clipped = np.clip(self._x_word_type_posterior_probs, epsilon, 1.0)
        return -(x_word_type_posterior_probs_clipped * np.log(x_word_type_posterior_probs_clipped)).sum(axis=1)

    def save_outputs(self, output_path: str) -> None:
        """Save output artifacts."""
        # asserts
        assert isinstance(output_path, str)
        assert self._log_probs_trace is not None
        assert self._class_counts_trace is not None
        assert self._samples is not None
        assert self._x_word_type_posterior_probs is not None

        _ = os.makedirs(output_path, exist_ok=True)

        _ = joblib.dump(self, f"{output_path}/sampler.joblib")

    def run(
        self, n_iterations: int, alpha: float, beta_left: float, beta_right: float, n_burn_in: int, n_thinning: int
    ) -> None:
        """Run Gibbs sampler."""
        # asserts
        assert isinstance(n_iterations, int)
        assert n_iterations > 0
        assert isinstance(alpha, float)
        assert isinstance(beta_left, float)
        assert isinstance(beta_right, float)

        log_probs_trace, class_counts_trace, samples = [], [], []
        # reset posterior accumulators
        x_word_type_posterior_counts: np.ndarray = np.zeros((self._n_wordtypes, self._n_classes), dtype=int)
        n_posterior_samples_kept: int = 0

        for ix_iteration in range(n_iterations):
            _ = self._run_sweep(alpha=alpha, beta_left=beta_left, beta_right=beta_right)
            log_prob: float = self._compute_log_joint_probability(
                alpha=alpha, beta_left=beta_left, beta_right=beta_right
            )

            # trace
            log_probs_trace.append(log_prob)
            class_counts_trace.append(self._x_class_counts.copy())

            # collect thinned samples after burn-in
            if ix_iteration >= n_burn_in and ((ix_iteration - n_burn_in) % n_thinning == 0):
                samples.append(self._x_wordtype_class_assignments.copy())

                # online posterior accumulation: increment (j, z_j)
                x_word_type_posterior_counts[np.arange(self._n_wordtypes), self._x_wordtype_class_assignments] += 1
                n_posterior_samples_kept += 1

            # log
            logging.info(f"Iteration: {ix_iteration} log_prob: {log_prob:.2f}")

        x_word_type_posterior_probs: np.ndarray = x_word_type_posterior_counts / float(n_posterior_samples_kept)

        self._log_probs_trace = log_probs_trace
        self._class_counts_trace = class_counts_trace
        self._samples = samples
        self._x_word_type_posterior_probs = x_word_type_posterior_probs
