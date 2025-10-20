"""Preprocess text data for POS tagging."""

import numpy as np
import regex as re
from glob import glob
from collections import Counter
from operator import itemgetter
from typing import List, Generator, Tuple
import joblib
import os


class Preprocessor:
    """Preprocessor."""

    def __init__(self):
        """Init."""
        self._bos = "<BOS>"
        self._eos = "<EOS>"
        # counter
        self._counter = None
        self._n_features = None
        self._feature_to_ix = None
        self._wordtype_to_ix = None
        self._ix_to_wordtype = None
        # featurise
        self._x_wordtype_counts_left = None
        self._x_wordtype_counts_right = None

    def _read_data(self, file_path: str) -> Generator[str, None, None]:
        """Read text file."""
        # asserts
        assert isinstance(file_path, str)

        for f in glob(file_path):
            for l in open(f, "r").readlines():
                yield l

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize."""
        # asserts
        assert isinstance(text, str)

        rx = re.compile(r"\b\p{L}[\p{L}\p{M}\p{N}'’-]*\b", re.UNICODE)
        return rx.findall(text)

    def _instantiate_counter(self, file_path: str) -> Counter:
        """Instantiate counter."""
        # asserts
        assert isinstance(file_path, str)

        # instantiate counter
        self._counter = Counter()
        for l in self._read_data(file_path=file_path):
            self._counter.update(self._tokenize(text=l))

        n_most_common_wordtypes: int = 100
        self._n_features: int = n_most_common_wordtypes + 2
        most_common_wordtypes: List[str] = list(
            map(itemgetter(0), self._counter.most_common(n_most_common_wordtypes))
        ) + [
            self._bos,
            self._eos,
        ]
        self._feature_to_ix = dict(zip(most_common_wordtypes, range(len(most_common_wordtypes))))
        self._wordtype_to_ix = dict(zip(self._counter.keys(), range(len(self._counter.keys()))))
        self._ix_to_wordtype = dict(zip(self._wordtype_to_ix.values(), self._wordtype_to_ix.keys()))

    def _featurise(self, file_path: str) -> None:
        """Featurise corpus."""
        # asserts
        assert isinstance(file_path, str)
        assert self._counter is not None
        assert self._n_features is not None
        assert self._wordtype_to_ix is not None
        assert self._ix_to_wordtype is not None

        n_wordtypes: int = len(self._wordtype_to_ix)
        # extra dimension to count ignored words. to be removed later.
        x_wordtype_counts_left = np.zeros((n_wordtypes, self._n_features + 1)).astype(int)
        x_wordtype_counts_right = np.zeros((n_wordtypes, self._n_features + 1)).astype(int)

        for l in self._read_data(file_path=file_path):
            tokens = [self._bos] + self._tokenize(text=l) + [self._eos]
            for ix in range(1, len(tokens) - 1):
                x_wordtype_counts_left[
                    self._wordtype_to_ix[tokens[ix]], self._feature_to_ix.get(tokens[ix - 1], -1)
                ] += 1
                x_wordtype_counts_right[
                    self._wordtype_to_ix[tokens[ix]], self._feature_to_ix.get(tokens[ix + 1], -1)
                ] += 1

        # remove ignored words
        self._x_wordtype_counts_left = x_wordtype_counts_left[
            :, : self._n_features
        ]  # M (n_wordtypes) x F (n_features)
        self._x_wordtype_counts_right = x_wordtype_counts_right[
            :, : self._n_features
        ]  # M (n_wordtypes) x F (n_features)

        # assert outputs
        assert (
            self._x_wordtype_counts_left.shape
            == self._x_wordtype_counts_right.shape
            == (n_wordtypes, self._n_features)
        )
        assert (
            self._x_wordtype_counts_right.shape
            == self._x_wordtype_counts_right.shape
            == (n_wordtypes, self._n_features)
        )

    def save_outputs(self, output_path: str) -> Tuple[str, str, str, str, str]:
        """Save output artifacts."""
        # asserts
        assert isinstance(output_path, str)
        assert self._counter is not None
        assert self._wordtype_to_ix is not None
        assert self._ix_to_wordtype is not None
        assert self._x_wordtype_counts_left is not None
        assert self._x_wordtype_counts_right is not None

        _ = os.makedirs(output_path, exist_ok=True)

        path_output_counter: str = f"{output_path}/counter.joblib"
        path_output_wordtype_to_ix: str = f"{output_path}/wordtype_to_ix.joblib"
        path_output_ix_to_wordtype: str = f"{output_path}/ix_to_wordtype.joblib"
        path_output_x_wordtype_counts_left: str = f"{output_path}/x_wordtype_counts_left.joblib"
        path_output_x_wordtype_counts_right: str = f"{output_path}/x_wordtype_counts_right.joblib"

        # dump
        _ = joblib.dump(self._counter, path_output_counter)
        _ = joblib.dump(self._wordtype_to_ix, path_output_wordtype_to_ix)
        _ = joblib.dump(self._ix_to_wordtype, path_output_ix_to_wordtype)
        _ = joblib.dump(self._x_wordtype_counts_left, path_output_x_wordtype_counts_left)
        _ = joblib.dump(self._x_wordtype_counts_right, path_output_x_wordtype_counts_right)

        return (
            path_output_counter,
            path_output_wordtype_to_ix,
            path_output_ix_to_wordtype,
            path_output_x_wordtype_counts_left,
            path_output_x_wordtype_counts_right,
        )

    def preprocess(self, file_path: str) -> None:
        """Preprocess corpus."""
        # asserts
        assert isinstance(file_path, str)
        assert len(glob(file_path)) > 0, f"No files found at location: {file_path}"

        _ = self._instantiate_counter(file_path=file_path)
        _ = self._featurise(file_path=file_path)
