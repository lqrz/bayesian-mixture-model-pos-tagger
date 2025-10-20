"""Trainer."""

import logging
from typing import List, Tuple

from lqrz.bayesian_mixture_model_pos_tagger.get_data.preprocessor import Preprocessor
from lqrz.bayesian_mixture_model_pos_tagger.train.gibbs_sampler import GibbsSampler


class Trainer:
    """Trainer."""

    def __init__(self):
        """Init."""
        pass

    @staticmethod
    def _run_sampler(
        path_wordtype_counts_left: str,
        path_wordtype_counts_right: str,
        n_classes: int,
        n_iterations: int,
        alpha: float,
        beta_left: float,
        beta_right: float,
        n_burn_in: int,
        n_thinning: int,
        path_output: str,
    ) -> None:
        """Run gibbs sampler."""
        # asserts
        assert isinstance(path_wordtype_counts_left, str)
        assert isinstance(path_wordtype_counts_right, str)
        assert isinstance(n_classes, int)
        assert isinstance(n_iterations, int)
        assert isinstance(alpha, float)
        assert isinstance(beta_left, float)
        assert isinstance(beta_right, float)
        assert isinstance(n_burn_in, int)
        assert isinstance(n_thinning, int)
        assert isinstance(path_output, str)

        # run
        sampler = GibbsSampler.instantiate(
            path_wordtype_counts_left=path_wordtype_counts_left,
            path_wordtype_counts_right=path_wordtype_counts_right,
            n_classes=n_classes,
        )
        _ = sampler.run(
            n_iterations=n_iterations,
            alpha=alpha,
            beta_left=beta_left,
            beta_right=beta_right,
            n_burn_in=n_burn_in,
            n_thinning=n_thinning,
        )

        # save outputs
        _ = sampler.save_outputs(output_path=path_output)

    @staticmethod
    def _run_preprocessor(path_input_data: str, path_output: str) -> Tuple[str, str]:
        """Run preprocessor."""
        # asserts
        assert isinstance(path_input_data, str)
        assert isinstance(path_output, str)

        # run
        preprocessor = Preprocessor()
        preprocessor.preprocess(file_path=path_input_data)

        # save outputs
        path_output_preprocessor: str = f"{path_output}/preprocessor"
        (
            _,
            _,
            _,
            path_wordtype_counts_left,
            path_wordtype_counts_right,
        ) = preprocessor.save_outputs(output_path=path_output_preprocessor)

        return path_wordtype_counts_left, path_wordtype_counts_right

    @classmethod
    def run(
        cls,
        path_input_data: str,
        path_output: str,
        n_classes: List[int],
        n_iterations: List[int],
        alphas: List[float],
        betas_left: List[float],
        betas_right: List[float],
        ns_burn_in: List[int],
        ns_thinning: List[int],
    ):
        """Run."""
        assert isinstance(path_input_data, str)
        assert isinstance(path_output, str) and len(path_output) > 0
        assert isinstance(n_classes, List)
        assert isinstance(n_iterations, List)
        assert isinstance(alphas, List)
        assert isinstance(betas_left, List)
        assert isinstance(betas_right, List)
        assert isinstance(ns_burn_in, List)
        assert isinstance(ns_thinning, List)
        assert all([isinstance(x, int) for x in n_classes])
        assert all([isinstance(x, int) for x in n_iterations])
        assert all([isinstance(x, float) for x in alphas])
        assert all([isinstance(x, float) for x in betas_left])
        assert all([isinstance(x, float) for x in betas_right])
        assert all([isinstance(x, int) for x in ns_burn_in])
        assert all([isinstance(x, int) for x in ns_thinning])

        # run preprocessor
        path_output_preprocessor: str = f"{path_output}/preprocessor"
        _ = logging.info(f"Running preprocessor. Saving outputs in: {path_output_preprocessor}")
        path_wordtype_counts_left, path_wordtype_counts_right = cls._run_preprocessor(
            path_input_data=path_input_data,
            path_output=path_output_preprocessor,
        )

        for nc, ni, alpha, beta_left, beta_right, nbi, nt in zip(
            n_classes, n_iterations, alphas, betas_left, betas_right, ns_burn_in, ns_thinning
        ):

            path_output_train_run: str = f"{path_output}/train/{nc}_{ni}_{alpha}_{beta_left}_{beta_right}_{nbi}_{nt}"
            _ = logging.info(
                f"Running sampler for {nc}_{ni}_{alpha}_{beta_left}_{beta_right}_{nbi}_{nt}. Saving outputs in: {path_output_train_run}"
            )
            _ = cls._run_sampler(
                path_wordtype_counts_left=path_wordtype_counts_left,
                path_wordtype_counts_right=path_wordtype_counts_right,
                n_classes=nc,
                n_iterations=ni,
                alpha=alpha,
                beta_left=beta_left,
                beta_right=beta_right,
                n_burn_in=nbi,
                n_thinning=nt,
                path_output=path_output_train_run,
            )
