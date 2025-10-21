"""Trainer."""

import logging
from typing import List, Tuple, Optional
from joblib import Parallel, delayed
from dataclasses import asdict

from lqrz.bayesian_mixture_model_pos_tagger.train.trainer_config import TrainerConfig
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
        seed: int,
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
        assert isinstance(seed, int)

        # run
        sampler = GibbsSampler.instantiate(
            path_wordtype_counts_left=path_wordtype_counts_left,
            path_wordtype_counts_right=path_wordtype_counts_right,
            n_classes=n_classes,
            seed=seed,
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
    def _run_concurrent(
        cls, configs: List[TrainerConfig], n_workers: int, prefer: str = "processes", verbose: int = 10
    ) -> None:
        """Run concurrently."""
        _ = Parallel(n_jobs=n_workers, prefer=prefer, verbose=verbose)(
            delayed(cls._run_sampler)(**asdict(x)) for x in configs
        )

    @classmethod
    def _run_sequential(cls, configs: List[TrainerConfig]) -> None:
        """Run sequentially."""
        for x in configs:
            _ = cls._run_sampler(**asdict(x))

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
        n_workers: Optional[int] = 1,
    ) -> None:
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
        assert isinstance(n_workers, int) and n_workers > 0

        # run preprocessor
        path_output_preprocessor: str = f"{path_output}/preprocessor"

        _ = logging.info(f"Running preprocessor. Saving outputs in: {path_output_preprocessor}")

        path_wordtype_counts_left, path_wordtype_counts_right = cls._run_preprocessor(
            path_input_data=path_input_data,
            path_output=path_output_preprocessor,
        )

        # trainer configs
        configs: List[TrainerConfig] = [
            TrainerConfig(
                path_wordtype_counts_left=path_wordtype_counts_left,
                path_wordtype_counts_right=path_wordtype_counts_right,
                path_output=f"{path_output}/train/{nc}_{ni}_{alpha}_{beta_left}_{beta_right}_{nbi}_{nt}",
                alpha=alpha,
                beta_left=beta_left,
                beta_right=beta_right,
                n_classes=nc,
                n_iterations=ni,
                n_burn_in=nbi,
                n_thinning=nt,
            )
            for nc, ni, alpha, beta_left, beta_right, nbi, nt in zip(
                n_classes, n_iterations, alphas, betas_left, betas_right, ns_burn_in, ns_thinning
            )
        ]

        if n_workers > 1:
            cls._run_concurrent(configs=configs, n_workers=n_workers)
        else:
            cls._run_sequential(configs=configs)
