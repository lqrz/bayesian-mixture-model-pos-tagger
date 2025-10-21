"""TrainerConfig."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TrainerConfig:
    path_wordtype_counts_left: str
    path_wordtype_counts_right: str
    path_output: str
    alpha: float
    beta_left: float
    beta_right: float
    n_classes: int
    n_iterations: int
    n_burn_in: int
    n_thinning: int
    seed: Optional[int] = 1234
