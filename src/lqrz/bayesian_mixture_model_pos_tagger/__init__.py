"""Init."""

import logging

MODULE = __name__
__version__ = "1.0.0"
LIBRARY_NAME = "bayesian-mixture-model-pos-tagger"


def set_logging():
    """Activate logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(processName)s  %(name)s  %(message)s",
    )
    logging.getLogger().setLevel(logging.INFO)
