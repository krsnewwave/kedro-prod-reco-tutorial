"""Project pipelines."""
from sys import version_info
from typing import Dict

from kedro.pipeline import Pipeline

from prod_reco.pipelines import data_engineering, train

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    de_pipe = data_engineering.create_pipeline()
    train_pipe = train.create_pipeline()
    return {
        "__default__": de_pipe + train_pipe,
        "de": de_pipe,
        "train": train_pipe
    }
