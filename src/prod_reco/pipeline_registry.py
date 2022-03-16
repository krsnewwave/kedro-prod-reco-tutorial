"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from prod_reco.pipelines import data_engineering, factorization

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    de_pipe = data_engineering.create_pipeline()
    factor_pipe = factorization.create_pipeline()
    return {
        "__default__": de_pipe + factor_pipe,
        "de" : de_pipe,
        "factorization" : factor_pipe,
    }
