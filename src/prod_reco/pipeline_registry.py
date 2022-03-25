"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from kedro_mlflow.pipeline import pipeline_ml_factory
from sys import version_info
import mlflow
import cloudpickle
import numpy as np
import pandas as pd

from prod_reco.pipelines import data_engineering, deploy_model, factorization

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)
conda_env = {
    'channels': ['defaults'],
    'dependencies': [
        'python={}'.format(PYTHON_VERSION),
        'pip',
        {
            'pip': [
                f"mlflow=={mlflow.__version__}",
                "annoy",
                f"cloudpickle=={cloudpickle.__version__}",
                f"numpy=={np.__version__}",
                f"pandas=={pd.__version__}"
            ],
        },
    ],
    'name': 'reco_env'
}


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    de_pipe = data_engineering.create_pipeline()
    factor_pipe = factorization.create_pipeline()

    deploy_pipe = deploy_model.create_pipeline()
    # serving_pipeline = pipeline_ml_factory(
    #     training=deploy_pipe.only_nodes_with_tags("indexing"),
    #     inference=deploy_pipe.only_nodes_with_tags("inference"),
    #     input_name="model_input",
    #     log_model_kwargs={
    #         "conda_env": conda_env,
    #         "model_signature": None},
    # )
    return {
        "__default__": de_pipe + factor_pipe + deploy_pipe,
        "de": de_pipe,
        "factorization": factor_pipe,
        "deploy": deploy_pipe,
        # "serve": serving_pipeline
    }
