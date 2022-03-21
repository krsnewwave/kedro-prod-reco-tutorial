"""
This is a boilerplate pipeline 'deploy_model'
generated using Kedro 0.17.7
"""


from kedro.pipeline import Pipeline, node, pipeline
from .nodes import build_index, validate_index


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_index,
                inputs=["item_factors", "params:index_params"],
                outputs="annoy_index",
                name="build_index",
            ),
            node(
                func=validate_index,
                inputs=["annoy_index", "idx_to_names"],
                outputs=None,
                name="validate_index",
            )
        ],
    )
