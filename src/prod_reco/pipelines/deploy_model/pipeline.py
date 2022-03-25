"""
This is a boilerplate pipeline 'deploy_model'
generated using Kedro 0.17.7
"""


from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_index, upload_to_mlflow, validate_index, recommend_node


def create_pipeline(**kwargs) -> Pipeline:
    indexing_pipeline = pipeline(
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
            ),
            node(
                func=upload_to_mlflow,
                inputs=["annoy_index", "idx_to_names", "item_factors", "user_factors",
                        "item_biases", "user_biases", "item_rank", "params:index_params"],
                outputs=None,
                name="upload_to_mlflow",
            )
        ],
        # tags=["indexing"]
        # namespace="deployed"
    )

    return indexing_pipeline

    # inference_pipeline = pipeline(
    #     [
    #         node(
    #             func=recommend_node,
    #             inputs=["reco_model", "model_input", "item_factors", "item_biases",
    #                "user_factors", "user_biases", "idx_to_names", "item_rank"],
    #             outputs="recommendations",
    #             name="recommend_node",
    #             tags=["inference"]
    #         ),
    #     ]
    # )
    # return indexing_pipeline + inference_pipeline
