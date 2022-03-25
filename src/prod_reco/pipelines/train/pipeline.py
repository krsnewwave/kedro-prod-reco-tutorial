"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.17.7
"""


from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_train_test, factorize, produce_sample_recos
from .nodes import build_index, upload_to_mlflow, validate_index

def create_pipeline(**kwargs) -> Pipeline:
    training_pipeline = pipeline(
        [
            node(
                func=split_train_test,
                inputs=["interactions", "params:evaluation"],
                outputs={"train": "train", "test": "test", "eval_train": "eval_train"},
                name="train_test_split",
            ),
            node(
                func=factorize,
                inputs=["train", "test", "eval_train", "sp_item_feats", "params:model"],
                outputs={"user_factors": "user_factors",
                         "item_factors": "item_factors",
                         "user_biases": "user_biases",
                         "item_biases": "item_biases",
                         "model_metrics": "model_metrics"},
                name="factorize",
            ),
            node(
                func=produce_sample_recos,
                inputs=["user_factors", "item_factors", "user_biases", "item_biases", "idx_to_names", "idx_to_rid"],
                outputs="sample_recos",
                name="produce_sample_recos",
            ),
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
                inputs=["idx_to_names", "item_factors", "user_factors",
                        "item_biases", "user_biases", "item_rank", "params:index_params"],
                outputs=None,
                name="upload_to_mlflow",
            )
        ],
    )

    return training_pipeline