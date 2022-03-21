"""
This is a boilerplate pipeline 'factorization'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_train_test, factorize, produce_sample_recos


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
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
            )
        ],
    )
