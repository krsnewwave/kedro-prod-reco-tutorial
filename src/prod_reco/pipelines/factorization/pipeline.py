"""
This is a boilerplate pipeline 'factorization'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_train_test, factorize

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_train_test,
                inputs=["interactions", "params:evaluation"],
                outputs={"train" : "train", "test" : "test", "eval_train" : "eval_train"},
                name="train_test_split",
            ),
            node(
                func=factorize,
                inputs=["train", "test", "eval_train", "sp_item_feats", "params:model"],
                outputs="warp_model",
                name="factorize",
            )
        ],
    )
