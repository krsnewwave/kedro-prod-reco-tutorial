"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.17.7
"""


from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_train_test, factorize, produce_sample_recos, factorize_optimize
from .nodes import build_index, upload_to_mlflow, validate_index

def create_pipeline(**kwargs) -> Pipeline:
    # determine if hyperparam optimize
    if "hyperparam_optimize" in kwargs:
        training_func = factorize_optimize
    else:
        training_func = factorize

    training_pipeline = pipeline(
        [
            node(
                func=split_train_test,
                inputs=["interactions", "params:evaluation"],
                outputs={"train": "train", "test": "test", "eval_train": "eval_train"},
                name="train_test_split",
            ),
            node(
                func=training_func,
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
                outputs="kedro_annoy_dataset",
                name="build_index",
            ),
            node(
                func=validate_index,
                inputs=["kedro_annoy_dataset", "idx_to_names"],
                outputs="validated_kedro_annoy_dataset",
                name="validate_index",
            ),
            node(
                func=upload_to_mlflow,
                inputs=["validated_kedro_annoy_dataset", "idx_to_names", "item_factors", "user_factors",
                        "item_biases", "user_biases", "item_rank", "params:index_params"],
                outputs=None,
                name="upload_to_mlflow",
            )
        ],
    )

    return training_pipeline