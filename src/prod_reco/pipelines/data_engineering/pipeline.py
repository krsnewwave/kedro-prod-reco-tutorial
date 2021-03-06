"""
Data prep pipeline
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prep_sparse_ratings, prep_item_features, create_default_item_ranking


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prep_sparse_ratings,
                inputs=["ratings", "params:preprocessing"],
                outputs=["interactions", "rid_to_idx", "idx_to_rid",
                         "cid_to_idx", "idx_to_cid"],
                name="prep_ratings",
            ),
            node(
                func=prep_item_features,
                inputs=["items", "cid_to_idx", "idx_to_cid"],
                outputs=["sp_item_feats", "idx_to_names"],
                name="prep_item_features",
            ),
            node(
                func=create_default_item_ranking,
                inputs=["ratings", "items", "cid_to_idx", "params:preprocessing"],
                outputs="item_rank",
                name="create_default_item_ranking",
            )
            
        ],
    )
