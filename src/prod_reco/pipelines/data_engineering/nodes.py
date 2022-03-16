"""
Data prep nodes
"""
import pandas as pd
from prod_reco.commons.recommender_utils import RecommenderUtils
from typing import Dict
from sklearn.feature_extraction import DictVectorizer
import scipy

# PARAMETERS
USER_ID = "userId"
ITEM_ID = "itemId"
RATING = "rating"
ITEM_NAME = "movieName"
TAGS = "tags"


def prep_sparse_ratings(df_ratings: pd.DataFrame, params: Dict):
    """Converts ratings to sparse with their representations

    Args:
        df_ratings (pd.DataFrame): ratings
        params (Dict): parameters

    Returns:
        interactions: interactions
        rid_to_idx: user ids (rid) to row indexes (idx)
        idx_to_rid: row indexes to user ids
        cid_to_idx: item ids (cid) to column indexes (idx)
        idx_to_cid: column indexes to item ids
    """
    # extract params
    user_min_bought = params["user_min_bought"]
    item_min_bought = params["item_min_bought"]

    # ratings of 5 are 1, everything else is deleted
    df_ratings = df_ratings[df_ratings[RATING] > 4]
    df_ratings[RATING] = 1

    # create utils object
    utils = RecommenderUtils(user_id=USER_ID, item_id=ITEM_ID, rating=RATING)

    # threshold interactions
    df_txn = RecommenderUtils.threshold_interactions_df(df_ratings, USER_ID, ITEM_ID,
                                                        user_min_bought, item_min_bought)
    utils.print_ratings_shape(df_txn)

    # to sparse matrix
    interactions, rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid = \
        RecommenderUtils.df_to_matrix(df_txn, USER_ID, ITEM_ID, interaction_var=RATING)

    return interactions, rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid

def prep_item_features(df_items: pd.DataFrame, cid_to_idx: Dict, idx_to_cid: Dict):
    """Prepare item features

    Args:
        df_items (pd.DataFrame): item data
        cid_to_idx (Dict): item id to column index mapping
        idx_to_cid (Dict): column index to item id mapping

    Returns:
        sp_item_feats: sparse item features
        idx_to_names: dictionary of item names
    """
    # reorder and filter df_items
    df_items = df_items.set_index(ITEM_ID).loc[cid_to_idx]

    # get metadata tags
    df_items_feats = df_items["tags"]

    # get movie names
    df_item_names = df_items["movieName"]
    idx_to_names = {cid_to_idx[k]:v for k,v in df_item_names.to_dict().items()}

    # convert item tags to sparse representation (features)
    # (1) convert tags to list of dictionaries
    dummies_items = df_items_feats.str.get_dummies(sep='|')

    list_of_dict_item_features = [{} for _ in idx_to_cid]
    for idx, row in dummies_items.iterrows():
        dict_item_feat = {k:v for k,v in row.items() if v > 0}
        list_of_dict_item_features[cid_to_idx[idx]] = dict_item_feat

    # (2) use DictVectorizer to convert to sparse
    item_vec = DictVectorizer()
    sp_item_feats = item_vec.fit_transform((list_of_dict_item_features))

    # (3) format identity matrix and hstack with item feats
    sp_items_eye = scipy.sparse.eye(sp_item_feats.shape[0])
    sp_item_feats = scipy.sparse.hstack((sp_items_eye, sp_item_feats))

    return sp_item_feats, idx_to_names
