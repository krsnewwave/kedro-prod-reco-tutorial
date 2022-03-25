"""
Modeling
"""

import pandas as pd
from typing import Dict
from prod_reco.commons.recommender_utils import RecommenderUtils
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
import numpy as np
import logging


def split_train_test(interactions, params: Dict):
    split_count = params["split_count"]
    split_fraction = params["split_fraction"]
    fraction = params["fraction"]
    random_seed = params["random_seed"]

    np.random.seed(random_seed)

    train, test, test_users = RecommenderUtils.train_test_split_sparse(
        interactions, split_count, split_fraction=split_fraction, fraction=fraction)

    # create a train set where the train-only users have unknown recos
    # this is for training set evaluation
    eval_train = train.copy()
    non_eval_users = list(set(range(train.shape[0])) - set(test_users))

    eval_train = eval_train.tolil()
    for u in non_eval_users:
        eval_train[u, :] = 0.0
    eval_train = eval_train.tocsr()
    return {"train": train, "test": test, "eval_train": eval_train}


def factorize(train, test, eval_train, sp_item_feats, params: Dict):

    random_seed = params["random_seed"]
    epochs = params["epochs"]
    k = params["k"]
    n_components = params["n_components"]
    loss = params["loss"]

    list_train_prec = []
    list_test_prec = []
    warp_model = LightFM(no_components=n_components,
                         loss=loss, random_state=random_seed)
    for _ in range(epochs):
        warp_model.fit_partial(train, item_features=sp_item_feats,
                               num_threads=2, epochs=1)
        test_prec = precision_at_k(
            warp_model, test, train_interactions=train, k=k, item_features=sp_item_feats)
        train_prec = precision_at_k(
            warp_model, eval_train, train_interactions=None, k=k, item_features=sp_item_feats)

        test_prec = np.mean(test_prec)
        train_prec = np.mean(train_prec)

        print(f"Train: {train_prec}, Test: {test_prec}")

        list_test_prec.append(test_prec)
        list_train_prec.append(train_prec)

    # TODO: take history using step
    dict_metrics = {f"train_precision_at_{k}": {"value" : train_prec, "step" : 0},
                    f"test_precision_at_{k}": {"value" : test_prec, "step" : 0}}

    item_biases, item_factors = warp_model.get_item_representations(features=sp_item_feats)
    user_biases, user_factors = warp_model.get_user_representations()

    # TODO: put these datasets into mlflow

    return {"user_factors": user_factors,
            "item_factors": item_factors,
            "user_biases": user_biases,
            "item_biases": item_biases,
            "model_metrics": dict_metrics}


def produce_sample_recos(user_factors, item_factors, user_biases, item_biases,
                         idx_to_names, idx_to_rid):
    """Produce sample recommendations that could be used for smoke testing.
    Based off of https://github.com/lyst/lightfm/issues/617

    Args:
        user_factors (np.array): _description_
        item_factors (np.array): _description_
        user_biases (np.array): _description_
        item_biases (np.array): _description_
        idx_to_names (dict): mapping array
    """
    # get only 3 users and 100 items
    m = 3
    n = 100
    user_factors = user_factors[:m, :]
    user_biases = user_biases[:m]

    item_factors = item_factors[:n, :]
    item_biases = item_biases[:n]

    scores = RecommenderUtils.produce_scores(item_factors, item_biases, user_factors, user_biases)
    # now you can sort and assert things here. I'll just let it pass
    # shape: 5x100
    assert scores.shape[0] == m
    assert scores.shape[1] == n
    sorted_scores_argsort = np.argsort(scores, axis=1)

    logger = logging.getLogger(__name__)
    list_dict_recos = []
    for idx in range(len(sorted_scores_argsort)):
        recos = [idx_to_names[v] for v in sorted_scores_argsort[idx]]
        logger.info(f"Recos for {idx_to_rid[idx]}:")
        logger.info(recos)

        list_dict_recos.append({"userId" : idx_to_rid[idx], "recos": recos})

    return pd.DataFrame(list_dict_recos)
    
