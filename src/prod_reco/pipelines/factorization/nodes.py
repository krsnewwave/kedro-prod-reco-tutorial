"""
Modeling
"""

import scipy
from typing import Dict
from prod_reco.commons.recommender_utils import RecommenderUtils
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
import numpy as np


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
    return {"train" : train, "test": test, "eval_train": eval_train}


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
    for idx_epoch in range(epochs):
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

    return warp_model
