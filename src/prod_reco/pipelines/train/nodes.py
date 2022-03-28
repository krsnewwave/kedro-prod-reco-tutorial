"""
Deployment phase, creating embeddings and index
"""
import logging
import tempfile
from functools import partial
from sys import version_info
from typing import Dict

import cloudpickle
import numpy as np
import optuna
import pandas as pd
from annoy import AnnoyIndex
from kedro_mlflow.io.artifacts import MlflowArtifactDataSet
from kedro_mlflow.io.models import MlflowModelLoggerDataSet
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from optuna.integration.mlflow import MLflowCallback

import mlflow
from prod_reco.commons.models import KedroMLFlowLightFM
from prod_reco.commons.datasets import KedroAnnoyIndex
from prod_reco.commons.recommender_utils import ITEM_ID, USER_ID, RecommenderUtils

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)
# TODO: hardcoded
N_RECOS = 10
N_NEIGHBORS = 5
N_RECOS = 10
ITEM_POSITIONAL_INDEX_NAME = "idx"
NUM_USERS_RANK_SORT_NAME = "num_users"
MOVIE_NAME = "movie_name"
INDEX_PATH = "data/07_model_output/deployed/annoy_index.ann"


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
    """Train model

    Args:
        train (COO sparse array): _description_
        test (COO sparse array): _description_
        eval_train (COO sparse array): train set without test users
        sp_item_feats (np array): _description_
        params (Dict): _description_

    Returns:
        _type_: _description_
    """
    random_seed = params["random_seed"]
    epochs = params["epochs"]
    k = params["k"]
    n_components = params["n_components"]
    loss = params["loss"]

    warp_model, test_prec, train_prec = train_model(
        train, test, eval_train, sp_item_feats, random_seed, epochs, k, n_components, loss)

    # TODO: take history using step
    dict_metrics = {f"train_precision_at_{k}": {"value": train_prec, "step": 0},
                    f"test_precision_at_{k}": {"value": test_prec, "step": 0}}

    item_biases, item_factors = warp_model.get_item_representations(
        features=sp_item_feats)
    user_biases, user_factors = warp_model.get_user_representations()

    return {"user_factors": user_factors,
            "item_factors": item_factors,
            "user_biases": user_biases,
            "item_biases": item_biases,
            "model_metrics": dict_metrics}


def train_model(train, test, eval_train, sp_item_feats,
                random_seed, epochs, k, n_components, loss):
    """Trains model

    Args:
        train (_type_): _description_
        test (_type_): _description_
        eval_train (_type_): _description_
        sp_item_feats (_type_): _description_
        random_seed (_type_): _description_
        epochs (_type_): _description_
        k (_type_): _description_
        n_components (_type_): _description_
        loss (_type_): _description_

    Returns:
        _type_: _description_
    """
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

        logger = logging.getLogger(__name__)
        logger.info(f"Train: {train_prec}, Test: {test_prec}")

    return warp_model, test_prec, train_prec


def optuna_objective(train, test, eval_train, sp_item_feats, params: Dict,
                     trial: optuna.trial):
    k = params["k"]
    random_seed = params["random_seed"]
    epochs = params["epochs"]
    loss = params["loss"]

    # optimize this
    n_components = trial.suggest_int("n_components", 10, 80)

    _, test_prec, _ = train_model(train, test, eval_train, sp_item_feats,
                                  random_seed, epochs, k, n_components, loss)
    return test_prec


def factorize_optimize(train, test, eval_train, sp_item_feats, params: Dict):
    k = params["k"]
    random_seed = params["random_seed"]
    epochs = params["epochs"]
    loss = params["loss"]

    study = optuna.create_study(study_name="optimize warp", direction="maximize")
    fun_objective = partial(optuna_objective, train, test,
                            eval_train, sp_item_feats, params)

    # mlflow callback for tracking
    # additional setting: nested runs
    mlflc = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name=f"test_precision_at_{k}",
        nest_trials=True
    )

    logger = logging.getLogger(__name__)
    logger.info("Optimizing model hyperparams")
    # increase trials for better success (>100)
    study.optimize(fun_objective, n_trials=10, callbacks=[mlflc])

    # storing best value and the model
    logger.info(f"Training best model (params: {study.best_params})")
    n_components = study.best_params["n_components"]
    # logging new parameter in MLFlow. TODO: how to do this better in the framework?
    # Hence, there are two n_components in the framework when using this node
    # Note that 'model.n_components' has already been logged even before
    #   this node ran (through kedro-mlflow hooks).
    mlflow.log_param("n_components", n_components)

    warp_model, test_prec, train_prec = train_model(
        train, test, eval_train, sp_item_feats,
        random_seed, epochs, k, n_components, loss)

    dict_metrics = {f"train_precision_at_{k}": {"value": train_prec, "step": 0},
                    f"test_precision_at_{k}": {"value": test_prec, "step": 0}}

    item_biases, item_factors = warp_model.get_item_representations(
        features=sp_item_feats)
    user_biases, user_factors = warp_model.get_user_representations()

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

    scores = RecommenderUtils.produce_scores(
        item_factors, item_biases, user_factors, user_biases)
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

        list_dict_recos.append({"userId": idx_to_rid[idx], "recos": recos})

    return pd.DataFrame(list_dict_recos)


def build_index(item_factors, params: Dict):
    metric = params["metric"]
    n_trees = params["n_trees"]

    factors = item_factors.shape[1]
    # dot product index
    annoy_idx = AnnoyIndex(factors, metric)
    for i in range(item_factors.shape[0]):
        v = item_factors[i]
        annoy_idx.add_item(i, v)

    annoy_idx.build(n_trees)
    # save
    annoy_dataset = MlflowArtifactDataSet(data_set={
        "type": KedroAnnoyIndex,
        "filepath": INDEX_PATH,
        "embedding_length": factors,
        "metric": metric
    })
    annoy_dataset.save(data=annoy_idx)
    return annoy_dataset


def validate_index(kedro_annoy_dataset: KedroAnnoyIndex, idx_to_names: Dict):
    # 1558 = Dark Knight
    # 1042 = Ratatouille
    # 2196 = Spy who loved me
    # 1246 = Rambo
    # 818 = Rashomon
    # 2481 = The Haunting
    annoy_index = kedro_annoy_dataset.load()
    item_ids_for_sampling = [1558, 1042, 2196, 1246, 818, 2481]
    for item_id in item_ids_for_sampling:
        nearest_movies_annoy(item_id, annoy_index, idx_to_names)

    return kedro_annoy_dataset


def nearest_movies_annoy(item_id, index, idx_to_names, n=10):
    nn = index.get_nns_by_item(item_id, n)
    titles = [idx_to_names[i] for i in nn]
    related_items = "\n".join(titles)

    str_message = 'Closest to %s : \n' % idx_to_names[item_id]
    str_message += related_items

    logger = logging.getLogger(__name__)
    logger.info(str_message)


def upload_to_mlflow(kedro_annoy_dataset: KedroAnnoyIndex, idx_to_names: Dict,
                     item_factors: np.array, user_factors: np.array, item_biases: np.array,
                     user_biases: np.array, item_rank: pd.DataFrame, params: Dict):
    """'Passthrough functions to enable uploading to mlflow for deployment

    Args:
        kedro_annoy_dataset (KedroAnnoyIndex): _description_
        idx_to_names (Dict): _description_
        item_factors (np.array): _description_
        user_factors (np.array): _description_
        item_biases (np.array): _description_
        user_biases (np.array): _description_
        item_rank (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    # store temporarily the different artifacts so mlflow facilitates the logging
    with tempfile.NamedTemporaryFile(prefix="idx_to_names-") as idx_to_names_file, \
            tempfile.NamedTemporaryFile(prefix="item_factors_file-") as item_factors_file, \
            tempfile.NamedTemporaryFile(prefix="user_factors_file-") as user_factors_file, \
            tempfile.NamedTemporaryFile(prefix="item_biases_file-") as item_biases_file, \
            tempfile.NamedTemporaryFile(prefix="user_biases_file-") as user_biases_file, \
            tempfile.NamedTemporaryFile(prefix="params_file-") as params_file, \
            tempfile.NamedTemporaryFile(prefix="item_rank_file-", mode='w') as item_rank_file:

        # save in temporary files
        cloudpickle.dump(idx_to_names, idx_to_names_file)
        cloudpickle.dump(params, params_file)
        np.save(item_factors_file, item_factors)
        np.save(user_factors_file, user_factors)
        np.save(item_biases_file, item_biases)
        np.save(user_biases_file, user_biases)
        item_rank.to_csv(item_rank_file, index=False)

        # flush files to disk
        idx_to_names_file.flush()
        item_factors_file.flush()
        user_factors_file.flush()
        item_biases_file.flush()
        user_biases_file.flush()
        params_file.flush()
        item_rank_file.flush()

        artifacts = {
            "annoy_index" : kedro_annoy_dataset._filepath.as_posix(),
            "idx_to_names": idx_to_names_file.name,
            "item_factors": item_factors_file.name,
            "user_factors": user_factors_file.name,
            "item_biases": item_biases_file.name,
            "user_biases": user_biases_file.name,
            "item_rank": item_rank_file.name,
            "params": params_file.name
        }

        mlflow_model_logger = MlflowModelLoggerDataSet(
            flavor="mlflow.pyfunc",
            pyfunc_workflow="python_model",
            save_args={
                "artifacts": artifacts
            },
        )
        mlflow_model_logger.save(KedroMLFlowLightFM())

        # comment this out to test uploaded artifacts
        # __test_artifacts(idx_to_names, params, item_rank, item_factors,
        #                  user_factors, item_biases, user_biases)


def __test_artifacts(idx_to_names, params, item_rank, item_factors, user_factors, item_biases, user_biases):
    # persist names
    idx_to_names_file = tempfile.NamedTemporaryFile(prefix="idx_to_names-")
    cloudpickle.dump(idx_to_names, idx_to_names_file)
    print(cloudpickle.load(open(idx_to_names_file.name, 'rb')))

    # persist params
    params_file = tempfile.NamedTemporaryFile(prefix="params_file-", delete=False)
    cloudpickle.dump(params, params_file)
    params_file.flush()
    print(cloudpickle.load(open(params_file.name, 'rb')))

    # persist item_rank_file
    g = tempfile.NamedTemporaryFile(prefix="item_rank_file-")
    item_rank.to_csv(g, index=False)
    print(pd.read_csv(g.name)[:2])

    # persist
    print("item factors")
    f = tempfile.NamedTemporaryFile(prefix="item_factors_file-")
    np.save(f, item_factors)
    print(np.load(f.name))

    print("user factors")
    f = tempfile.NamedTemporaryFile(prefix="user_factors_file-")
    np.save(f, user_factors)
    print(np.load(f.name))

    print("item biases")
    f = tempfile.NamedTemporaryFile(prefix="item_biases_file-")
    np.save(f, item_biases)
    print(np.load(f.name))

    print("user biases")
    f = tempfile.NamedTemporaryFile(prefix="user_biases_file-")
    np.save(f, user_biases)
    print(np.load(f.name))


def test_uploaded_artifact():
    # TODO: load the mlflow model, then serve
    pass
