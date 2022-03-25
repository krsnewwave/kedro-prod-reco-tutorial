"""
Deployment phase, creating embeddings and index
"""
import logging
import tempfile
from sys import version_info
from typing import Dict

import cloudpickle
import mlflow
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from kedro_mlflow.io.models import MlflowModelLoggerDataSet
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import time

from prod_reco.commons.models import KedroMLFlowLightFM
from prod_reco.commons.recommender_utils import ITEM_ID, USER_ID, RecommenderUtils

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)
# TODO: hardcoded n recos
N_RECOS = 10
N_NEIGHBORS = 5
N_RECOS = 10
ITEM_POSITIONAL_INDEX_NAME = "idx"
NUM_USERS_RANK_SORT_NAME = "num_users"
MOVIE_NAME = "movie_name"


def build_nn(item_factors, params: Dict):
    metric = params["metric"]
    n_trees = params["n_trees"]

    # factors = item_factors.shape[1]
    # # dot product index
    # annoy_idx = AnnoyIndex(factors, metric)
    # for i in range(item_factors.shape[0]):
    #     v = item_factors[i]
    #     annoy_idx.add_item(i, v)

    # annoy_idx.build(n_trees)
    # return annoy_idx

    nn_model = NearestNeighbors(
        n_neighbors=n_trees, algorithm='brute', metric='cosine')
    nn_model.fit(item_factors)
    print(nn_model.n_samples_fit_)
    return nn_model


def validate_nn(nn_model: NearestNeighbors, idx_to_names: Dict, item_factors: np.array):
    # 1558 = Dark Knight
    # 1042 = Ratatouille
    # 2196 = Spy who loved me
    # 1246 = Rambo
    # 818 = Rashomon
    # 2481 = The Haunting
    item_ids_for_sampling = [1558, 1042, 2196, 1246, 818, 2481]
    for item_id in item_ids_for_sampling:
        query_item_factors = item_factors[item_id]
        nearest_movies(item_id, query_item_factors, nn_model, idx_to_names)


def nearest_movies(item_id: int, query_item_factors: np.array, index: NearestNeighbors, idx_to_names: Dict, n: int = 10):
    nn = index.kneighbors(X=query_item_factors.reshape(1, -1),
                          n_neighbors=n, return_distance=False)
    nn = nn[0]
    titles = [idx_to_names[i] for i in nn]
    related_items = "\n".join(titles)

    str_message = 'Closest to %s : \n' % idx_to_names[item_id]
    str_message += related_items

    logger = logging.getLogger(__name__)
    logger.info(str_message)


def recommend_node(reco_model, model_input, item_factors, item_biases,
                   user_factors, user_biases, idx_to_names, item_rank):
    if isinstance(model_input, dict) and __validate_as_warm_user_prediction(model_input):
        items = model_input[ITEM_ID]
        user_id = model_input[USER_ID]
        # get nearest neighbors
        list_nn = []
        for item_id in items:
            query_item_factors = item_factors[item_id]
            list_nn.append(nearest_movies(
                item_id, query_item_factors, reco_model, idx_to_names, N_NEIGHBORS))

        # get indexes of items
        item_factors = item_factors[items]
        item_biases = item_biases[items]

        # get index of user
        user_factors = user_factors[user_id]
        user_bias = user_biases[user_id]

        # perform scoring
        scores = RecommenderUtils.produce_scores(
            item_factors, item_biases, user_factors, user_bias)

        # argsort then reindex to old
        sorted_items = np.array(items)[np.argsort(scores)]

        # get item names
        recos = [idx_to_names[v] for v in sorted_items][:N_RECOS]
        return recos

    elif isinstance(model_input, list) and len(model_input[ITEM_ID]) > 0:
        items = model_input[ITEM_ID]
        # get nearest neighbors
        list_nn = []
        for item_id in items:
            query_item_factors = item_factors[item_id]
            list_nn.append(nearest_movies(
                item_id, query_item_factors, reco_model, idx_to_names, N_NEIGHBORS))

        # get ranking (pandas)
        df_rank = item_rank.set_index(ITEM_POSITIONAL_INDEX_NAME)
        df_rank_subset = df_rank.loc[list_nn]
        df_rank_subset = df_rank_subset.sort_values(
            by=NUM_USERS_RANK_SORT_NAME, ascending=False)
        return df_rank_subset[MOVIE_NAME][:N_RECOS].tolist()
    else:
        raise ValueError("Please input either dict or list with the correct keys")


def __validate_as_warm_user_prediction(model_input):
    # correct keys
    is_warm_user = USER_ID in model_input
    is_warm_user = is_warm_user and ITEM_ID in model_input
    # correct value types
    is_warm_user = is_warm_user and isinstance(model_input[USER_ID], int)
    is_warm_user = is_warm_user and isinstance(model_input[ITEM_ID], list)
    # more than one
    is_warm_user = is_warm_user and len(model_input[ITEM_ID]) > 0
    return is_warm_user


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
    return annoy_idx


def validate_index(annoy_index: AnnoyIndex, idx_to_names: Dict):
    # 1558 = Dark Knight
    # 1042 = Ratatouille
    # 2196 = Spy who loved me
    # 1246 = Rambo
    # 818 = Rashomon
    # 2481 = The Haunting
    item_ids_for_sampling = [1558, 1042, 2196, 1246, 818, 2481]
    for item_id in item_ids_for_sampling:
        nearest_movies_annoy(item_id, annoy_index, idx_to_names)


def nearest_movies_annoy(item_id, index, idx_to_names, n=10):
    nn = index.get_nns_by_item(item_id, n)
    titles = [idx_to_names[i] for i in nn]
    related_items = "\n".join(titles)

    str_message = 'Closest to %s : \n' % idx_to_names[item_id]
    str_message += related_items

    logger = logging.getLogger(__name__)
    logger.info(str_message)


def upload_to_mlflow(annoy_index: AnnoyIndex, idx_to_names: Dict,
                     item_factors: np.array, user_factors: np.array, item_biases: np.array,
                     user_biases: np.array, item_rank: pd.DataFrame, params: Dict):
    """'Passthrough functions to enable uploading to mlflow for deployment

    Args:
        annoy_index (AnnoyIndex): _description_
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
            "idx_to_names": idx_to_names_file.name,
            "item_factors": item_factors_file.name,
            "user_factors": user_factors_file.name,
            "item_biases": item_biases_file.name,
            "user_biases": user_biases_file.name,
            "item_rank": item_rank_file.name,
            "params": params_file.name
        }

        # mlflow.pyfunc.save_model(python_model=KedroMLFlowLightFM(), artifacts=artifacts)
        mlflow_model_logger = MlflowModelLoggerDataSet(
            flavor="mlflow.pyfunc",
            pyfunc_workflow="python_model",
            save_args={
                "conda_env": conda_env,
                "artifacts": artifacts
            },
        )
        mlflow_model_logger.save(KedroMLFlowLightFM())

        __test_artifacts(idx_to_names, params, item_rank, item_factors,
                        user_factors, item_biases, user_biases)


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


# TODO: build project, then pip to local
conda_env = None
# conda_env = {
#     'channels': ['defaults'],
#     'dependencies': [
#         'python={}'.format(PYTHON_VERSION),
#         'pip',
#         {
#             'pip': [
#                 f"mlflow=={mlflow.__version__}",
#                 "annoy",
#                 f"cloudpickle=={cloudpickle.__version__}",
#                 f"numpy=={np.__version__}",
#                 f"pandas=={pd.__version__}",
#                 "prod_reco"
#             ],
#         },
#     ],
#     'name': 'reco_env'
# }
