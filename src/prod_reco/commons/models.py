"""
Contains wrappers for MLFlow pyfunc model
"""

import mlflow.pyfunc
import cloudpickle
import numpy as np
import pandas as pd
from prod_reco.commons.datasets import KedroAnnoyIndex
from prod_reco.commons.recommender_utils import ITEM_ID, USER_ID, RecommenderUtils
from annoy import AnnoyIndex

# TODO: clean up hardcoded index values
n = 5
N_RECOS = 10
ITEM_POSITIONAL_INDEX_NAME = "idx"
NUM_USERS_RANK_SORT_NAME = "num_users"
MOVIE_NAME = "movieName"
# Define the model class


class KedroMLFlowLightFM(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        contents = context.artifacts

        self.idx_to_names = cloudpickle.load(open(contents["idx_to_names"], 'rb'))
        self.item_factors = np.load(contents["item_factors"])
        self.user_factors = np.load(contents["user_factors"])
        self.item_biases = np.load(contents["item_biases"])
        self.user_biases = np.load(contents["user_biases"])
        self.item_rank = pd.read_csv(contents["item_rank"])

        # create annoy index
        params = cloudpickle.load(open(contents["params"], 'rb'))
        metric = params["metric"]
        n_trees = params["n_trees"]

        factors = self.item_factors.shape[1]
        # dot product index
        annoy_index = AnnoyIndex(factors, metric)
        for i in range(self.item_factors.shape[0]):
            v = self.item_factors[i]
            annoy_index.add_item(i, v)

        annoy_index.build(n_trees)
        self.annoy_index = annoy_index

    def predict(self, context, model_input : pd.DataFrame):
        """Prediction

        Args:
            model_input (pd.DataFrame): if contains userid then prediction is warm start
                                        if not, then prediction is item-based only

        Returns:
            _type_: _description_
        """

        # (1) if dataframe contains user id
        if USER_ID in model_input:
            # group every user to item id
            users_to_items = model_input.groupby(USER_ID)[ITEM_ID].unique()
            list_recos = []
            for user_id, items in users_to_items.iteritems():
                # get nearest neighbors
                list_nn = []
                for item_id in items:
                    list_nn.extend(self.annoy_index.get_nns_by_item(item_id, n))

                # get indexes of items
                item_factors = self.item_factors[list_nn]
                item_biases = self.item_biases[list_nn]

                # get index of user
                user_factors = self.user_factors[user_id].reshape(1, -1)
                user_bias = self.user_biases[user_id].reshape(1)

                # perform scoring
                scores = RecommenderUtils.produce_scores(
                    item_factors, item_biases, user_factors, user_bias)

                # argsort then reindex to old
                sorted_items = np.array(list_nn)[np.argsort(scores)][0]

                # get item names
                recos = [self.idx_to_names[v] for v in sorted_items][:N_RECOS]
                
                list_recos.append({USER_ID : user_id, "recos": recos})
            return list_recos

        elif ITEM_ID in model_input:
            items = model_input[ITEM_ID]
            # get nearest neighbors
            list_nn = []
            for item_id in items:
                list_nn.extend(self.annoy_index.get_nns_by_item(item_id, n))

            # get ranking (pandas)
            df_rank = self.item_rank.set_index(ITEM_POSITIONAL_INDEX_NAME)
            df_rank_subset = df_rank.loc[list_nn]
            df_rank_subset = df_rank_subset.sort_values(
                by=NUM_USERS_RANK_SORT_NAME, ascending=False)
            return df_rank_subset[MOVIE_NAME][:N_RECOS].tolist()
        else:
            raise ValueError("Please input either dict or list with the correct keys")

    # def __validate_as_warm_user_prediction(self, model_input):
    #     # correct keys
    #     is_warm_user = USER_ID_KEY in model_input
    #     is_warm_user = is_warm_user and ITEM_ID_KEY in model_input
    #     # correct value types
    #     is_warm_user = is_warm_user and isinstance(model_input[USER_ID_KEY], int)
    #     is_warm_user = is_warm_user and isinstance(model_input[ITEM_ID_KEY], list)
    #     # more than one
    #     is_warm_user = is_warm_user and len(model_input[ITEM_ID_KEY]) > 0
    #     return is_warm_user
