"""
Contains wrappers for MLFlow pyfunc model
"""

import mlflow.pyfunc
from lightfm import LightFM
import pickle

USER_ID_KEY = "userId"
ITEM_ID_KEY = "itemIds"

# Define the model class


class KedroMLFLowLightFM(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # LightFM model
        self.reco_model = pickle.load(open(context["model_path"], 'rb'))
        # sorted list of items (column index form)
        self.popularity_index = pickle.load(open(context["popularity_index"], 'rb'))
        # ANNOY Index

    def predict(self, context, model_input):
        """Prediction

        Args:
            model_input (dict, list): if dict, then userid and itemids are expected. 
            if list, then it's a list of item ids

        Returns:
            _type_: _description_
        """
        # (1) if dataframe contains user id
        if isinstance(model_input, dict) and self.__validate_as_warm_user_prediction(model_input):
            reco_model = self.reco_model
            scores = reco_model.predict(users_coo, items_coo, 
                            item_features=sp_item_feats, 
                            user_features=sp_user_feats, num_threads=4)
        elif isinstance(model_input, list) and len(model_input[ITEM_ID_KEY]) > 0:
            pass
        else:
            raise ValueError("Please input either dict or list with the correct keys")

    def __validate_as_warm_user_prediction(self, model_input):
        # correct keys
        is_warm_user = USER_ID_KEY in model_input
        is_warm_user = is_warm_user and ITEM_ID_KEY in model_input
        # correct value types
        is_warm_user = is_warm_user and isinstance(model_input[USER_ID_KEY], int)
        is_warm_user = is_warm_user and isinstance(model_input[ITEM_ID_KEY], list)
        # more than one
        is_warm_user = is_warm_user and len(model_input[ITEM_ID_KEY]) > 0
        return is_warm_user