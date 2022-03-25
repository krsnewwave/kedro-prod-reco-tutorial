from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import traceback

USER_ID = "userId"
ITEM_ID = "itemId"
RATING = "rating"
ITEM_NAME = "movieName"
TAGS = "tags"

class RecommenderUtils():
    def __init__(self, user_id = "userId", item_id = "itemId", rating="rating"):
        self.userId = user_id
        self.itemId = item_id
        self.rating = rating

    def split_ratings_transactions(self, ratings, k = 18, test_size = 0.2,):
        """Split movielens ratings data to 2, training and testing. Difference with users
        is that this gets train_size % for each splittable user's ratings
        Args:
            ratings (pandas DataFrame): movielens ratings
            k (int, optional): How many ratings to be candidate for splitting
            test_size: float or int
                If float, then the proportion of test samples. If int, then the absolute value
                of test samples. Uses sklearn's train_test_split.
        Returns:
            train_data
            test_data
        """
        
        cols = [self.userId, self.itemId, self.rating]
        users_to_rated = ratings.groupby(self.userId)[self.itemId].size()
        splittable_users = users_to_rated[users_to_rated > k].index.tolist()
        to_split = ratings[ratings[self.userId].isin(splittable_users)]
        pure_train = ratings[~ratings[self.userId].isin(splittable_users)].index

        test_indexes = []
        train_indexes = []
        train_indexes.append(pure_train)

        for name, group in to_split[cols].groupby(self.userId):
            X1, X2 = train_test_split(group, train_size=train_size)
            train_indexes.extend(X1.index.tolist())
            test_indexes.extend(X2.index.tolist())

        test_data = ratings[ratings.index.isin(test_indexes)]
        train_data = ratings[ratings.index.isin(train_indexes)]

        return train_data, test_data


    def split_movielens_users(self, ratings, k = 18, train_size = 0.75):
        """Split movielens ratings data to 2, training and testing. In this one,
        take train_size % of users and put them into the train set. The rest are
        put into the test set.
        Args:
            ratings (pandas DataFrame): movielens ratings
            k (int, optional): How many ratings to be candidate for splitting
            train_size (double, optional) : Float from 0 to 1
        Returns:
            train_data
            test_data
        """

        users_to_rated = ratings.groupby(self.userId)[self.itemId].size()
        splittable_users = users_to_rated[users_to_rated > k].index.tolist()
        train_users, test_users = train_test_split(splittable_users, train_size=train_size)
        pure_train = ratings[~ratings[self.userId].isin(splittable_users)].index.tolist()
        train_indexes = ratings[ratings[self.userId].isin(train_users)].index.tolist()
        test_indexes = ratings[ratings[self.userId].isin(test_users)].index.tolist()

        train_indexes.append(pure_train)

        test_data = ratings[ratings.index.isin(test_indexes)]
        train_data = ratings[ratings.index.isin(train_indexes)]

        return train_data, test_data
    
    def print_ratings_shape(self, ratings):
        num_users = ratings[self.userId].nunique()
        num_items = ratings[self.itemId].nunique()
        num_possible_combinations = num_users * num_items

        print("Number of users:", num_users)
        print("Number of items:", num_items)
        print("Number of rows:", ratings.shape)
        print("Sparsity:", ratings.shape[0] / float(num_possible_combinations))
        return ratings.shape[0], ratings.shape[0] / float(num_possible_combinations)
        
    def sparse_to_df(self, coo, id_to_uid=None, id_to_pid=None):
        """
        Converts sparse matrices to dataframe
        Params
        ------
        coo : scipy.sparse matrix in coo
            Interactions between users and items.
        id_to_uid : int
           Mapping row indexes to user ids
        id_to_pid : float
            Mapping column indexes to item ids
        """
        if id_to_uid is not None:
            return pd.DataFrame({self.userId: [id_to_uid[v] for v in coo.row], 
                                 self.itemId: [id_to_pid[v] for v in coo.col], self.rating: coo.data}
                             )[[self.userId, self.itemId, self.rating]].sort_values([self.userId, self.itemId]
                             ).reset_index(drop=True)
        else:
            return pd.DataFrame({self.userId: [v for v in coo.row], 
                                 self.itemId: [v for v in coo.col], self.rating: coo.data}
                             )[[self.userId, self.itemId, self.rating]].sort_values([self.userId, self.itemId]
                             ).reset_index(drop=True)

    
    @staticmethod
    def apk(list_actual, list_predicted, k=10):
        """
        Computes the precision at k.
        This function computes the precision at k between two lists of lists of
        items.
        """

        # TODO: remove items that are part of the training set
        list_precs = []
        for (actual, predicted) in zip(list_actual, list_predicted):
            if len(actual) > 0:
                prec = len(set(predicted).intersection(set(actual)))/float(min(len(actual), k))
                list_precs.append(prec)

        return np.mean(list_precs)
    
    @staticmethod
    def apk_list(actual, predicted, k=10):
        """
        Computes the average precision at k.
        This function computes the average precision at k between two lists of
        items.
        Parameters
        ----------
        actual : list
                 A list of elements that are to be predicted (order doesn't matter)
        predicted : list
                    A list of predicted elements (order does matter)
        k : int, optional
            The maximum number of predicted elements
        Returns
        -------
        score : double
                The average precision at k over the input lists
        """
        if len(predicted)>k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i,p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)

        # if not actual:
        if len(actual) == 0:
            return 0.0

        return score / min(len(actual), k)
    
    @staticmethod
    def mapk(actual, predicted, k=10):
        """
        Computes the mean average precision at k.
        This function computes the mean average prescision at k between two lists
        of lists of items.
        Parameters
        ----------
        actual : list
                 A list of lists of elements that are to be predicted 
                 (order doesn't matter in the lists)
        predicted : list
                    A list of lists of predicted elements
                    (order matters in the lists)
        k : int, optional
            The maximum number of predicted elements
        Returns
        -------
        score : double
                The mean average precision at k over the input lists
        """
        return np.mean([RecommenderUtils.apk_list(a,p,k) for a,p in zip(actual, predicted)])
        

    @staticmethod
    def covk(list_predicted, num_items_train, k=10):
        recos = np.array(list_predicted)[:, :k]
        unique_recos = len(pd.Series(np.array(recos).ravel()).unique())
        return unique_recos / num_items_train
    
    
    @staticmethod
    def ark(list_actual, list_predicted, k=10, do_minimum=False):
        """
        Computes the average recall at k.
        This function computes the average recall at k between two lists of lists of
        items.
        """

        # TODO: remove items that are part of the training set
        recalls = []
        for (actual, predicted) in zip(list_actual, list_predicted):
            if len(actual) > 0:
                if do_minimum:
                    recalls.append(len(set(predicted).intersection(set(actual))) / float(min(k, len(actual))))
                else:
                    recalls.append(len(set(predicted).intersection(set(actual))) / float(len(actual)))
        return np.mean(recalls)

    @staticmethod
    def personalization(predicted: List[list]) -> float:
        """
        Personalization measures recommendation similarity across users.
        A high score indicates good personalization (user's lists of recommendations are different).
        A low score indicates poor personalization (user's lists of recommendations are very similar).
        A model is "personalizing" well if the set of recommendations for each user is different.
        Parameters:
        ----------
        predicted : a list of lists
            Ordered predictions
            example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
        Returns:
        -------
            The personalization score for all recommendations.
        """

        def make_rec_matrix(predicted: List[list]) -> sp.csr_matrix:
            df = pd.DataFrame(data=predicted).reset_index().melt(
                id_vars='index', value_name='item',
            )
            df = df[['index', 'item']].pivot(index='index', columns='item', values='item')
            df = pd.notna(df)*1
            rec_matrix = sp.csr_matrix(df.values)
            return rec_matrix

        #create matrix for recommendations
        # print(pd.DataFrame(data=predicted).reset_index().melt(id_vars = 'index', value_name='item')[["index", "item"]].sort_values(by="index"))
        predicted = np.array(predicted)
        try:
            rec_matrix_sparse = make_rec_matrix(predicted)
        except Exception as e:
            traceback.print_exc()
            raise e

        #calculate similarity for every user's recommendation list
        similarity = cosine_similarity(X=rec_matrix_sparse, dense_output=False)

        #get indicies for upper right triangle w/o diagonal
        upper_right = np.triu_indices(similarity.shape[0], k=1)

        #calculate average similarity
        personalization = np.mean(similarity[upper_right])
        return 1-personalization
    
    @staticmethod
    def bootstrapped_personalization(list_recos, k, n_pers_sample, n_times_pers_sample):
        list_dict_persk = []
        for _ in range(n_times_pers_sample):
            random_indexes = np.random.choice(range(len(list_recos)), size=n_pers_sample, replace=False)
            list_recos_sample = np.array(list_recos)[random_indexes, :k]
            pers_k = RecommenderUtils.personalization(list_recos_sample)
            list_dict_persk.append(pers_k)
        return np.mean(list_dict_persk)
    
    @staticmethod
    def to_list_from_sparse(sp_data):
        np_test = []
        for user_id, row in enumerate(sp_data):
            true_mids = row.indices[row.data > 0].tolist()
            np_test.append(true_mids)
        return np_test
        
    @staticmethod
    def threshold_interactions_df(df, row_name, col_name, row_min, col_min):
        """Limit interactions df to minimum row and column interactions.
        Parameters
        ----------
        df : DataFrame
            DataFrame which contains a single row for each interaction between
            two entities. Typically, the two entities are a user and an item.
        row_name : str
            Name of column in df which corresponds to the eventual row in the
            interactions matrix.
        col_name : str
            Name of column in df which corresponds to the eventual column in the
            interactions matrix.
        row_min : int
            Minimum number of interactions that the row entity has had with
            distinct column entities.
        col_min : int
            Minimum number of interactions that the column entity has had with
            distinct row entities.
        Returns
        -------
        df : DataFrame
            Thresholded version of the input df. Order of rows is not preserved.
        Examples
        --------
        df looks like:
        user_id | item_id
        =================
          1001  |  2002
          1001  |  2004
          1002  |  2002
        thus, row_name = 'user_id', and col_name = 'item_id'
        If we were to set row_min = 2 and col_min = 1, then the returned df would
        look like
        user_id | item_id
        =================
          1001  |  2002
          1001  |  2004
        """

        n_rows = df[row_name].unique().shape[0]
        n_cols = df[col_name].unique().shape[0]
        sparsity = float(df.shape[0]) / float(n_rows*n_cols) * 100
        print('Starting interactions info')
        print('Number of rows: {}'.format(n_rows))
        print('Number of cols: {}'.format(n_cols))
        print('Sparsity: {:4.3f}%'.format(sparsity))

        done = False
        while not done:
            starting_shape = df.shape[0]
            col_counts = df.groupby(row_name)[col_name].count()
            df = df[~df[row_name].isin(col_counts[col_counts < col_min].index.tolist())]
            row_counts = df.groupby(col_name)[row_name].count()
            df = df[~df[col_name].isin(row_counts[row_counts < row_min].index.tolist())]
            ending_shape = df.shape[0]
            if starting_shape == ending_shape:
                done = True

        n_rows = df[row_name].unique().shape[0]
        n_cols = df[col_name].unique().shape[0]
        sparsity = float(df.shape[0]) / float(n_rows*n_cols) * 100
        print('Ending interactions info')
        print('Number of rows: {}'.format(n_rows))
        print('Number of columns: {}'.format(n_cols))
        print('Sparsity: {:4.3f}%'.format(sparsity))
        return df


    @staticmethod
    def get_df_matrix_mappings(df, row_name, col_name):
        """Map entities in interactions df to row and column indices
        Parameters
        ----------
        df : DataFrame
            Interactions DataFrame.
        row_name : str
            Name of column in df which contains row entities.
        col_name : str
            Name of column in df which contains column entities.
        Returns
        -------
        rid_to_idx : dict
            Maps row ID's to the row index in the eventual interactions matrix.
        idx_to_rid : dict
            Reverse of rid_to_idx. Maps row index to row ID.
        cid_to_idx : dict
            Same as rid_to_idx but for column ID's
        idx_to_cid : dict
        """


        # Create mappings
        rid_to_idx = {}
        idx_to_rid = {}
        for (idx, rid) in enumerate(df[row_name].unique().tolist()):
            rid_to_idx[rid] = idx
            idx_to_rid[idx] = rid

        cid_to_idx = {}
        idx_to_cid = {}
        for (idx, cid) in enumerate(df[col_name].unique().tolist()):
            cid_to_idx[cid] = idx
            idx_to_cid[idx] = cid

        return rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid

    @staticmethod
    def df_to_matrix(df, row_name, col_name, interaction_var=None):
        """Take interactions dataframe and convert to a sparse matrix
        Parameters
        ----------
        df : DataFrame
        row_name : str
        col_name : str
        Returns
        -------
        interactions : sparse csr matrix
        rid_to_idx : dict
        idx_to_rid : dict
        cid_to_idx : dict
        idx_to_cid : dict
        """

        rid_to_idx, idx_to_rid,\
            cid_to_idx, idx_to_cid = RecommenderUtils.get_df_matrix_mappings(df,
                                                            row_name,
                                                            col_name)

        def map_ids(row, mapper):
            return mapper[row]

        I = df[row_name].apply(map_ids, args=[rid_to_idx]).values
        J = df[col_name].apply(map_ids, args=[cid_to_idx]).values
        if interaction_var:
            V = df[interaction_var]
        else:
            V = np.ones(I.shape[0])
        interactions = sp.coo_matrix((V, (I, J)), dtype=np.float64)
        interactions = interactions.tocsr()
        return interactions, rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid

    @staticmethod
    def train_test_split_sparse(interactions, split_count, split_fraction = 0.25, fraction=None):
        """
        Split recommendation data into train and test sets
        Params
        ------
        interactions : scipy.sparse matrix
            Interactions between users and items.
        split_count : int
            Number of user-item-interactions per user to become eligible to be part of the test set
        split_fraction : float / int
            Fraction of items from each eligible test users bucket of items to be transfered to the test set
            Put int if exact number
        fractions : float
            Fraction of users to split off some of their
            interactions into test set. If None, then all
            users are considered.
        """
        # Note: likely not the fastest way to do things below.
        train = interactions.copy().tocoo()
        test = sp.lil_matrix(train.shape)

        if fraction:
            try:
                user_index = np.random.choice(
                    np.where(np.bincount(train.row) >= split_count)[0],
                    replace=False,
                    size=np.int64(np.floor(fraction * train.shape[0]))
                ).tolist()
            except:
                print(('Not enough users with > {} '
                      'interactions for fraction of {}')\
                      .format(split_count, fraction))
                raise
        else:
            user_index = np.where(np.bincount(train.row) >= split_count)[0].tolist()

        train = train.tolil()

        for user in user_index:
            if isinstance(split_fraction, int):
                test_interactions = np.random.choice(interactions.getrow(user).indices,
                                            size=split_fraction,
                                            replace=False)
            elif isinstance(split_fraction, float):
                num_interactions = len(interactions.getrow(user).indices)
                test_interactions = np.random.choice(interactions.getrow(user).indices,
                                                size=np.int64(np.floor(num_interactions * split_fraction)),
                                                replace=False)
            else:
                raise ValueError("Input only int or float in split fraction")
            train[user, test_interactions] = 0.
            # These are just 1.0 right now
            test[user, test_interactions] = interactions[user, test_interactions]


        # Test and training are truly disjoint
        assert(train.multiply(test).nnz == 0)
        return train.tocsr(), test.tocsr(), user_index

    @staticmethod
    def produce_scores(item_factors, item_biases, user_factors, user_biases):
        # combine item_factors with biases for dot product
        item_factors = np.concatenate(
            (item_factors, np.ones((item_biases.shape[0], 1))), axis=1)
        item_factors = np.concatenate((item_factors, item_biases.reshape(-1, 1)), axis=-1)

        # combine user_factors with biases for dot product
        user_factors = np.concatenate((user_factors, user_biases.reshape(-1, 1)), axis=-1)
        user_factors = np.concatenate(
            (user_factors, np.ones((user_biases.shape[0], 1))), axis=1)

        scores = user_factors.dot(item_factors.T)
        return scores