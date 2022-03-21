"""
Deployment phase, creating embeddings and index
"""
import logging
from annoy import AnnoyIndex
from typing import Dict
from prod_reco.commons.datasets import KedroAnnoyIndex


def build_index(item_factors, params:Dict):
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


def validate_index(annoy_index: KedroAnnoyIndex , idx_to_cid: Dict):
    # 1558 = Dark Knight
    # 1042 = Ratatouille
    # 2196 = Spy who loved me
    # 1246 = Rambo
    # 818 = Rashomon
    # 2481 = The Haunting
    item_ids_for_sampling = [1558, 1042, 2196, 1246, 818, 2481]
    
    print(annoy_index)
    print(type(annoy_index))
    for item_id in item_ids_for_sampling:
        nearest_movies_annoy(item_id, annoy_index, idx_to_cid)


def nearest_movies_annoy(item_id, index, idx_to_names, n=10):
    nn = index.get_nns_by_item(item_id, n)
    titles = [idx_to_names[i] for i in nn]
    related_items = "\n".join(titles)
    
    str_message = 'Closest to %s : \n' % idx_to_names[item_id]
    str_message += related_items

    logger = logging.getLogger(__name__)
    logger.info(str_message)
