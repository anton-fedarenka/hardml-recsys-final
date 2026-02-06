import random
from typing import List, Any
import os
import sys
import time
import json
from dataclasses import asdict
from collections import deque

import numpy as np
import polars as pl
import redis
from fastapi import FastAPI
from qdrant_client import QdrantClient
from sklearn.preprocessing import normalize
from loguru import logger
# import mlflow

from models import InteractEvent, RecommendationsResponse, NewItemsEvent
from bandit import ThompsonSamplingBandit, create_bandit_instance
# from regular_pipeline.bandit import ThompsonSamplingBandit

app = FastAPI()

# --------- Docker settings: ----------------
redis_connection = redis.Redis('redis')
# qdrant_connection = QdrantClient("qdrant", port=6333)
# mlflow.set_tracking_uri("http://mlflow-server:8080")
logger.add('/logs/recommendations.log',enqueue=True, backtrace=False, colorize=False)
# -------------------------------------------

# --------- Local settings: -----------------
# redis_connection = redis.Redis('localhost') 
# qdrant_connection = QdrantClient("localhost", port=6333)
# logger.add('./logs/recommendations.log',enqueue=True, backtrace=False, colorize=True)
# -------------------------------------------
sys.tracebacklimit = 4

unique_item_ids = set()
movie_mapping = {}
movie_inv_mapping = {}

rec_history = {}
df_rec_history = None
df_hist_diversity = None

response_td = []
pers_response_td = []
top_response_td = []

tops = []

EPSILON = 0.05
TOP_K = 10
N_recs = 100
calc_diversity_flag = True
divers_coeff = 0.1
bandit_params  = {
    'alpha_weight': 1,
    'beta_weight': 1000 
}
top_bunch_num = 100

run_params = {
    'TOP_K': TOP_K,
    'N_recs': N_recs,
    'calc_diversity_flag': calc_diversity_flag,
    'divers_coeff': divers_coeff,
    'bandit_params': bandit_params,
    'top_bunch_num': top_bunch_num
}
with open('data/run_params.json', 'w') as f: 
    json.dump(run_params, f)

phase = 0
batch = 0

# mlflow_experiment_id = mlflow.create_experiment('experiment_1')
# with mlflow.start_run() as run:
#     mlflow_run_id = run.info.run_id
#     mlflow.log_params(bandit_params)
#     mlflow.log_param('N_recs', N_recs)
#     mlflow.log_param('calc_diversity_flag', calc_diversity_flag)
#     mlflow.log_param('divers_coeff', divers_coeff)


@app.get('/healthcheck')
def healthcheck():
    logger.warning("!!! HEALTH CHECK !!!")
    return 200


@app.get('/cleanup')
def cleanup():
    global unique_item_ids
    global df_rec_history
    global df_hist_diversity
    global rec_history
    global movie_mapping
    global movie_inv_mapping
    global phase
    global batch

    logger.warning("Cleaning procedure has been called!..")

    # if phase > 0:
    #     calc_metrics()
    
    phase += 1
    batch = 0
    
    unique_item_ids = set()
    rec_history = {}
    movie_mapping = {}
    movie_inv_mapping = {}
    try:
        redis_connection.flushdb() # Delete all keys from redis
        redis_connection.set('clear', 1)
    except redis.exceptions.ConnectionError:
        logger.exception("Redis connection failure while cleaning.")

    # try:
    #     qdrant_connection.delete_collection(collection_name="movie_embs")
    # except Exception as e: 
    #     logger.exception(f'Exception occurs when qdrant collection deleting: {e}')

    try: 
        if df_rec_history is not None: 
            # Saving dataframes to csv files:
            os.makedirs('./data/history', exist_ok=True)
            #suffix = time.strftime("%H_%M_%S", time.localtime())
            # logger.warning(f'!!!>>> df_rec_history = {df_rec_history}')
            df_rec_history.write_csv(f'./data/history/recommendations_ph_{phase}.csv')
            if df_hist_diversity is not None: 
                df_hist_diversity.write_csv(f'./data/history/diversity_ph_{phase}.csv')
            df_rec_history = None
            df_hist_diversity = None
    except Exception as e: 
        logger.exception(f'Exception occurs while history dataframes saving: {e}')
        
    if os.path.exists('./data/interactions.csv'):
        #suffix = time.strftime("%H_%M_%S", time.localtime())
        os.rename('./data/interactions.csv', f'./data/interactions_ph_{phase}.csv')

    logger.warning('>>> !Cleaned UP! <<<')
    return 200


@app.post('/add_items')
def add_movie(request: NewItemsEvent):
    global unique_item_ids
    global movie_mapping
    global movie_inv_mapping
    global batch

    logger.warning("Add new data")

    batch += 1
    if phase > 0: 
        filepath = f'./data/added_items/items_ph_{phase}_bt_{batch}.json'
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath,'w') as f:
            json.dump(request.model_dump(), f)

    n = len(movie_mapping)
    movie_mapping.update(
        {movie_id: n + i for i, movie_id in enumerate(request.item_ids)}
        )
    movie_inv_mapping.update(
        {n + i: movie_id for i, movie_id in enumerate(request.item_ids)}
        )
    redis_connection.json().set('movie_mapping','.',movie_mapping)
    redis_connection.set('top_updated', 0)

    for item_id in request.item_ids:
        unique_item_ids.add(item_id)

    return 200


@app.get('/recs/{user_id}')
def get_recs(user_id: str):
    global unique_item_ids
    global rec_history
    global df_rec_history
    global df_hist_diversity
    global response_td
    global top_response_td
    global pers_response_td
    global tops

    item_ids = []
    # user_mapping = redis_connection.json().get('user_mapping')

    # history = redis_connection.json().get(f'user_history_{user_id}') or []
    history = rec_history.get(user_id, [])
    logger.info(f'-- Looks recommendations for user {user_id} with history length {len(history)} --')

    # try:
    #     # personal_items = redis_connection.json().get(f'recs_user_{user_id}')
    #     top_items = redis_connection.json().get(f'thompson_top')
    # except redis.exceptions.ConnectionError:
    #     print(f'Exception while Redis connecting: Conncection fail while retrieving recommendations for user {user_id}')

    top_updated = int(redis_connection.get('top_updated'))
    if top_updated:
        try:
            tops = deque(redis_connection.json().get('thompson_top'))
            redis_connection.set('top_updated', 0)
        except redis.exceptions.ConnectionError:
            print(f'Exception while Redis connecting: Conncection fail while getting top recs for user {user_id}')
    
    if len(tops) > 0:
        top_items = tops[0]
        tops.rotate(1)
        item_ids = [item for item in top_items if item not in history]
        logger.info(f' ===== Use T. TOP for recs! Rest of tops is {len(tops)} ===== ' )
    else:
        # top_items = []
        # logger.warning('<<<<<<< !!! TOPS COLLECTION IS EMPTY !!! >>>>>>>>>>')
        logger.info(f'Thompson top item is empty for user {user_id}')

    # if personal_items is None: 
    #     personal_items = [] #if personal_items is None else personal_items
    #     logger.info(f'Personal item is empty for user {user_id}')
    # else:
    #     logger.info(f'N_pers = {len(personal_items)} pesonal items are retrieved for user {user_id}')
    
    # if top_items is None: 
    #     top_items = [] # if top_items is None else top_items
    #     logger.info(f'Thompson top item is empty for user {user_id}')
    # else:
    #     logger.info(f'N_top = {len(top_items)} top items are retrieved for user {user_id}')

    # item_ids = [item for item in personal_items + top_items if item not in history]

    # if user_mapping and user_id in user_mapping:
    #     pers_t_start = time.time()
    #     logger.info(f"Retriving personal recommentdations for user {user_id}")
    #     try: 
    #         # item_ids = _get_personal_recs(user_id=user_id, user_history= history, k = 5*TOP_K, min_score=0)
    #         item_ids = redis_connection.json().get(f'recs_user_{user_id}')
    #         logger.info(f'Retrieved {len(item_ids)} personal recommendations for user {user_id}')
    #     except Exception as e:
    #         logger.exception(f'Fail to retrieve personal recommendations for user {user_id}. Exception: {e}')
    #         item_ids = []            
    #     pers_response_td.append(time.time() - pers_t_start)

    # if len(item_ids) < TOP_K:
    #     logger.info(f"Getting thompson top for user {user_id}")
    #     top_t_start = time.time()
    #     try:
    #         top_items = redis_connection.json().get('thompson_top')
    #     except redis.exceptions.ConnectionError:
    #         print(f'Exception while Redis connecting: Conncection fail while retrieve thompson top recommendations for user {user_id}')
    #     if top_items is not None:
    #         item_ids.extend(top_items)
        
    #     if len(history) > 0: 
    #         item_ids = [item for item in item_ids if item not in history]
    #     # try:
    #     #     top_indeces = _get_thompson_top(TOP_K + len(history))
    #     #     top_items = [movie_inv_mapping[item] for item in top_indeces if item not in history]
    #     #     item_ids.extend(top_items)
    #     # except Exception as e: 
    #     #     logger.exception(f'Fail to get thompson top for user {user_id}. Exception: {e}')
    #     top_response_td.append(time.time() - top_t_start)
        
    if len(item_ids) < TOP_K: # or random.random() < EPSILON:
        logger.warning(f"Fail to get enough precalculated recommendations for user {user_id}; number of the items retrieved: {len(item_ids)}! Random choice...")
        rng = np.random.default_rng()
        random_recs = rng.choice(list(unique_item_ids), size=TOP_K + len(history), replace=False).tolist()
        item_ids += [item for item in random_recs if item not in history]

    item_ids = item_ids[:TOP_K]
    history.extend(item_ids)
    rec_history[user_id] = history

    # try:
    #     df_rec_history = _update_history_df(user_id=user_id, recs=item_ids, time_mark=time.time(), df_hist=df_rec_history)
    # except Exception: 
    #     logger.exception("Exception while df_rec_history dataframe updating")

    # print(f'df_rec_history = {df_rec_history}')
    # print(f'df_hist_diversity = {df_hist_diversity}')
    # os.makedirs('./data/history', exist_ok=True)
    # suffix = time.strftime("%H_%M_%S", time.localtime())
    # df_rec_history.write_csv(f'./data/history/recommendations_{suffix}.csv')
    
    # response_td.append(time.time() - t_response_start)

    return RecommendationsResponse(item_ids=item_ids)


@logger.catch
def _get_thompson_top(k: int) -> List[int]:  
    try: 
        bandit_state = redis_connection.json().get('bandit_state')
        if type(bandit_state) == str:
            logger.warning('Bandit state type is str')
            bandit_state = json.loads(bandit_state)
    except redis.exceptions.ConnectionError:
        logger.exception(f'Exception while Redis connecting: Conncection fail while retrieve current thompson bandit state')
    if bandit_state is None: 
        logger.warning('!!!!... Bandit state is empty!!! Default settings for bandit are used!')
        local_bandit = create_bandit_instance(n_arms=len(movie_mapping), **bandit_params)
        redis_connection.json().set('bandit_state', '.', asdict(local_bandit))
    else: 
        local_bandit = ThompsonSamplingBandit(**bandit_state)
    return local_bandit.get_top_indices(k)


@logger.catch
def _get_personal_recs(user_id: str, user_history: List[int], k: int = TOP_K, min_score: float = -float("inf")): 
    global df_hist_diversity
    user_emb = redis_connection.json().get(f'user_emb_{user_id}')
    try:
        closest_points = qdrant_connection.query_points(
            collection_name = 'movie_embs',
            query = user_emb,
            limit = k + len(user_history)
        ).points
    except Exception as e: 
        logger.exception(f'Error while the closest points retrieving: {e}')
        return []

    rec_indices = [s.id for s in closest_points if movie_inv_mapping[s.id] not in user_history and s.score > min_score]
    if len(rec_indices) == 0:
        logger.warning('No one personal recommendation has been retrieved!')
        return []
    elif len(rec_indices) == 1: 
        return [movie_inv_mapping[rec_indices[0]]]

    rec_scores = np.array([s.score for s in closest_points if s.id in rec_indices])
    divers = np.zeros(rec_scores.size)
    unexpect = np.zeros(rec_scores.size)

    # If retrieved number of recs is enough, can apply for the diversity correction: 
    # if len(rec_indices) > TOP_K:
    # logger.info(f'Diversity calculation')
    rec_vecs = []
    hist_vecs = []
    try:
        rec_vecs = np.array([
            v.vector for v in 
            qdrant_connection.retrieve(
                collection_name='movie_embs',
                ids=rec_indices,
                with_vectors=True
            )
        ])
    except Exception as e: 
        logger.exception(f'Exception occured during retrieving vectors for diversity calculations: {e}')
        rec_vecs = [] 

    if len(rec_vecs) > 0:
        # rec_vecs = (rec_vecs.T/np.linalg.norm(rec_vecs, axis=1)).T
        rec_vecs = normalize(rec_vecs, axis=1)
        divers = inner_diversity(rec_vecs)
        rec_scores += diversity_coeff * divers # May be unnecessary in the case of len(rec_indeces) < TOP_K and could redece NDCG@k?... 
    
    # If there is a user history, the unexpectedness value can also be taken in account:
    # logger.info('Enexpectedness calculation')
    try:
        hist_vecs = np.array([
            v.vector for v in
            qdrant_connection.retrieve(
                collection_name='movie_embs',
                ids=[movie_mapping[i] for i in user_history],
                with_vectors=True
            )
        ])
    except Exception as e:
        logger.exception(f'Exception occured during retrieving vectors for unexpectedness calculations: {e}')
        hist_vecs = []

    if len(hist_vecs) > 0 and len(rec_vecs) > 0: 
        # hist_vecs = (hist_vecs.T/np.linalg.norm(hist_vecs, axis=1)).T
        hist_vecs = normalize(hist_vecs, axis=1)
        unexpect =  unexpectedness(rec_vecs, hist_vecs)
        rec_scores *= unexpect # unexpectedness(rec_vecs, hist_vecs)

    top_args = rec_scores.argsort()[-TOP_K:][::-1]
    
    total_rec_divers = 2 * divers[top_args].mean() # There are doubts about this formula
    total_rec_unexpect = unexpect[top_args].mean() 

    df = pl.DataFrame({
            'user_id': user_id,
            'diversity': total_rec_divers,
            'unexpect': total_rec_unexpect,
            'timestamp': time.time()
        })
    if df_hist_diversity is None: 
        df_hist_diversity = df
    else:
        df_hist_diversity = pl.concat([df_hist_diversity, df])

    return [movie_inv_mapping[rec_indices[i]] for i in top_args]

    
def unexpectedness(recs: np.ndarray, hist: np.ndarray):
    '''
    Calculate so called Unexpectedness: the distance between recommendation and history of user
    '''
    sims = recs @ hist.T 
    return  (1 - sims).sum(axis=1)/sims.shape[1] # !!! В нормировочном знаменателе есть сомнения!


def inner_diversity(vectors: np.ndarray):
    '''
    Calculate the diversity between the objects in recommedation
    '''
    sims = vectors @ vectors.T
    return (1 - sims).sum(axis=1)/sims.shape[1]
    

@logger.catch
def _update_history_df(user_id, recs, time_mark, df_hist = None): 
    new_data = pl.DataFrame(
        {
            'user': user_id, 
            'item_id': recs, 
            'timestamp':time_mark
        }
        ).with_columns(pl.col('item_id').cast(pl.Utf8))
    if df_hist is None:
        df = new_data
    else:
        df = pl.concat([df_hist, new_data])
    return df

def precision(recs: List[Any], likes: List[Any]):
    if len(likes) == 0:
        return 0
    if len(recs) == 0:
        # logger.warning("Exception while precision calculation: list of recommendations is empty!")
        return None
    return len(set(recs).intersection(set(likes)))/len(recs) # len(y_rec[:k])


# @app.get('/calc_metrics')
@logger.catch
def calc_metrics(): 
    logger.info('.... Calculation of metrics.... ')
    # t_start = time.time()
    if not os.path.exists('./data/interactions.csv'):
        logger.warning('Exception during metric calculation: file with interactions does not exist ')
        return # 200
    if not rec_history:
        logger.warning('Exception during metric calculation: recommendation history does not exist ')
        return # 200
    df = (
        pl.read_csv('./data/interactions.csv')
        .filter(pl.col('action') == 'like')
        .with_columns(pl.col('item_id').cast(pl.Utf8))
        .group_by('user_id')
        .agg(pl.col('item_id'))
    )
    metrics = {}
    precs = []
    for user_id, liked_items in df.rows():
        user_hist = rec_history.get(user_id, [])
        val = precision(user_hist, liked_items)
        if val is not None: 
            precs.append(val)
    precs.extend([0] * len(set(rec_history.keys()) - set(df['user_id'])))
    if len(precs) > 0: 
        metrics['precision'] = sum(precs)/len(precs)

    used_items = set(item for sublist in rec_history.values() for item in sublist)
    metrics['coverage'] = len(used_items)/len(unique_item_ids)    

    if df_hist_diversity is not None:
        metrics['diversity'] = df_hist_diversity['diversity'].mean() 
        metrics['unexpectedness'] = df_hist_diversity['unexpect'].mean()
    metrics['response_time'] = sum(response_td)/len(response_td) if len(response_td) > 0 else 0
    metrics['pers_resp_time'] = sum(pers_response_td)/len(pers_response_td) if len(pers_response_td) > 0 else 0
    metrics['top_resp_time'] = sum(top_response_td)/len(top_response_td) if len(top_response_td) > 0 else 0
    
    if any(v is None for v in metrics.values()):
        logger.warning(f'Unable to log metrics in mlflow: {metrics}')
        return # 200

    # with mlflow.start_run(run_id=mlflow_run_id) as run:
    #     mlflow.log_metrics(metrics)
    #     # mlflow.log_metric('calc_metric_time', time.time() - t_start)

    return # 200