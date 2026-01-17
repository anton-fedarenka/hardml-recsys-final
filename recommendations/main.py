import random
from typing import List
import os
import time
import json

# import pandas as pd

import numpy as np
import redis
from fastapi import FastAPI
from qdrant_client import QdrantClient

from models import InteractEvent, RecommendationsResponse, NewItemsEvent
# from watched_filter import WatchedFilter

app = FastAPI()

# redis_connection = redis.Redis('redis')
redis_connection = redis.Redis('localhost') 
qdrant_connection = QdrantClient("localhost", port=6333)
# watched_filter = WatchedFilter()

unique_item_ids = set()
movie_mapping = None
movie_inv_mapping = None

EPSILON = 0.05
TOP_K = 10
diversity_coeff = 0.4


@app.get('/healthcheck')
def healthcheck():
    return 200


@app.get('/cleanup')
def cleanup():
    global unique_item_ids
    unique_item_ids = set()
    try:
        # redis_connection.delete('*')
        # redis_connection.json().delete('*')
        # redis_connection.flushall()
        redis_connection.json().delete('thompson_top')
        redis_connection.json().delete('movie_ids_mapping')
        redis_connection.json().set('clear_bandit','.', True)
    except redis.exceptions.ConnectionError:
        pass

    if os.path.exists('./data/interactions.csv'):
        # suffix = int(time.time())
        suffix = time.strftime("%H_%M_%S", time.localtime())
        os.rename('./data/interactions.csv', f'./data/interactions_{suffix}.csv')
    return 200


@app.post('/add_items')
def add_movie(request: NewItemsEvent):
    global unique_item_ids
    global movie_mapping
    global movie_inv_mapping

    # Treat request as dataframe and drop duplicates items. 
    # This part did not test
    # df = pd.DataFrame(request.model_dump())
    # df.drop_duplicates(inplace=True)

    # Write down the added items to the file for further testing
    suffix = time.strftime("%H_%M_%S", time.localtime())
    filepath = f'./data/added_items/item_batch_{suffix}.json'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath,'w') as f:
        json.dump(request.model_dump(), f)

    movie_mapping = {movie_id: i for i, movie_id in enumerate(request.item_ids)}
    movie_inv_mapping = {i: movie_id for i, movie_id in enumerate(request.item_ids)}
    redis_connection.json().set('movie_ids_mapping','.',movie_mapping)

    for item_id in request.item_ids:
        unique_item_ids.add(item_id)

    return 200


@app.get('/recs/{user_id}')
def get_recs(user_id: str):
    print(f'looks recommendations for user {user_id}')
    global unique_item_ids
    item_ids = []
    user_mapping = redis_connection.json().get('user_mapping')

    print(f'--->>> stored user_mapping is {user_mapping}')

    history = redis_connection.json().get(f'user_history_{user_id}') or []

    if user_mapping and user_id in user_mapping:
        print(" ==>>> Retrive personal recommentdations <<<=== ")
        try: 
            item_ids = _get_personal_recs(user_id=user_id, k = TOP_K + len(history))
        except:
            print(f'Exception: Fail to retrieve personal recommendations for user {user_id}')

        print(f' ~~~~~~ found {len(item_ids)} items: {item_ids} ')

    print(f' ------ retrieved personal items = {item_ids}')

    if len(item_ids) < TOP_K:
        print(">>>>  Cold user. Getting thompson top <<<,")
        try:
            top_items = redis_connection.json().get('thompson_top')
            item_ids.extend(top_items)
        except redis.exceptions.ConnectionError:
            print(f'Exception while Redis connecting: Conncection fail while retrieve thompson top recommendations for user {user_id}')
        
    if len(item_ids) < TOP_K: # or random.random() < EPSILON:
        print(">>>> !!! FAIL TO GET ANY PRECALCULATED RECS! RANDOM CHOICE. <<<< ")
        item_ids += np.random.choice(list(unique_item_ids), size=20, replace=False).tolist()

    item_ids = item_ids[:TOP_K]
    history.extend(item_ids)
    redis_connection.json().set(f'user_history_{user_id}','.',history)
    print(f' ------ items just before return = {item_ids}')

    return RecommendationsResponse(item_ids=item_ids)

# def _get_user_history(user_id: str) -> List[int]:  
#     hist = redis_connection.json().get(f'user_history_{user_id}') or []
#     return hist

def _get_personal_recs(user_id: str, user_history: List[int], k: int = TOP_K, min_score: float = -float("inf")): 
    user_emb = redis_connection.json().get(f'user_emb_{user_id}')
    closest_points = qdrant_connection.query_points(
        collection_name = 'movie_embs',
        query = user_emb,
        limit = k
    ).points

    rec_indices = [s.id for s in closest_points if movie_inv_mapping[s.id] not in user_history and s.score > min_score]
    rec_scores = np.array([s.score for s in closest_points if s.id in rec_indices])

    if len(rec_indices) > TOP_K:
        rec_vecs = np.array([
            v.vector for v in 
            qdrant_connection.retrieve(
                collection_name='movie_embs',
                ids=rec_indices,
                with_vectors=True
            )
        ])
        rec_vecs = np.linalg.norm(rec_vecs)
        divers = inner_diversity(rec_vecs)
        rec_scores += diversity_coeff * divers #inner_diversity(rec_vecs)

        if len(user_history) > 0:
            hist_vecs = np.array([
                v.vector for v in
                qdrant_connection.retrieve(
                    collection_name='movie_embs',
                    ids=history,
                    with_vectors=True
                )
            ])
            hist_vecs = np.linalg.norm(hist_vecs)
            unexpect =  unexpectedness(rec_vecs, hist_vecs)
            rec_scores *= unexpect # unexpectedness(rec_vecs, hist_vecs)

    top_args = rec_scores.argsort()[-TOP_K:][::-1]
    
    total_rec_divers = 2 * divers[top_args].mean() # There are doubts about this formula
    total_rec_unexpect = unexpect[top_args].mean()

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



# @app.post('/interact')
# async def interact(request: InteractEvent):
#     watched_filter.add(request.user_id, request.item_id)
#     return 200
