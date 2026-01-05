import random
from typing import List
import os
import time

import numpy as np
import redis
from fastapi import FastAPI

from models import InteractEvent, RecommendationsResponse, NewItemsEvent
from watched_filter import WatchedFilter

app = FastAPI()

# redis_connection = redis.Redis('redis')
redis_connection = redis.Redis('localhost') 
watched_filter = WatchedFilter()

unique_item_ids = set()
EPSILON = 0.05


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
    redis_connection.json().set('movie_ids','.',request.item_ids)
    for item_id in request.item_ids:
        unique_item_ids.add(item_id)
    return 200


@app.get('/recs/{user_id}')
def get_recs(user_id: str):
    global unique_item_ids
    item_ids = None

    try:
        item_ids = redis_connection.json().get('thompson_top')
    except redis.exceptions.ConnectionError:
        item_ids = None

    if item_ids is None: # or random.random() < EPSILON:
        item_ids = np.random.choice(list(unique_item_ids), size=20, replace=False).tolist()
    return RecommendationsResponse(item_ids=item_ids)


@app.post('/interact')
async def interact(request: InteractEvent):
    watched_filter.add(request.user_id, request.item_id)
    return 200
