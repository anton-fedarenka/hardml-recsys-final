import asyncio
import json
import os.path
import sys
import time
from dataclasses import asdict

import numpy as np
import scipy.sparse as ss
import implicit
import polars as pl
import redis
import requests
import aio_pika
from sklearn.preprocessing import normalize
from aio_pika import Message
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, SearchRequest

from bandit import ThompsonSamplingBandit, create_bandit_instance

# --------- Docker settings: ----------------
redis_connection = redis.Redis('redis')
# qdrant_connection = QdrantClient("qdrant", port=6333)
rabbitmq_url = "amqp://guest:guest@rabbitmq/"
recommendation_service_url = "http://recommendations_dc:5001"
logger.add('/logs/regular_pipeline.log',enqueue=True, backtrace=False, colorize=False) # Docker version
# -------------------------------------------

# --------- Local settings: -----------------
# redis_connection = redis.Redis('localhost')
# qdrant_connection = QdrantClient("localhost", port=6333)
# rabbitmq_url = "amqp://guest:guest@localhost/"
# recommendation_service_url = "http://127.0.0.1:5001"
# logger.add('./logs/regular_pipeline.log',enqueue=True, backtrace=False, colorize=True) # Local version
# -------------------------------------------

# movie_mapping = redis_connection.json().get('movie_mapping') 
# movie_inv_mapping = redis_connection.json().get('movie_inv_mapping')
# movie_inv_mapping = {v: k for k, v in movie_mapping.items()}
# if not movie_mapping or len(movie_mapping) == 0: 
#     raise TypeError("Error durring movie ids extraction: movie ids mapping is absent in redis database!")


sys.tracebacklimit = 4

movie_mapping = None
movie_inv_mapping = None
local_bandit = None 

with open('data/run_params.json', 'r') as f: 
    params = json.load(f)

TOP_K = params['TOP_K']
N_recs = params['N_recs']
calc_diversity_flag = params['calc_diversity_flag']
divers_coeff = params['divers_coeff']
bandit_params = params['bandit_params']


@logger.catch
def _update_data() -> bool:
    global local_bandit
    global movie_mapping
    global movie_inv_mapping

    # Updating thompson sampling bandit if clean process is triggered
    logger.info(f'Updating movie items and bandit instance')
    movie_mapping = redis_connection.json().get('movie_mapping')
    if movie_mapping is None or len(movie_mapping) == 0: 
        logger.warning('Fail to get movie mapping data from redis!')
        return False
    movie_inv_mapping = {v: k for k, v in movie_mapping.items()}
    try:
        local_bandit = create_bandit_instance(
            n_arms=len(movie_mapping), 
            alpha_weight=bandit_params['alpha_weight'], 
            beta_weight=bandit_params['beta_weight']
            )
    except Exception as e:
        logger.exception(f'Exception while local_bandit updating in_update_data: {e}')
        return False
    return True


@logger.catch
def _update_bandit(data: pl.DataFrame) -> None: 
    global local_bandit
    global movie_mapping
    global movie_inv_mapping

    data_likes = (
        data
        .with_columns(pl.col('item_id').cast(pl.Utf8))
        .group_by(['item_id','action'])
        .len()
    )

    clear = redis_connection.get('clear')
    clear = int(clear) if clear is not None else 1
    if clear or local_bandit is None:
        logger.warning(f'Data updating!!! Clear flag is {bool(clear)}')
        if _update_data():
            redis_connection.set('clear', 0)
        else:
            logger.warning(f'!!! >>> Data updating is failed <<< !!!')
            return

    for item_id, action, num in data_likes.rows():
        local_bandit.retrieve_reward(arm_ind=movie_mapping[item_id], action=action, n=num)

    logger.info('calculating top recommendations')
    # top_inds = local_bandit.get_top_indices(k=TOP_K + 20)
    # top_item_ids = [movie_inv_mapping[i] for i in top_inds]
    top_items = _random_top_bunch(n_bunch=100)
    redis_connection.json().set('thompson_top', '.', top_items)
    redis_connection.set('top_updated', 1)
    logger.info('--->> Bandit updated. Thompson items updated! <<---')
    # redis_connection.json().set('bandit_state', '.', asdict(local_bandit))
    return 


@logger.catch
def _random_top_bunch(n_bunch: int):
    tops = []
    for _ in range(n_bunch):
        top_inds = local_bandit.get_top_indices(k=TOP_K + 20)
        top_item_ids = [movie_inv_mapping[i] for i in top_inds]
        tops.append(top_item_ids)
    return tops


async def update_top_recomendations():
    while True:
        if local_bandit is not None:
            top_items = _random_top_bunch(n_bunch=100)
            redis_connection.json().set('thompson_top', '.', top_items)
            redis_connection.set('top_updated', 1)
            logger.info('---> Top items updated <---')
        await asyncio.sleep(5)



async def collect_messages():

    connection = await aio_pika.connect_robust(
        rabbitmq_url,
        loop=asyncio.get_event_loop()
    )

    queue_name = "user_interactions"
    routing_key = "user.interact.message"

    async with connection:
        # Creating channel
        channel = await connection.channel()

        # Will take no more than 10 messages in advance
        await channel.set_qos(prefetch_count=50)

        # Declaring queue
        queue = await channel.declare_queue(queue_name)

        # Declaring exchange
        exchange = await channel.declare_exchange("user.interact", type='direct')
        await queue.bind(exchange, routing_key)
        # await exchange.publish(Message(bytes(queue.name, "utf-8")), routing_key)

        t_start = time.time()
        data = []
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    message = message.body.decode()
                    message = json.loads(message)
                    data.append(message)

                    if time.time() - t_start > 1:
                        logger.info('saving events from rabbitmq')
                        # update data if 10s passed
                        new_data = pl.DataFrame(data).explode(['item_ids', 'actions']).rename({
                            'item_ids': 'item_id',
                            'actions': 'action'
                        }).with_columns(pl.col('item_id').cast(pl.Utf8))

                        if len(new_data) > 0:
                            # --- Training thompson sampling algorithm of manyhands bandits --- 
                            _update_bandit(new_data)
                            
                            # ----- Saving interaction data ---------
                            if os.path.exists('./data/interactions.csv'):
                                df = pl.read_csv('./data/interactions.csv').with_columns(pl.col('item_id').cast(pl.Utf8))
                                data = pl.concat([df, new_data])
                            else:
                                data = new_data
                            data.write_csv('./data/interactions.csv')

                        data = []
                        t_start = time.time()


async def train_matrix_factorization(algo: str = 'ALS'):
    while True:
        if os.path.exists("./data/interactions.csv"):
            logger.info('Run matrix factorization')
            interact_data = pl.read_csv('./data/interactions.csv')
            user_mapping = {user: i for i, user in enumerate(interact_data['user_id'].unique())}
            # redis_connection.json().set('user_mapping', '.', user_mapping)
            
            clear = redis_connection.get('clear')
            clear = int(clear) if clear is not None else 1
            if clear or movie_mapping is None:
                logger.warning(f'Data updating in matrix factorization function!!! Flag is {redis_connection.get("clear")}')
                if _update_data():
                    redis_connection.set('clear', 0)
                else:
                    sleep_dt = 5
                    logger.warning(f'!!! >>> Data updating is failed <<< !!! Sleeping for {sleep_dt} dt')
                    await asyncio.sleep(sleep_dt)
                    continue
            
            df = (
                interact_data
                .sort(by='timestamp', descending=True)
                # !!!!!!!! Remove in prod!!!!!!
                # .unique(subset = ['user_id','item_id'], keep='first') 
                .with_columns([
                    pl.col('action').replace({'like': 1, 'dislike': -1}).cast(int).alias('value'),
                    pl.col('user_id').replace(user_mapping).cast(int).alias('user_index'),
                    pl.col('item_id').replace(movie_mapping).cast(int).alias('movie_index')
                ])
                .group_by('user_index')
                .agg([
                    pl.col('movie_index').alias('movies'),
                    pl.col('value').alias('values')
                ])
            )

            rows, cols, values = [], [], []

            for user_index, movie_indexes, likes in df.rows():
                rows.extend([user_index] * len(movie_indexes))
                cols.extend(movie_indexes)
                values.extend(likes)
            users_movies_data =  ss.csr_matrix((values, (rows, cols)), dtype = np.float32)
            
            logger.info(f'... Train {algo}')

            if algo == 'BPR':
                model = implicit.bpr.BayesianPersonalizedRanking(
                    random_state=None,
                )
            elif algo == 'ALS':
                model = implicit.als.AlternatingLeastSquares(
                    random_state=None
                )
            else:
                logger.exception(f'Algorith of matrix factorization is not defined! Function exit')
                return

            model.fit(users_movies_data)

            movie_embs = model.item_factors
            user_embs  = model.user_factors
            
            rec_start = time.time()
            rec_indeces, rec_scores = model.recommend(
                userid = np.arange(users_movies_data.shape[0]), 
                user_items = users_movies_data,
                N = N_recs)

            rec_time = round((time.time() - rec_start) * 1e3, 3)
            logger.info(f'Recommendation time = {rec_time} ms')
            
            if calc_diversity_flag:
                div_start = time.time()
                emb_size = movie_embs.shape[1]
                norm_movie_embs = normalize(movie_embs[rec_indeces].reshape(-1, emb_size), axis=1).reshape(-1,N_recs,emb_size)
                diversity = 1 - np.einsum('bij,bkj->bi', norm_movie_embs, norm_movie_embs)/N_recs
                rec_scores += divers_coeff * diversity
                score_args_sorted = rec_scores.argsort(axis=-1)[:,::-1]
                rec_indeces = np.take_along_axis(rec_indeces, score_args_sorted, axis=1)
                rec_scores = np.take_along_axis(rec_scores, score_args_sorted, axis=1)
                
                div_time = round((time.time() - div_start) * 1e3, 3)
                logger.info(f'Diversity calc time = {div_time} ms')

            cc_start  = time.time()
            for user_id in user_mapping:
                user_num = user_mapping[user_id]
                positive_recs = rec_indeces[user_num][rec_scores[user_num] > 0]
                recs = [movie_inv_mapping[i] for i in positive_recs]
                redis_connection.json().set(f'recs_user_{user_id}','.',recs)

            cc_time = round((time.time() - cc_start) * 1e3, 3)
            logger.info(f'Recommendation collection creation time = {cc_time} ms')

            # for user_id in user_mapping: 
            #     redis_connection.json().set(f'user_emb_{user_id}', '.', user_embs[user_mapping[user_id]].tolist())

        await asyncio.sleep(10)


async def calculate_top_recommendations():
    global local_bandit
    global movie_mapping
    global movie_inv_mapping

    while True:
        clear = redis_connection.get('clear')
        clear = int(clear) if clear is not None else 1
        if clear:
            # Updating thompson sampling bandit if clean process is triggered
            if _update_data(): 
                redis_connection.set('clear',0)
            else:
                await asyncio.sleep(5)
                continue
            
            if os.path.exists('./data/interactions.csv'):
                interactions = pl.read_csv('./data/interactions.csv')
                _update_bandit(local_bandit, interactions)

        # bandit = ThompsonSamplingBandit(len(movie_ids_list), alpha_weight=1, beta_weight=5)
#             top_items = (
#                 interactions
#                 .sort('timestamp')
#                 .unique(['user_id', 'item_id', 'action'], keep='last')
#                 .filter(pl.col('action') == 'like')
#                 .group_by('item_id')
#                 .len()
#                 .sort('len', descending=True)
#                 .head(100)
#             )['item_id'].to_list()
# 
#             top_items = [str(item_id) for item_id in top_items]
# 
#             redis_connection.json().set('top_items', '.', top_items)
        logger.info('calculating top recommendations')
        top_inds = local_bandit.get_top_indices(k=10)
        top_item_ids = [movie_inv_mapping[i] for i in top_inds]
        redis_connection.json().set('thompson_top', '.', top_item_ids)

        # Save bandit state to use it in recommendation service: 
        redis_connection.json().set('bandit_state', '.', asdict(local_bandit))

        await asyncio.sleep(5)


async def dump_metrics():
    while True: 
        logger.info('dump metrics')
        try:
            requests.get(f'{recommendation_service_url}/calc_metrics')
        except Exception as e:
            logger.warning(f"Cannot calculate metrics: {e}")
        await asyncio.sleep(10)



async def main():
    await asyncio.gather(
        collect_messages(),
        update_top_recomendations(),
        # train_matrix_factorization(),
        # calculate_top_recommendations(),
        # dump_metrics()
    )


if __name__ == '__main__':
    asyncio.run(main())
