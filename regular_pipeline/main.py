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
from aio_pika import Message
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, SearchRequest

from bandit import ThompsonSamplingBandit, create_bandit_instance

# --------- Docker settings: ----------------
redis_connection = redis.Redis('redis')
qdrant_connection = QdrantClient("qdrant", port=6333)
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

movie_mapping = redis_connection.json().get('movie_mapping') 
movie_inv_mapping = redis_connection.json().get('movie_inv_mapping')
# movie_inv_mapping = {v: k for k, v in movie_mapping.items()}
if not movie_mapping or len(movie_mapping) == 0: 
    raise TypeError("Error durring movie ids extraction: movie ids mapping is absent in redis database!")

sys.tracebacklimit = 4

bandit_params  = {
    'alpha_weight': 1,
    'beta_weight': 4 
}

# local_bandit = create_bandit_instance(
#     n_arms = len(movie_mapping), 
#     alpha_weight = bandit_params['alpha_weight'],
#     beta_weight = bandit_params['beta_weight']
#     )


def _update_data() -> bool:
    global local_bandit
    global movie_mapping
    global movie_inv_mapping

    # Updating thompson sampling bandit if clean process is triggered
    logger.info(f'Updating movie items and bandit instance')
    movie_mapping = redis_connection.json().get('movie_mapping')
    if not movie_mapping: 
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


def _update_bandit(data: pl.DataFrame) -> None: 
    data_likes = (
        data
        .with_columns(pl.col('item_id').cast(pl.Utf8))
        .group_by(['item_id','action'])
        .len()
    )
    bandit_state = redis_connection.json().get('bandit_state')
    if bandit_state is None: 
        logger.warning(f'Cannot update bandit state since bandit_state in redis database is empty!!!')
        return

    movie_mapping = redis_connection.json().get('movie_mapping')
    if movie_mapping is None:
        logger.warning(f'Cannot update bandit state since movie_mapping in redis database is empty!!!')
        return

    bandit_instance = ThompsonSamplingBandit(**bandit_state)

    for item_id, action, num in data_likes.rows():
        bandit_instance.retrieve_reward(arm_ind=movie_mapping[item_id], action=action, n=num)
    redis_connection.json().set('bandit_state', '.', asdict(bandit_instance))
    logger.info('Bandit state is updated successfully!')
    return 



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

                    if time.time() - t_start > 5:
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


async def train_matrix_factorization():
    while True:
        if os.path.exists("./data/interactions.csv"):
            logger.info('Run matrix factorization')
            interact_data = pl.read_csv('./data/interactions.csv')
            user_mapping = {user: i for i, user in enumerate(interact_data['user_id'].unique())}
            redis_connection.json().set('user_mapping', '.', user_mapping)
            
            # if redis_connection.json().get('clear'):
            #     _update_data(sleep_time = 20) # Update data of movies ids and Thompson bandit state 
            #     redis_connection.json().set('clear','.',False)
            movie_mapping = redis_connection.json().get('movie_mapping')
            if movie_mapping is None:
                asyncio.sleep(5)
                continue
            
            df = (
                interact_data
                .sort(by='timestamp', descending=True)
                # !!!!!!!! Remove in prod!!!!!!
                .unique(subset = ['user_id','item_id'], keep='first') 
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
            
            logger.info('... Train ALS')

            # model = implicit.bpr.BayesianPersonalizedRanking(
            #     random_state=42,
            #     )
            model = implicit.als.AlternatingLeastSquares(
                random_state=42
            )
            model.fit(users_movies_data)

            movie_embs = model.item_factors
            user_embs  = model.user_factors
            
            cct_start = time.time()
            qdrant_connection.recreate_collection(
                collection_name="movie_embs",
                # задаем размерность векторов и метрику дистанции
                vectors_config=VectorParams(size=movie_embs.shape[1], distance=Distance.COSINE),
            )

            # To add all item embeddings to qdrant database, but there are many all zero vectors. 
            # Hence, this operation is very time consumming.
            # qdrant_connection.upsert(
            #     collection_name="movie_embs",
            #     points=[
            #         PointStruct(id = idx, vector = emb.tolist())
            #         for idx, emb in enumerate(movie_embs)
            #     ]
            # )

            # To add only nonzero vectors to the qdrant db: 
            mask = ~np.all(movie_embs == 0, axis =1)
            pos_idxs = np.where(mask)[0].tolist()

            qdrant_connection.upsert(
                collection_name="movie_embs",
                points=[
                    PointStruct(id = idx, vector = movie_embs[idx].tolist())
                    for idx in pos_idxs
                ]
            )
            create_coll_time = round(time.time() - cct_start, 5)
            logger.info(f'qdrant collection creation time = {create_coll_time} s')

            for user_id in user_mapping: 
                redis_connection.json().set(f'user_emb_{user_id}', '.', user_embs[user_mapping[user_id]].tolist())

        await asyncio.sleep(15)


async def calculate_top_recommendations():
    global local_bandit
    global movie_mapping
    global movie_inv_mapping

    while True:
        if redis_connection.json().get('clear'):
            # Updating thompson sampling bandit if clean process is triggered
            if not _update_data(): 
                asyncio.sleep(5)
                continue
            redis_connection.json().set('clear','.',False)
            
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
        # train_matrix_factorization(),
        # calculate_top_recommendations(),
        # dump_metrics()
    )


if __name__ == '__main__':
    asyncio.run(main())
