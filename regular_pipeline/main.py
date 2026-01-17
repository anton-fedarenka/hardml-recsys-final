import asyncio
import json
import os.path
import time
import numpy as np
import scipy.sparse as ss
import implicit

import aio_pika
import polars as pl
import redis
from aio_pika import Message
from bandit import ThompsonSamplingBandit

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, SearchRequest

# redis_connection = redis.Redis('redis')
redis_connection = redis.Redis('localhost')
qdrant_connection = QdrantClient("localhost", port=6333)

movie_mapping = redis_connection.json().get('movie_ids_mapping') 
# movie_inv_mapping = redis_connection.json().get('movie_inv_mapping') 
movie_inv_mapping = {v: k for k, v in movie_mapping.items()}
if not movie_mapping or len(movie_mapping) == 0: 
    raise TypeError("Error durring movie ids extraction: movie ids mapping is absent in redis database!")

bandit_inst = ThompsonSamplingBandit(n_arms=len(movie_mapping), alpha_weight=1, beta_weight=4)


async def collect_messages():
    connection = await aio_pika.connect_robust(
        # "amqp://guest:guest@rabbitmq/",
        "amqp://guest:guest@localhost/",
        loop=asyncio.get_event_loop()
    )

    queue_name = "user_interactions"
    routing_key = "user.interact.message"

    global bandit_inst

    async with connection:
        # Creating channel
        channel = await connection.channel()

        # Will take no more than 10 messages in advance
        await channel.set_qos(prefetch_count=10)

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

                    if time.time() - t_start > 10:
                        print('saving events from rabbitmq')
                        # update data if 10s passed
                        new_data = pl.DataFrame(data).explode(['item_ids', 'actions']).rename({
                            'item_ids': 'item_id',
                            'actions': 'action'
                        }).with_columns(pl.col('item_id').cast(pl.Int64))

                        if len(new_data) > 0:
                            # --- Training thompson sampling algorithm of manyhands bandits --- 
                            update_bandit(bandit_inst, new_data)
                            
                            # ----- Saving interaction data ---------
                            if os.path.exists('../data/interactions.csv'):
                                data = pl.concat([pl.read_csv('../data/interactions.csv'), new_data])
                            else:
                                data = new_data
                            data.write_csv('../data/interactions.csv')

                        data = []
                        t_start = time.time()


def update_bandit(bandit_instance: ThompsonSamplingBandit, data: pl.DataFrame) -> None: 
    data_likes = (
        data
        .with_columns(pl.col('item_id').cast(pl.Utf8))
        .group_by(['item_id','action'])
        .len()
    )
    for item_id, action, num in data_likes.rows():
        bandit_instance.retrieve_reward(arm_ind=movie_mapping[item_id], action=action, n=num)
    return 
 

async def train_matrix_factorization():
    while True:
        if os.path.exists("../data/interactions.csv"):
            print('----- !!!! Run matrix factorization !!!!! ----- ')
            interact_data = pl.read_csv('../data/interactions.csv')
            user_mapping = {user: i for i, user in enumerate(interact_data['user_id'].unique())}
            print(f'user_mapping = {user_mapping}')
            redis_connection.json().set('user_mapping', '.', user_mapping)
            
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
            
            print('===>> Train BPR <<===')

            bpr_model = implicit.bpr.BayesianPersonalizedRanking(
                random_state=42,
                )
            bpr_model.fit(users_movies_data)

            movie_embs = bpr_model.item_factors
            user_embs  = bpr_model.user_factors
            
            np.save('../data/movie_embs.npy', movie_embs)

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

            for user_id in user_mapping: 
                redis_connection.json().set(f'user_emb_{user_id}', '.', user_embs[user_mapping[user_id]].tolist())

        await asyncio.sleep(30)


async def calculate_top_recommendations():
    global bandit_inst
    global movie_mapping
    global movie_inv_mapping

    while True:
        if redis_connection.json().get('clear_bandit'):
            # Updating thompson sampling bandit if clean process is triggered
            print(f'-------- >>>>>>> CLEANING BANDIT!!!! <<<<<<<<< ----------------')
            movie_mapping = redis_connection.json().get('movie_ids_mapping')
            if not movie_mapping: 
                continue
            movie_inv_mapping = {v: k for k, v in movie_mapping.items()} 
            bandit_inst = ThompsonSamplingBandit(n_arms=len(movie_mapping), alpha_weight=1, beta_weight=4)
            redis_connection.json().set('clear_bandit','.',False)
            
            if os.path.exists('../data/interactions.csv'):
                print(' >>> updating bandit by interaction file')
                interactions = pl.read_csv('../data/interactions.csv')
                update_bandit(bandit_inst, interactions)

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
        print('calculating top recommendations')
        top_inds = bandit_inst.get_top_indices(k=10)
        top_item_ids = [movie_inv_mapping[i] for i in top_inds]
        redis_connection.json().set('thompson_top', '.', top_item_ids)

        await asyncio.sleep(10)


async def main():
    await asyncio.gather(
        collect_messages(),
        calculate_top_recommendations(),
        train_matrix_factorization(),
    )


if __name__ == '__main__':
    asyncio.run(main())
