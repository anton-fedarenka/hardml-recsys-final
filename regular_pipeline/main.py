import asyncio
import json
import os.path
import time

import aio_pika
import polars as pl
import redis
from aio_pika import Message
from bandit import ThompsonSamplingBandit

# redis_connection = redis.Redis('redis')
redis_connection = redis.Redis('localhost')

movie_ids_list = redis_connection.json().get('movie_ids')
# print(f'movie_ids_list from redis: {movie_ids_list[-10:]}')
bandit_inst = ThompsonSamplingBandit(n_arms=len(movie_ids_list), alpha_weight=1, beta_weight=4)


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
        bandit_instance.retrieve_reward(arm_ind=movie_ids_list.index(item_id), action=action, n=num)
    return 


async def calculate_top_recommendations():
    global bandit_inst
    while True:
        if redis_connection.json().get('clear_bandit'):
            # Updating thompson sampling bandit if clean process is triggered
            print(f'-------- >>>>>>> CLEANING BANDIT!!!! <<<<<<<<< ----------------')
            bandit_inst = ThompsonSamplingBandit(n_arms=len(movie_ids_list), alpha_weight=1, beta_weight=4)
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
        top_item_ids = [movie_ids_list[i] for i in top_inds]
        redis_connection.json().set('thompson_top', '.', top_item_ids)

        await asyncio.sleep(10)


async def main():
    await asyncio.gather(
        collect_messages(),
        calculate_top_recommendations(),
    )


if __name__ == '__main__':
    asyncio.run(main())
