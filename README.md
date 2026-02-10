*The project was completed as part of the [HardML](https://karpov.courses/ml-hard) course by
[karpov.courses](https://karpov.courses/). The project, as well as the entire course, was
successfully completed with the highest possible grade
([certificate](https://lab.karpov.courses/certificate/d22be6be-aa1f-4f8a-a6f0-8dc4546b11de/en/)).*


# Recommendation system
## Project description

The project is a service of movie and TV series recommendations based on user reactions 
(likes/dislikes). To provide relevant recommendations to so called "cold" users, i.e. users 
without available interaction history, the reinforcement learning algorithm was implemented —
a multi-armed bandit algorithm implementing Thompson sampling scenario. 
It turned out that even this approach alone is capable of overcoming the upper limits 
of targer metrics and obtaining the maximum score.
However, to further improve the quality of recommendations for users with history, 
matrix factorization algorithms were also applied. The ALS and BPR algorithms were tested, but only BPR actually improved the final results.

## Project Structure
The recommender system consists of three main components:

* **Recommendations Service** — a backend service responsible for recommendations to end users, with the following endpoints:

    * /healthcheck returns the service status. The grader requires a 200 (OK) response code.

    * /cleanup resets the environment before running the grader again (each run has new identifiers, and all possible caches must be cleared for proper validation).

    * /add_items adds new recommendation objects to the system (the item_ids field contains a list of object identifiers, and genres contains a list of genre lists for the corresponding objects in item_ids).

    * /recs/{user_id} returns a list of recommendations as a list of objects for the user specified by user_id.

* **Event Collector** — a service for collecting and processing user reactions (likes/dislikes),
 which processes the /interact endpoint. It stores received events in RabbitMQ message broker for
 further processing in the Regular pipeline service.

* **Regular pipeline** — a task scheduler. Asynchronously processes data with interaction events
and trains recommendation algorithms based on it. The final version uses two main recommendation
algorithms: a multi-armed bandit implementing the Thompson sampling scenario and a martix
factorization algorithm. 

* **Webapp** — a frontend service used for debugging and testing the recommendation system. 

## Project launch

To implement the project, it was required to deploy it on a virtual machine with a static IP
address accessible on the network. This requirement determined the scheme of deployment and launch 
of the project. To run the project:

1. Install Docker and Docker Compose; 
2. Execute ``docker compose up --build``

## Description of versions

1. v0.0 — The first working version with an additional endpoint in the
recommendation service for regular logging of performace metrics in MLflow. 
2. v1.0 — The latest version uses qdrant and calculations within the
recommendation service: working, but very slow.
3. v2.0 — Working and very well performed version based on multi-armed bandit
algorithm only.
4. v3.0 — The best and final version using both the multi-armed bandit 
along with matrix factorization algorithm.




