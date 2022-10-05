# Introduction to Reinforcement Learning based Recommender Systems

Most practical recommender systems focus on estimating immediate user engagement without considering the long-term effects of recommendations on user behaviour. Reinforcement learning (RL) methods offer the potential to optimize recommendations for long-term user engagement.

This demo introduces a slate recommendation reinforcement learning algorithm called ```SlateQ```, as well as the simulation environment ```RecSim```:
* [```SlateQ```](https://arxiv.org/pdf/1905.12767.pdf) decomposes the long-term value (LTV) of a slate of items into a tractable function of its component item-wise LTVs to address the RL challenge of a large combinatorial action space. 
* [```RecSim```](https://arxiv.org/pdf/1909.04847.pdf) is a configurable platform for authoring simulation environments to allow both researchers and practitioners to challenge and extend existing reinforcement learning (RL) methods in synthetic recommender settings.

This demo is inspired by the tutorials provided in the [```RecSim``` repository](https://github.com/google-research/recsim/tree/master/recsim/colab)

## Requirements
* recsim
* pandas
* matplotlib