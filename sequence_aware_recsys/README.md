## Introduction to Sequence Aware Recommender Systems
Traditional recommender system methods such as matrix factorization assume static preference of users. However, users preferences often evolve over time based on their past interactions. For example, it is reasonable to assume that users who purchase an iPhone on Amazon are more likely to purchase products such as phone cases and chargers in the future. This implies there is inherent value in modelling the temporal dynamics of user interactions. In general, sequential interaction data of a user takes the following form: 

<p align="center">
<img width="818" alt="image" src="https://user-images.githubusercontent.com/34798787/182863761-6f195aa8-7a73-4abb-a440-5debf1cb44fb.png">
</p>

In the recommender system literature, there are two overlapping classes of approaches that model sequence of user interactions: sequence aware recommender systems and session based recommender systems. The distinction between the approaches lies in the data. Sequence Aware Recommender systems have sequences of interaction for a set of known users. This sequence of interactions exists across one or more sessions. Alternatively, Session-Based recommender systems have sequences of interactions for an uknown set of users. Thus, there is no concept sequences existing across sessions. 

<p align="center">
<img width="956" alt="image" src="https://user-images.githubusercontent.com/34798787/182865897-b2ff207f-111d-4716-a809-3682009abf2c.png">
</p>

The topic of this demo is sequence aware recommender systems. In general, sequence aware recommender systems start by generating embeddings for each interaction in a given sequence. This involves mapping the continuos and categorical variables of each interaction to an embedding using embedding tables or Multi Layer Perceptrons (MLP). This results in a sequence of embeddings that is then fed into a Neural Network model that predicts the probability distribution over items given their previous interactions. The neural network can be chosen to have different architectures. Typically, Recurrent Neural Network (RNN) or Transformers are used because their success with sequential data. 

<p align="center">
<img width="977" alt="image" src="https://user-images.githubusercontent.com/34798787/182866390-4e1322b5-2d68-4dd2-ad5f-f1814a965b95.png">
</p>



