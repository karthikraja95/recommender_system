## Dataset
In this series of demos we will explore the [Amazon Review Dataset](https://nijianmo.github.io/amazon/index.html). This dataset includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs). This version provides 233.1 million reviews in total and additonal metadata. The reviews are distributed across 26 high level groups, as illustrated below: 

<p align="center">
<img width="600" alt="Amazon categories" src="https://user-images.githubusercontent.com/34798787/177803504-89909b59-a2cd-497b-b892-64f40a9a9e29.png">
</p>

We will specifically be looking at the Movies and TV category - which is commonly used as a benchmark in the recommender system literature. The data is processed and stored in a table where each row contains the features and label of a single interaction. The features include user id, item id, category id and the timestamp of the interaction. The label is binary indicating whether a review exists between that specific user-item pair. 

## Notebooks
There are several notebooks in this series of demos that outline the application of sequence aware recommender systems to the Amazon review dataset. These demos go from data preproccessing to hyperparameter tuning. In particular, the demos include the following: 

1. **amazon_preprocessing**: Download amazon data, preprocess and split into train, validation and test sets.
2. **a2svd**: Train and evaluate baseline non-sequential recommender system method Asymmetric SVD (a2SVD)
3. **slirec**: Train and evaluate sequential recommender system method Short-term and Long-term preference Integrated Recommender system (SLi-Rec)

