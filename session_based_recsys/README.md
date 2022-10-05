# Session-Based Recommender Systems

It is almost always the case that choices have a time-sensitive context; for example, some items may be more relevant than others based on **when they were last viewed** or **purchased**.

Despite being embedded in the user's most recent interactions, short-term preferences may only represent a small portion of history. Additionally, a user's preference for certain items can be **dynamic rather than static**; it can evolve over time. 

As a result, **session-based recommendation** algorithms have been developed, which **rely heavily on the user's recent interactions** instead of their historical preferences. Moreover, this approach is especially advantageous since **a user may appear anonymously** if they are not logged in or browsing incognito.  

## Benefits of Session-Based Recommender Systems

*   These methods can be implemented even in the **absence of historical user data**, and doesn’t explicitly rely on user population statistics. This is helpful because, as just noted above, users aren’t always logged in when they browse a website.

*   A wealth of new, **publicly available, session-centric datasets** have been released, especially in the e-commerce domain, allowing for model development and research in this area.

*   Session-based recommenders can benefit from the rise of **deep learning approaches** expressly suited for sequences.

## Notebooks
There are two notebooks in this series of demos that outline the application of session-based recommender systems to the YOOCHOOSE dataset. These demos go from data preproccessing to hyperparameter tuning. In particular, the demos include the following: 

1. **GRU4Rec**: Preprocess YOOCHOOSE dataset, split into train, validation and test sets, and train and evaluate GRU4Rec model on the dataset
2. **NARM**: Preprocess YOOCHOOSE dataset, split into train, validation and test sets, and train and evaluate NARM model on the dataset
