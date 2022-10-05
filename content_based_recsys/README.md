# Content Based Filtering

With the rapid growth of data readily available online, recommendation systems have become popular as a type of information filtering system. They improve the relevancy of results shown to users. 

One popular type of recommendation system is content-based filtering. In this method, the system recommends items that are similar to ones that a user has already liked or interacted with. It could use item metadata such as description, title, director for a movie to make these recommendations. The main idea motivating these sytems is that a user is likely to interact with an item that is similar to one with which they have already interacted.

![](https://www.naukri.com/learning/articles/wp-content/uploads/sites/11/2022/01/Content-Based-Filtering.png)

**Advantages** 
- Content based filtering can capture the specific interests of a user, and can recommend niche items that very few other users are interested in. 
- The model doesn't need any data about other users, since the recommendations are specific to this user. This makes it easier to scale to a large number of users. 

**Disadvantages**
- Since the feature representation of the items are hand-engineered to some extent, this technique requires a lot of domain knowledge. Therefore, the model can only be as good as the hand-engineered features. 
- The model can only make recommendations based on existing interests of the user. In other words, the model has limited ability to expand on the users' existing interests.

To surmount the aformentioned disadvantages, content-enriched models have been proposed that leverage both content features and user-item interactions to generate recommendations. These approaches are also known as hybrid recommender systems since they are a hybrid of collaboarive and content-based filtering. 

## Notebooks 
There are two notebooks related to content-based filtering. These demos go from data preproccessing to model evaluation. In particular, the demos include the following:

1. **content_based_recsys**: Content Based Filtering on the TMDB movie dataset. 
2. **two_two_recsys**: Content Enriched Filtering on MovieLens Dataset. 
