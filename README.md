# Project 3 - Machine Learning with Reddit Posts
---

## Problem Statement

Over the years, the Android platform has grew with a large user base. Many users would browse forums such as Reddit to retrieve information regarding the latest updates and features released with its firmware. My instructor, being a huge Android fan, struggles to differentiate the posts from the Android Subreddit and Android Apps Subreddit. In hopes of aiding our instructor, we want to construct a model that properly distinguishes each forum post to their appropriate Subreddits.


## Relevant Files
* [Notebook](https://git.generalassemb.ly/tempyst/project_3/blob/master/code/Subreddit%20Classification.ipynb)
* [Dataset](https://git.generalassemb.ly/tempyst/project_3/tree/master/data)
* [Graphs and Charts](https://git.generalassemb.ly/tempyst/project_3/tree/master/graphs)

---

## Executive Summary

Reddit has been part of the online community for many years. The amount of users that visit the forum and posts opinions are unimaginable. For this project we examine two subreddit(forums), Android and Android Apps. We want to build a model that can predict which post belongs to their appropriate forum with a decent amount of accuracy. Since accuracy along is not enough to determine a model's effectiveness, we will also utilize other classfication metrics such as sensitivity and specificity to properly evaluate that particular model.

Because we are dealing with texts, we cannot examine the data in a quanitifiable manner without performing some pre-processing work. We must first retrieve the forum posts from Reddit's json version of their Subreddit and extract the relevant information from the json data. Since Reddit caps our data pull at 1000 posts, we created a loop to ensure we stop the retrieval at the target value otherwise we would receive duplicate posts. Even after we obtain the posts, we still had to ensure none of the posts were indeed duplicates. As a precaution, we quickly converted the data into a dataframe of posts for their respective forums and saved them as a CSV file. This is crucial because we cannot always pull Reddit everytime we want to examine our data and since the forum is a live discussion, the forum posts will be different everytime we performed a data pull. We then proceed to our cleaning and EDA portion of the project.

We want to examine how the texts were compiled and if there were any formatting of text that were corrupted when we pulled the data. We also have to consider removing the punctuations within the data as well as setting everything word to lower case. In addition to all of the above cleaning, we also want to stem the words. Being a forum posts, there could be a lot of similar ways to refer to the same term and could inflate the features to have multiple words having the same root word. 

For our EDA, we want to examine the before and after cleaning of dataset and see if that had a substantial affect to the features and the frequency of words. We will have to use a CountVectorizer to transform the each word in the corpus into a feature and tally their frequency. This could tell which word or words would have a huge influence on the model in predicting the association of subreddits. We also looked at the bigrams (2 word terms) and see which combination were most frequently used. 

We finally proceed to our modeling process utilizing a variety of models and try to pick the best one out of the models chosen. For each model, we applied both the CountVectorizer variant and TFDIFVectorizer variant. Some models might perform better defpending on the transformer we apply but we will not know which model favors which transformers until the model is fitted. We wil examine the metrics as well as the ROC curve and perform an aggregate judgement on the model's performance and hopefully find a great predictor.


## Conclusion and Recommendations

We choose the Support Vector Machine model using CountVectorizer to be the model of choice. With an accuracy score of 84.9% and a specificity of .867. We wanted to minimize false positives and believed to have achieved our goal. The optimal parameters are: 
    - having a maximum of 2000 features
    - using single words and bigrams together, 
    - utilizing 80% of the features,
    - minimum number of data is 2
    - the degree of polynomial for the SVC model is 1
    - the penalty parameter for the SVC model is 1
    
These combination of parameters will yeild us the optimal result for accuracy and provide us with a very workable model to predict which posts belong to which forum. Another trend we seem to notice is that the CountVectorizer transformer appears to optimize for the false positives while the TFIDFVectorizer optmizes for false negatives. This could be help in future model if we are trying to optimize or minimize false positives/negatives, we can focus specifically on one of the transformers. This model however, is not perfect. This model would fall short on explain which feature would contribute the most to being a predictor to the forums. Because the model is a black box setting, we can only say it was a good model but cannot conclusive say why. This is similar to the Lasso and Ridge models for continuous value targets where we have to apply a standardizing transformation which will cause us to lose interpretation of the features as a trade-off to get a better model. 

We believe that our model could have improve if users wrote every post without slangs. It is highly unlikely NLP tools such as lemmatizers and stemming will understand the presense of slangs. If every post were written in grammatically correct English, it might improve our model. In addition, some posts were too short and cost us a reddit pull. Shorter posts means less data for the model to correlate why that particular post belongs in their assigned forum. If we would filter the size of posts pulled and only keep the long post from Reddit, the model could improve. We understand that we could truncate the data after we create the dataframe of all shorter posts. But that would decrease the size of our dataset and increase bias. 

We hope our instructor would be satisfy with the outcome of the model and hope he is able to better differentiate the difference in posts from their respective forums as well as our model.
