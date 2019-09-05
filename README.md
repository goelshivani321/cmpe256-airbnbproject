# AirBnb Recommender System
“Travel is the only thing you buy that makes you richer” - Author Unknown
					
A view of the Statue of Liberty or closeness to Times Square cannot be the only two factors to decide where a traveller would want to stay in New York. With the increase in importance of customer satisfaction, we believe it is more than that. Every city is unique and where we stay, plays a major role in how our vacation is going to pan out. Airbnb plays a major role in offering more of these unique stays and experiences. 

However, before booking any stay with AirBnb, a user just does not rely on the listing. She checks the security, neighbourhood and specially the reviews. This leads to an endless scrolling spree and there is  high probability that the user might go to a hotel instead, which has ratings on websites like Tripadvisor and Trivago. 

The goal of this project is to build a recommender system for AirBNB customers in order  to help them select  their perfect stay. 

## Objective
The objective of the project is to provide Airbnb listings recommendations to a user based on
i. Their profile and requirements 
ii Ratings derived based on user reviews

## Algorithms Considered
For the AirBnB data, user Id is the reviewer Id and the item Id is the listing Id. We are provided with metadata for various listings and also reviews by various customers. One of the major challenges faced was that no rating was provided by the user. The only input from the user was in the form of  review. Hence, we used Natural Language Processing(NLP) to generate user ratings for every listing.

### 1. Generating base ratings using NLP 
Our first challenge included not having ratings from customers. The only input from customers was in the form of reviews. Hence, we decided to use the reviews to generate base ratings. We generate ratings for all the listings using the reviews provided. The data was cleaned and polarity was generated. There were reviews in about 56 different languages. We analyzed the data and segregated different languages. We used NLP on 5 major languages. The ratings lie between -1 to 1, where 1 is a positive rating. Here, 0 is a neutral rating while -1 is a negative rating. 

### 2. Recommendation Engine: Content Based Rating Prediction
We used feature engineering and feature selection for the metadata provided for various listings. We tried to predict individual ratings for different combinations of user id and item id using XGBoost and Regression. Along with that GridSearchCV was used to train data for feature analysis.

### 3. Recommendation Engine: Surprise Library Recommendations
We used Surprise Library to train and test our data(UserId, ReviewerId, Ratings based on polarity of the reviews). We predicted ratings using SVD, SVDpp, NMF, KNNBaseline, KNNBasic, BaseLineOnly, NormalPredictor, KNNWithZScore, KNNWithMeans, Co-clustering. We used HPC to streamline our workflow.

## Technologies and Tools Used
Python Packages: scikit-surprise, Pyspark, pandas, numpy, matplotlib, seaborn, scikit-learn, Hunspell, Vader lexicon
IDE/Environment : Jupyter Notebook with Python 3.7, HPC

## Implementation Details
Below section details the project implementation.
### 1. Generating ratings using NLP
Firstly, we analyz the reviews to check which language the review is in. After that we cleaned the whole data which included:
* Removing Nan
* Spelling fixes
* Lowercasing the review
* Removing words with numbers
* Removing stop words
* Removing empty tokens
* Lemmatizing the text
* Removing words with a single letter

After that we performed Sentiment Analysis for 4 major languages, including English, Spanish, German and  French to generate ratings between -1 to 1. These ratings are the polarity of different reviews. We also identified bot generated reviews and removed them.

### 2. Content Based Rating Prediction using meta-data  
### Data Exploration
In order to understand the data better, we analyzed correlation between ratings and properties of various listings using visualizations.
We tried to find if certain boroughs achieve more reviews or higher reviews than the other.
We also found if correlation exists between the amenities provided and the ratings.
### Feature Engineering
The meta data was cleaned and analyzed based on various features to create item profile:
* Instant Bookable - removed Nan and converted to categorical feature
* Number of reviews per month - removed Nan and imputed missing values
* Room Type - removed Nan and converted to categorical feature
* Availability 365 - imputed missing values
* Minimum Nights - bucketed the feature
* Price - bucketed feature using quantile cut
* Is the host super host or not - Removed Nan and converted to categorical feature
* Is the host verified - Removed Nan and converted to categorical feature
* Cancellation Policy - used one-hot encoding
* Does the host respond on time - imputed missing values and converted to integer data as the feature had an order.
* Amenities Length - We found the length of the amenities column to find a rough estimate of how many amenities the host provides
* Description Length - We assumed that some hosts give a detailed description because of which the customer knows what to expect and gives a better rating. However, we were proved wrong as hosts might not give a fair description of their property.
* Neighborhood Group - One hot encoding used to create 5 columns, one for each borough

* Data Analysis  *
A new csv was created with updated features and correlation matrix visualization was used to determine which features are useful. 

* Data Modelling *
We split the total data into 2 halves, Training data which is 80% and Testing data which is 20%. We used Regression based models along with GridSearchCV on HPC to find the best model.  We finally predict top 10 recommendations for a user based on what listings he might highly rate.

### 3. Recommendation Engine: Surprise Library Recommendations
We created a user - item matrix with the ratings that were generated using polarity of the reviews. We did a 3:1 split on our data and predicted ratings using Matrix Factorization Algorithms as well as Neighbourhood Based Algorithm. We used gridSearchCV to tune our algorithm.

## Dataset used
* Name : * AirBnB  New York Dataset, New York crime data, New York subway access points
* Type of Files : *  CSV
* Details: *
Airbnb reviews (1.1 million user reviews) : http://data.insideairbnb.com/united-states/ny/new-york-city/2019-03-06/data/reviews.csv.gz

## Data preprocessing Decisions
* In absence of explicit ratings, decision was made to use sentiment analysis of user reviews to arrive at ratings.
* Decision to merge diverse yet related datasets to enrich features as part of feature engineering.

## Methodology followed 
### Training Data : 
We have used GridSearchCV to train our clean data. The default value for cv was chosen to be 3 . A 70:30 split was created for training and testing data. The root mean squared error was chosen as an evaluation metric.

### Modelling the Data
Content Based Recommendation using meta-data  
Scikit-learn’s Gradient Boosting Trees was used to model the data. ‘Gbtree’ booster was used with maximum tree depth set to 24. N_estimator was set to 58. gamma=0.927 and learning_rate=0.42288 were chosen.

### Surprise Library Recommendations
We used cross validation with cv = 3 for initial analysis on all the models. The best performing models were SVD and NMF. We fine tuned these models to get a better rating. We used 3 fold cross validation to compare models and 5 fold cross validation to tune hyper parameters.

## Analysis of Results
Based on my analysis I found that one of the  important features to successfully predict ratings include if the host is a super host or not. We also find that matrix factorization performs the best to predict if a customer will rate and airbnb listing higher or not. The following are the models we worked on :

| Algorithm                                                                                                                                                      |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RMSE                                                                                                                                                           |
| Hyperparameters                                                                                                                                                |
| BaselineOnly                                                                                                                                                   |
| 0.265576                                                                                                                                                       |
| reg_i=10, reg_u = 15, n_epochs = 10, method = ALS                                                                                                              |
| KNNBaseline                                                                                                                                                    |
| 0.266391                                                                                                                                                       |
| k=40, min_k=1, user based                                                                                                                                      |
| KNNWithMeans                                                                                                                                                   |
| 0.266391                                                                                                                                                       |
| k=40, min_k=1, user based                                                                                                                                      |
| KNNWithZScore                                                                                                                                                  |
| 0.272735                                                                                                                                                       |
| k=40, min_k=1, user based                                                                                                                                      |
| KNNBasic                                                                                                                                                       |
| 0.27275                                                                                                                                                        |
| k=40, min_k=1, user based                                                                                                                                      |
| NMF                                                                                                                                                            |
| 0.308257                                                                                                                                                       |
| n_factors = 15, n_epochs = 50                                                                                                                                  |
| NormalPredictor                                                                                                                                                |
| 0.350181                                                                                                                                                       |
| No Hyper Parameters as random rating predicted based on training set distribution                                                                              |
| CoClustering                                                                                                                                                   |
| 0.440552                                                                                                                                                       |
| n_cltr_u=3, n_cltr_i=3, n_epochs=20                                                                                                                            |
| Spark ALS                                                                                                                                                      |
| 0.773566                                                                                                                                                       |
| maxIter=20, regParam=0.3, rank=10                                                                                                                              |
| SVD                                                                                                                                                            |
| 0.2661                                                                                                                                                         |
| n_epochs': 50, 'n_factors': 100, 'random_state': 42, 'lr_all': 0.005, 'reg_all': 0.02                                                                          |
| SVDpp                                                                                                                                                          |
| 0.26688                                                                                                                                                        |
| n_factors = 20, n_epochs = 20, lr_all = 0.007, reg_all = 0.02                                                                                                  |
| XGB                                                                                                                                                            |
| 0.26743                                                                                                                                                        |
| base_score=0.5, booster='gbtree', colsample_bylevel=1,  colsample_bytree=0.9809552080191412, gamma=0.9277541727646688,                                         |
|  importance_type='gain', learning_rate=0.422880206247594,                                                                                                      |
| max_delta_step=0, max_depth=24, min_child_weight=20.316769021830574,                                                                                           |
| missing=None, n_estimators=58, n_jobs=1, nthread=None, objective='reg:linear', random_state=0, reg_alpha=19.294198125487966, reg_lambda=1, scale_pos_weight=1, |
|  seed=None, silent=True, subsample=0.9819023896694455                                                                                                          |

## Tools used (Detailed view)
| Tool                         | Version                  | Purpose                                                                 |
|------------------------------|--------------------------|-------------------------------------------------------------------------|
| Jupyter Notebook with Python | Conda 4.6.14, Python 3.7 | For preprocessing, model training and evaluation                        |
| scikit-surprise              | 1.0.6                    | To implement Matrix factorization algorithm (SVD)                       |
| Pyspark                      | 2.4.0                    | To implement Collaborative filtering using MLLib ALS algorithm          |
| Hunspell                     |                          | For sentiment analysis                                                  |
| Pandas                       | 0.23.4                   | To manipulate utility matrix                                            |
| Numpy                        | 1.15.4                   | To use nparray data structures for preprocessing and data manipulation. |
| Matplotlib                   | 3.0.2                    | For plotting graphs for visualization                                   |
| Seaborn                      | 0.9.0                    | For plotting graphs for visualization                                   |

## References

* Pazzani, M. J., & Billsus, D. (2007). Content-based recommendation systems. In The adaptive web (pp. 325-341). Springer, Berlin, Heidelberg.

* Zervas, G., Proserpio, D., & Byers, J. (2015). A first look at online reputation on Airbnb, where every stay is above average. Where Every Stay is Above Average (January 28, 2015).

* Wang, X., He, X., Feng, F., Nie, L., & Chua, T. S. (2018, April). Tem: Tree-enhanced embedding model for explainable recommendation. In Proceedings of the 2018 World Wide Web Conference on World Wide Web (pp. 1543-1552). International World Wide Web Conferences Steering Committee.








