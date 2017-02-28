# twitter-sentimental-analysis
This is a supervised learning of twitter data, and classifying tweets as positive or negative according to their sentiment.

This project is aimed at building a model to analyze the public sentiment toward presidential candidates in the 2012 U.S. election as expressed on Twitter. Twitter is a social networking service where people post their views and opinions on various things including views on political parties and candidates. So, analyzing the tweets gave potential information on people’s views on Obama and Romney during the campaign time and deriving a predictive model based on the analysis even helped in estimating the election outcome. Currently, the project is to analyze the large number of tweets on Obama and Romney collected and to derive their sentiment (positive or negative). 

## Introduction:
The data consisted of tweets during the election campaign for the presidential candidates Obama and Romney were given specifically mentioning their sentiment whether it was a positive, negative or a neutral tweet. The Classes given were:
	1 = positive tweet
	0 = neutral tweet
 -1 = negative tweet
	2 = unknown.
This is a typical supervised text analysis problem where the sentiment of the tweets is specified in the training data. So, we had to train our classifiers which are different in their approach of models and independence assumptions for the features selected so that, we converge on their parameters and hyperparameters (alpha, ngrams, regularization etc.) to minimize the error and improve the accuracy using chosen metrics. And finally, after building the classifier we use it to test its performance on the testing data.


## Techniques:
### Data Analysis:
#### Correlation: 
The data had no meaning in terms of correlation as the words in the tweets were mostly not dependent on one another. Hence no measures have been taken to account for the correlation in data.
#### Skewed Data: 
The tweets in the romney training dataset was skewed in the context that many of them were neutral in their sentiment (class 0). When dealing with highly skewed data we observed that the typical feature selection methods like information gain etc., leads to mostly selecting features for the minor class, hence considerably affecting the classification performance. 
To deal with this, we employed two methods to improve the classifiers performance.
1. Over sampling the minor class: This considerably improved the accuracy of the prediction from about 49% to 64%. This was a major boost in improving performance of the classifier, especially for the Romney dataset.
2. Under sample the major class: This did not affect the classifier performance as much as the over-sampling did, as probably some of the important features were removed from being selected because of this.
Additional measures were taken to remove the redundancy and noise in the data while sampling as per tweets.
## Preprocessing:
### Feature Selection:
*“Bag of words”* model was considered in feature selection where a corpus was formed from all the tweets in the training data and specific words or idioms were selected as features from those. Three different methods of feature selection were tried to analyze and see how these schemes were selecting features for a common classifier. These are a Bag of N-grams representation in the form of a vectorizer.
1. **Term counts**:  Here we tokenize the tweets and give an integer for each possible token (white spaces as separators). Each individual token occurrence frequency in each tweet is treated as a feature. Vector of all frequencies for a given tweet is considered a sample. This representation will be sparse as the tweets will be using only a very small subset of words from the corpus.
2. **Hashing** (based on counts): This is much similar method to the above one based on term counts, but it uses a hashing trick that holds an in-memory mapping of the tokens.
3. **Term frequency-Inverse Document Frequency**(TF-IDF): This method of feature vector representation was more effective, as it weighs the words that are selected as features per their importance based on the frequency of occurrence in tweets using inverse document frequency.
### Training and Classification:
During the training phase, I used10-fold cross validation give us a measure of classifier performance on training data so that it doesn’t overfit on training dataset.
Below are the various models that I employed for classification of the tweets:
- Multinomial Naïve Bayes
- Support Vector Machine
- SGD (Stochastic gradient descent)
- Logistic Regression
- Random Forest Classifier
- Adaboost Classifier

## Parameter Tuning:

### Manual Tweaking (by trial):
Manually tried several parameters (say like ngrams or smoothing parameter for MultinomialNB or a penalty parameter alpha for SGD classifier) for individual models to see how that affects the classifier performance.

### Grid Search:
I used an exhaustive search for parameter tuning to select optimal parameters that give best performance of our model where we search on a range of parameters to tune the hyperparameters that best suit the classifiers.

### Additional optimizations
• built pipelines using the vectorizer, transformer and classifiers to fit, transform and train our model, making it concise, and single purposed.
• As the feature selection, cleaning tweets, running pipelines for different models, cross-validation and grid search for optimal parameters is heavily computational intensive, I made use of pickling, so that processed data can be written to flat files on disk and can be read from that.
• Also, used flags to turn pickling on/off, and also one for turning cross-validation on/off.
• can automate abstract pipeline to extend numerous classifiers.

## Evaluation metrics and results:

### Metrics 
1. Accuracy
2. Precision
3. Recall
4. F-score

### Results:
Below are the results using the performance metrics mentioned above for Romney and Obama. Also specified are the Classifier names and the hyperparameters used for each.
SVC (C=1, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma='auto', kernel='linear', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
Prediction accuracy of Obama on test data:  0.563076923077

|          |    Precision    |    Recall    |    F1-   Score    |    Support    |
|----------|-----------------|--------------|-------------------|---------------|
|    -1    |    0.57         |    0.64      |    0.60           |    687        |
|    0     |    0.54         |    0.52      |    0.53           |    681        |
|    1     |    0.58         |    0.52      |    0.55           |    582        |

Prediction accuracy of Romney on test data:  0.647548566142

|          |    Precision    |    Recall    |    F1-   Score    |    Support    |
|----------|-----------------|--------------|-------------------|---------------|
|    -1    |    0.71         |    0.81      |    0.76           |    1920       |
|    0     |    0.39         |    0.25      |    0.30           |    555        |
|    1     |    0.58         |    0.53      |    0.55           |    768        |

Logistic Regression (C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=None, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
Orediction accuracy of Obama on test data:  0.551794871795

|          |    Precision    |    Recall    |    F1-   Score    |    Support    |
|----------|-----------------|--------------|-------------------|---------------|
|    -1    |    0.56         |    0.63      |    0.59           |    687        |
|    0     |    0.52         |    0.51      |    0.51           |    681        |
|    1     |    0.58         |    0.51      |    0.54           |    582        |

Prediction accuracy of Romney on test data:  0.671600370028

|          |    Precision    |    Recall    |    F1-   Score    |    Support    |
|----------|-----------------|--------------|-------------------|---------------|
|    -1    |    0.68         |    0.94      |    0.79           |    1920       |
|    0     |    0.60         |    0.11      |    0.18           |    555        |
|    1     |    0.66         |    0.41      |    0.50           |    768        |

Random Forest Classifier (bootstrap=True, class_weight=None, criterion='gini', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)
Prediction accuracy of Obama on test data:  0.493333333333

|          |    Precision    |    Recall    |    F1-   Score    |    Support    |
|----------|-----------------|--------------|-------------------|---------------|
|    -1    |    0.54         |    0.44      |    0.48           |    687        |
|    0     |    0.48         |    0.46      |    0.47           |    681        |
|    1     |    0.48         |    0.60      |    0.53           |    582        |

Prediction accuracy of Romney on test data:  0.612395929695

|          |    Precision    |    Recall    |    F1-   Score    |    Support    |
|----------|-----------------|--------------|-------------------|---------------|
|    -1    |    0.67         |    0.82      |    0.73           |    1920       |
|    0     |    0.33         |    0.19      |    0.24           |    555        |
|    1     |    0.55         |    0.41      |    0.47           |    768        |


MultinomialNB (alpha=1.0, class_prior=None, fit_prior=True): 

Prediction accuracy of Obama on test data:  0.572307692308

|          |    Precision     |    Recall    |    F1-   Score    |    Support    |
|----------|------------------|--------------|-------------------|---------------|
|    -1    |    0.55          |    0.69      |    0.61           |    687        |
|    0     |    0.55          |    0.52      |    0.54           |    681        |
|    1     |    0.64          |    0.49      |    0.55           |    582        |

SGDClassifier(alpha=0.001, average=False, class_weight=None, epsilon=0.1, eta0=0.0, fit_intercept=True, l1_ratio=0.15, learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5, random_state=42, shuffle=True, verbose=0, warm_start=False)
Prediction accuracy of Obama on test data:  0.558461538462

|          |    Precision    |    Recall    |    F1-   Score    |    Support    |
|----------|-----------------|--------------|-------------------|---------------|
|    -1    |    0.57         |    0.61      |    0.59           |    687        |
|    0     |    0.56         |    0.46      |    0.50           |    681        |
|    1     |    0.54         |    0.62      |    0.58           |    582        |

AdaBoost Classifier (algorithm='SAMME.R', base_estimator=None, learning_rate=1.0, n_estimators=100, random_state=None)
Prediction accuracy of Obama on test data:  0.492820512821

|          |    Precision    |    Recall    |    F1-   Score    |    Support    |
|----------|-----------------|--------------|-------------------|---------------|
|    -1    |    0.57         |    0.44      |    0.50           |    687        |
|    0     |    0.47         |    0.46      |    0.46           |    681        |
|    1     |    0.46         |    0.60      |    0.52           |    582        |

Prediction accuracy of Romney on test data:  0.607462226334

|          |    Precision    |    Recall    |    F1-   Score    |    Support    |
|----------|-----------------|--------------|-------------------|---------------|
|    -1    |    0.65         |    0.85      |    0.74           |    1920       |
|    0     |    0.46         |    0.10      |    0.16           |    555        |
|    1     |    0.46         |    0.38      |    0.42           |    768        |

## Conclusion:
Given above are the set of models on which data is trained and evaluated. After assessing all the models, SVM was found to give best results for Romney data(Which was mostly skewed) and MultinomiaNB for Obama.
