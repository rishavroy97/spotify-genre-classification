# Spotify Genre Classification

Predictive Analytics on Spotify music meta-data to identify the correct genre of music


## Problem Statement:

Can we classify music genre based on some of the music attributes?


## Description:

The problem lies in the data classification domain of Predictive Analytics.
"Given a set of musical attributes of a song, can we identify the genre of the song?"


## Step 1: Data Collection (Data Source):

For this problem statement, the **[30000 Spotify song meta-data](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs)** dataset from Kaggle has been used.

__The following can be described about the dataset:__

* The dataset has a total of 32,787 rows, each row containing columns.
* These 23 columns contain meta-data about the song.
* Few columns include: Track Name, Artist Name, Album Name, Danceability, Tempo, Loudness, etc.
* Each song has 2 fields - Genre and Sub-genre.
* Genre has 6 values - Pop, Rock, Rap, R&B, Latin and EDM.
* The goal is to predict the genre of the song.


## Step 2: Feature Engineering and Data Preprocessing:

RapidMiner has been used for the project. The entire dataset excluding the Track Date column has been retrieved for the project.
After retrieving the data, it was passed through a set of pre-processing operations (RapidMiner Processes) in the following order:


### Attribute Selection/Feature Selection

In this step we removed the textual fields such as ```playlist_name```, ```track_album_name```, ```track_name```, ```track_artist```.
These fields do not provide any real value for prediction.
```track_artist``` is a field on which Unsupervised learning can be run and then implementing the results into a supervised learning classification problem.
This falls under semi-supervised learning which is not in the scope of the project.
All the ID related fields have also been removed.
```playlist_subgenre``` has also been removed as we are only interested in predicting the main genre.


As part of feature selection, each attribute's histogram and bell curve was looked at by grouping them by ```playlist_genre```.
It was discovered that the ```key``` attribute has a similar bell distribution for every genre and cannot be used to effectively identify the genre.
Hence ```key``` column has also been excluded.


![excluded-feature-Key](https://github.com/rishavroy97/spotify-genre-classification/assets/28308372/5ea0982b-ed20-405c-b1cb-247bf6290911)

The final count of columns stands at **13** - **1 label class** and **12 feature attributes**.


### Label Identification

In this step, the ```playlist_genre``` field has been marked as a poly-nominal label field using the **Set Role** Operator.


### Handle missing values

On looking at the dataset statistics, it was observed that the dataset had a few blank rows with most attributes including the label missing value.
These rows were removed and the resultant dataset was left with **zero missing values** for any attribute for any row.


### Normalization

A few attributes, such as tempo (in beats per minute) and duration (in milliseconds) have values in 100s and 100000s. 
These attributes were fixed by passing all the attributes through the standard Normalization Operator in RapidMiner.



## Step 3: Modelling:

The following algorithms were used to train on the dataset.

### k-NN

KNN (k Nearest Neighbors) is a lazy learning algorithm used for classification and regression. This means that for training it just stores data in memory and does not perform any calculations on the training data. For testing an unseen new data point, its distance is calculated with every other data point stored in memory for training. The k nearest neighbors to this new datapoint are selected to predict the final result.

For classification tasks, the majority class among the k-nearest neighbors is assigned to the new data point. For regression tasks, the average (or weighted average) of the target values of the k-nearest neighbors is calculated and assigned to the target value of the new data point.

**Advantages of KNN:**

1. __No Training Period:__ Since KNN is a lazy learner, there is no explicit training phase. The model is built at the time of prediction using the entire dataset.
2. __Versatility:__ KNN can be used for both classification and regression tasks.

**Disadvantages of KNN**

1. __Computational Complexity:__ Calculating distances between the new data point and all training examples can be computationally expensive, especially for large datasets or high-dimensional data.
2. __Curse of Dimensionality:__ In high-dimensional spaces, the concept of distance becomes less meaningful, and KNN may perform poorly. This is known as the curse of dimensionality.

**Hyperparamters**

1. __Number of Neighbors (k):__ The number of nearest neighbors to consider when making a prediction.
2. __Distance Metric:__ The measure used to calculate the distance between data points. Common distance metrics include Euclidean distance, Manhattan distance, Minkowski distance, and others.


**Performance with KNN**

![1  k-NN Accuracy table](https://github.com/rishavroy97/spotify-genre-classification/assets/28308372/a6eded93-8763-48b1-8f67-37b7dbb4c298)



### SVM

Support Vector Machines (SVM) is a supervised machine learning algorithm used for classification and regression tasks. The primary objective of SVM is to find a hyperplane in an N-dimensional space (N being the number of features) that separates data points of one class from another with a maximum margin. SVM is effective in high-dimensional spaces and is versatile for both linear and non-linear classification.

Support Vector Machines (SVM) are inherently binary classifiers, meaning they are designed for two-class classification problems. However, several strategies can be employed to extend SVM for multi-class classification tasks. In RapidMiner, the ```libSVM``` Operator is used to classify a multi-catergory dataset.

**Advantages of SVM**

1. __Effective in High-Dimensional Spaces:__ SVMs work well in high-dimensional spaces, making them suitable for problems with a large number of features, such as image classification.
2. __Robust to Overfitting:__ SVMs are less prone to overfitting, especially in high-dimensional spaces, because they aim to maximize the margin between classes.

**Disadvantages of SVM**

1. __Limited to Binary Classification (by default):__ Traditional SVMs are binary classifiers. While there are strategies to extend them to multi-class problems, this may involve training multiple models.
2. __Computationally Intensive:__ Training an SVM can be computationally intensive, especially for large datasets. The time complexity is often between O(N^2) and O(N^3), where N is the number of training samples.

**Hyperparameters**

1. __Kernel Type (kernel):__ SVMs use kernel functions to transform the input data into a higher-dimensional space. Common kernel functions include linear, polynomial, radial basis function (RBF), and sigmoid.
2. __Regularization Parameter (C):__ C is the regularization parameter that controls the trade-off between having a smooth decision boundary and classifying the training points correctly. A smaller C allows for a more flexible boundary, potentially allowing some misclassifications, while a larger C enforces a stricter boundary.

**Performance with SVM**

![5  SVM](https://github.com/rishavroy97/spotify-genre-classification/assets/28308372/45a21b94-7fbc-422c-a4a1-3192f603c3e0)



### Decision Trees

A Decision Tree is a supervised machine learning algorithm used for both classification and regression tasks. It works by recursively partitioning the dataset into subsets based on the most significant attribute at each step, forming a tree-like structure of decisions.

For testing, a new unknown data point traverses through the decisions of a decision tree till it reaches a leaf node which specifices which category the data point should fall under.

**Advantages of Decision Trees**

1. __Handling Mixed Data Types:__ Decision Trees can handle both numerical and categorical data without the need for extensive data preprocessing, making them versatile for various types of datasets.
2. __Robust to Outliers:__ Decision Trees are not sensitive to outliers since they make decisions based on relative comparisons rather than absolute values.

**Disadvantages of Decision Trees**

1. __Overfitting:__ Decision Trees are prone to overfitting, especially when the tree is deep and not pruned. Overfitting occurs when the model fits the training data too closely, capturing noise instead of the underlying patterns.
2. __Difficulty in Capturing Complex Relationships:__ Decision Trees may struggle to capture complex relationships in the data compared to more advanced algorithms. They might oversimplify the underlying patterns.

**Hyperparameters**

1. __Criterion (criterion):__ The function used to measure the quality of a split. Common criteria include "gini" for the Gini impurity and "entropy" for information gain.
2. __Minimal Gain:__ The gain of a node is calculated before splitting it. The node is split if its gain is greater than the minimal gain parameter. A higher value of minimal gain results in fewer splits and a smaller tree.

**Performance with Decision Trees**

![4  Decision Trees](https://github.com/rishavroy97/spotify-genre-classification/assets/28308372/c4b32db9-816a-49f6-961c-82dd6790057e)



### Random Forest

Random Forest is an ensemble learning algorithm that operates by constructing a multitude of decision trees during training and outputs the mode of the classes (classification) or the mean prediction (regression) of the individual trees.

Random Forest builds multiple decision trees during training. Each tree is trained on a random subset of the data, and the final prediction is obtained by aggregating the predictions of all trees.

**Advantages of Random Forest**

1. __Robust to Overfitting:__ The combination of bagging (bootstrap aggregating) and random feature selection helps prevent overfitting, making Random Forest robust to noise and outliers in the data.
2. __Parallelization:__ The training and prediction processes of Random Forest can be easily parallelized, making it efficient for large datasets and parallel computing environments.

**Disadvantage of Random Forest**

1. __Computational Resources:__ Training a large number of decision trees can be computationally expensive, especially for large datasets and a high number of features.
2. __Lack of Interpretability:__ The ensemble nature of Random Forest can make it challenging to interpret the reasoning behind individual predictions, especially when dealing with a large number of trees.

**Hyperparameters**

1. __n_estimators:__ The number of decision trees in the ensemble. Increasing the number of trees generally improves performance, but it also increases computational cost. It is often set through cross-validation.
2. __max_depth:__ The maximum depth of each decision tree in the ensemble. A smaller max depth can help prevent overfitting, but setting it too low may lead to underfitting. Cross-validation can help determine an optimal value.


**Performance with Random Forest**

![2  Random Forest Accuracy table](https://github.com/rishavroy97/spotify-genre-classification/assets/28308372/c5ee6165-364d-47b1-8848-1c7f7f1d5cf6)



### Gradient Boosting

Gradient Boosting Trees is an ensemble learning technique that builds a series of decision trees sequentially, with each tree correcting the errors of the previous one. This iterative process leads to the creation of a powerful predictive model. The general concept of Gradient Boosting can be applied to both regression and classification problems. Gradient Boosting Trees is often used for supervised learning tasks where the goal is to predict an outcome variable based on input features.

**Advantages of Gradient Boosting**

1. __High Predictive Accuracy:__ Gradient Boosting often provides high predictive accuracy, making it one of the most powerful machine learning algorithms.
2. __Handles Nonlinear Relationships:__ It is capable of capturing complex, nonlinear relationships in the data, making it suitable for a wide range of problems.

**Disadvantages of Gradient Boosting**

1. __Tuning Complexity:__ Tuning the hyperparameters of Gradient Boosting can be complex, and the optimal values may depend on the specific characteristics of the dataset.
2. __Computational Complexity:__ Training a large number of trees can be computationally expensive, especially with deep trees. This can be a limitation for real-time applications.

**Hyperparamters**

1. __Learning Rate(learning_rate):__ A hyperparameter that scales the contribution of each tree. Lower values make the model more robust, but too low a value may require more trees to achieve the same level of performance. Typical values range from 0.01 to 0.3.
2. __Number of Trees (n_estimators):__ The total number of trees in the ensemble. Increasing the number of trees generally improves performance, but it also increases computational cost. It is often set through cross-validation.

**Performance with Gradient Boosting**

![3  Gradient Boosted Trees](https://github.com/rishavroy97/spotify-genre-classification/assets/28308372/493e12ef-e18c-44a4-aeb9-a3ce3351f7de)



## Step 5: Hyperparameter Tuning

###  Decision Trees

Without hyperparameter tuning, the accuracy for Decision trees was 23.90% which is slightly better than randomly selection one out of 6 genres (1/6 = 16.677%)
Using the most common hyperparameters for Decision Trees, a grid search was conducted between Criterion, Minimal Gain and Maximum Depth taking 2 parameters at a time.

Accuracy for the following are listed:

* Criterion and Minimal Gain - **47.87% (Criterion = information_gain and Minimal Gain = 0.001)**
* Criterion and Maximum Depth - **48.16% (Criterion = information_gain and Maximum Depth = 11)**
* Minimal Gain and Maximum Depth - **48.22% (Criterion = information_gain and Minimal Gain = 0.001 and Maximum Depth = 11)**

The best accuracy comes from the combination of 
**1. Criterion = information_gain
2. Minimal Gain = 0.001
3. Maximum Depth = 11**

![Criterion_MaxDepth](https://github.com/rishavroy97/spotify-genre-classification/assets/28308372/9fc6a18b-c70b-444c-b5d6-3e23efee1187)
![MaxDepth_MinimalGain](https://github.com/rishavroy97/spotify-genre-classification/assets/28308372/79dcee5b-1d29-4fb5-a03a-889afd9e53f8)
![Criterion_MinimalGain](https://github.com/rishavroy97/spotify-genre-classification/assets/28308372/1a9f8555-218a-42f1-9b49-5fb56cb11044)


### Gradient Boosting

Without hyperparameter tuning, the accuracy for Gradient Boosting trees was 50.42% which is better than the most optimized decision tree accuracy
Using the most common hyperparameters for Gradient Boosting, a grid search was conducted between Learning Rate and Number of trees.

The best accuracy comes from a combination of
**1. Learning Rate = 0.241
2. number of trees = 100**

**Final Accuracy - 56.1%**

![Learning Rate - accuracy](https://github.com/rishavroy97/spotify-genre-classification/assets/28308372/0a7ab076-e90c-49f9-baae-ff899e002c4d)
![Learning Rate - accuracy iterations](https://github.com/rishavroy97/spotify-genre-classification/assets/28308372/8d67e7c6-6e78-4338-8b4e-3798790a68c9)



## Step 6: Results

|Models|Accuracy|
|---|---|
|KNN|48.50%|
|SVM|36.27%|
|Decision Trees|23.90%|
|Random Forest|24.92%|
|Gradiant Boost|50.42%|
|Optimized Decision Trees|48.22%|
|Optimized Gradiant Boost|56.1%|

The accuracy for this project is on the lower side. 

Music genre classification is difficult to perform especially today where colaboration is done among artists of different genres to create music that is an amalgamation of multiple different genres.

Pop music which stands for popular music encompasses music that has become popular globally. Consequently, songs sung by R&B artists such as the Weeknd would be labelled as pop when it is technically R&B. Similarly, this dataset labels songs from popular EDM artists such as Martin Garrix as Pop. This creates a disparity between the atrributes of the song and the actual label class.

A more accurate solution would be to take the Artist name into account and perform supervised learning and perform an unsupervised learning over the rest of the song attributes.
