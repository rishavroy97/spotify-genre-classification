# spotify-genre-classification

Predictive Analytics on Spotify music meta-data to identify the correct genre of music


## Problem Statement:

Can we classify music genre based on some of the music attributes?


## Description:

The problem lies in the data classification domain of Predictive Analytics.
"Given a set of musical attributes of a song, can we identify the genre of the song?"


## Data Source:

For this problem statement, the **[30000 Spotify song meta-data](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs)** dataset from Kaggle has been used.

__The following can be described about the dataset:__

* The dataset has a total of 32,787 rows, each row containing columns.
* These 23 columns contain meta-data about the song.
* Few columns include: Track Name, Artist Name, Album Name, Danceability, Tempo, Loudness, etc.
* Each song has 2 fields - Genre and Sub-genre.
* Genre has 6 values - Pop, Rock, Rap, R&B, Latin and EDM.
* The goal is to predict the genre of the song.


## Feature Engineering and Data Preprocessing:

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
