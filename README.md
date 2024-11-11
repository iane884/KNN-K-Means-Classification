# K-Nearest Neighbors and K-Means Classification Project

## Project Overview
This project involves implementing and testing machine learning models, specifically the K-Nearest Neighbors (KNN) and K-Means algorithms, to classify data from two different datasets. In Part I, I worked with the MNIST dataset of handwritten digits to classify images based on their pixel intensity values. In Part II, I developed a collaborative filtering-based recommendation system for the MovieLens dataset to recommend movies to users based on similar user preferences.

## Part I: MNIST Handwritten Digits Classification
In Part I, I implemented a classifier for the MNIST dataset, focusing on both KNN and K-Means. The goal was to determine how effectively these models classify images of digits based on grayscale intensities.

### Steps Completed

#### Distance Metrics Implementation
I implemented functions to calculate Euclidean Distance, Cosine Similarity, Pearson Correlation, and Hamming Distance between two vectors. This was a foundation for measuring similarity in the KNN and K-Means algorithms. Testing confirmed accurate results compared with standard libraries.

#### K-Nearest Neighbors Classifier
Using the Euclidean and Cosine similarity metrics, I built a KNN classifier. I experimented with feature transformations (PCA and binary mapping) to reduce computational load and improve classification accuracy. Hyperparameters like the number of neighbors \( K \) were tuned, and I presented results in confusion matrices and analyzed the model's performance. Cosine similarity generally achieved better accuracy than Euclidean distance, with unprocessed data yielding the highest accuracy.

#### K-Means Classifier
Implementing a K-Means classifier, I tuned the number of clusters \( K \) and compared the results with those from KNN. Since labels were not used in the training process, I evaluated clustering performance by comparing cluster labels to known digit labels. Although K-Means was less accurate than KNN, binary mapping helped achieve performance comparable to that of unprocessed data.

## Part II: MovieLens Recommendation System
In Part II, I developed a recommendation system for the MovieLens dataset using collaborative filtering. The system leveraged user preferences to recommend movies to target users based on similarities with other users.

### Steps Completed

#### Collaborative Filter with Similarity Metrics
Using the distance metrics from Part I, I built a collaborative filter. I implemented logic to identify users with similar tastes and recommend movies they had rated highly. Hyperparameters, including the number of neighbors \( K \) and number of recommendations \( M \), were tuned based on validation data. I reported precision, recall, and F1-scores, analyzing how these metrics varied with different values of \( M \).

#### Enhanced Collaborative Filter with Additional Features
I improved the recommendation system by incorporating movie genres and user demographics (e.g., age, gender, occupation). Adjusting similarity scores based on these additional features led to more personalized recommendations and higher performance metrics. Precision, recall, and F1-score improved, confirming the value of genre and demographic data in collaborative filtering.
