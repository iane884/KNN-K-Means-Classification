import math
import starter
import random
import numpy as np
import pandas as pd

def test_distance_functions():
    # Test cases for euclidean function
    print("Testing euclidean function:")
    print(f"euclidean([1, 2, 3], [4, 5, 6]) = {starter.euclidean([1, 2, 3], [4, 5, 6]):.4f}, Expected: 5.1962")
    print(f"euclidean([0, 0], [3, 4]) = {starter.euclidean([0, 0], [3, 4]):.4f}, Expected: 5.0000")
    print(f"euclidean([1, 1], [1, 1]) = {starter.euclidean([1, 1], [1, 1]):.4f}, Expected: 0.0000")
    print(f"euclidean([-1, -2], [1, 2]) = {starter.euclidean([-1, -2], [1, 2]):.4f}, Expected: 4.0000")
    
    # Test case for unequal vector lengths
    print("euclidean([1, 2], [1, 2, 3]):", starter.euclidean([1, 2], [1, 2, 3]))
    
    print("\nTesting cosim function:")
    print(f"cosim([1, 2, 3], [4, 5, 6]) = {starter.cosim([1, 2, 3], [4, 5, 6]):.4f}, Expected: 0.9746")
    print(f"cosim([1, 0], [0, 1]) = {starter.cosim([1, 0], [0, 1]):.4f}, Expected: 0.0000")
    print(f"cosim([1, 1], [1, 1]) = {starter.cosim([1, 1], [1, 1]):.4f}, Expected: 1.0000")
    print(f"cosim([-1, -2], [1, 2]) = {starter.cosim([-1, -2], [1, 2]):.4f}, Expected: -1.0000")
    
    # Test case for unequal vector lengths
    print("cosim([1, 2], [1, 2, 3]):", starter.cosim([1, 2], [1, 2, 3]))

def test_knn():
    train = starter.read_data('mnist_train.csv')
    query = starter.read_data('mnist_test.csv') 
    metric = "cosim"

    labels = starter.knn(train, query, metric, 0)
    print(f"knn base accuracy: {starter.get_accuracy(labels, [q[0] for q in query])}")
    print(starter.create_matrix(labels, [q[0] for q in query]))

    labels = starter.knn(train, query, metric, 1)
    print(f"knn cosim greyscale accuracy: {starter.get_accuracy(labels, [q[0] for q in query])}")
    print(starter.create_matrix(labels, [q[0] for q in query]))

    labels = starter.knn(train, query, metric, 2)
    print(f"knn cosim pca accuracy: {starter.get_accuracy(labels, [q[0] for q in query])}")
    print(starter.create_matrix(labels, [q[0] for q in query]))

    metric = "euclidean"

    labels = starter.knn(train, query, metric, 0)
    print(f"knn euc base accuracy: {starter.get_accuracy(labels, [q[0] for q in query])}")
    print(starter.create_matrix(labels, [q[0] for q in query]))

    labels = starter.knn(train, query, metric, 1)
    print(f"knn euc greyscale accuracy: {starter.get_accuracy(labels, [q[0] for q in query])}")
    print(starter.create_matrix(labels, [q[0] for q in query]))

    labels = starter.knn(train, query, metric, 2)
    print(f"knn euc pca accuracy: {starter.get_accuracy(labels, [q[0] for q in query])}")
    print(starter.create_matrix(labels, [q[0] for q in query]))

def test_kmeans():
    train = starter.read_data('mnist_train.csv')
    query = starter.read_data('mnist_test.csv') 
    metric = "cosim"

    labels = starter.kmeans(train, query, metric, 0)
    print(f"kmeans cosim base accuracy: {starter.get_accuracy(labels, [q[0] for q in query])}")
    print(starter.create_matrix(labels, [q[0] for q in query]))

    labels = starter.kmeans(train, query, metric, 1)
    print(f"kmeans cosim bin accuracy: {starter.get_accuracy(labels, [q[0] for q in query])}")
    print(starter.create_matrix(labels, [q[0] for q in query]))

    labels = starter.kmeans(train, query, metric, 2)
    print(f"kmeans cosim pca accuracy: {starter.get_accuracy(labels, [q[0] for q in query])}")
    print(starter.create_matrix(labels, [q[0] for q in query]))
    
    metric = "euclidean"

    labels = starter.kmeans(train, query, metric, 0)
    print(f"kmeans euc base accuracy: {starter.get_accuracy(labels, [q[0] for q in query])}")
    print(starter.create_matrix(labels, [q[0] for q in query]))

    labels = starter.kmeans(train, query, metric, 1)
    print(f"kmeans euc bin accuracy: {starter.get_accuracy(labels, [q[0] for q in query])}")
    print(starter.create_matrix(labels, [q[0] for q in query]))

    labels = starter.kmeans(train, query, metric, 2)
    print(f"kmeans euc pca accuracy: {starter.get_accuracy(labels, [q[0] for q in query])}")
    print(starter.create_matrix(labels, [q[0] for q in query]))



def test_part2():
    train = starter.read_data('train_a.txt')
    test = starter.read_data('test_a.txt')



#print(pca_preprocessing(train))

# Run the tests
test_kmeans()
test_knn()
#test_distance_functions()