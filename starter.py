import math
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# get the accuracy of predictions
def get_accuracy(true, pred):
    sum = 0
    for i,_ in enumerate(pred):
        if pred[i] == true[i]:
            sum += 1
    return float(sum / len(pred))

def create_matrix(true, pred):
    cm = confusion_matrix(true, pred)
    printable_cm = np.array2string(cm, separator=', ')
    return  printable_cm

# returns Euclidean distance between vectors and b
def euclidean(a,b):
    if len(a) != len(b):
        print("In Euclidean, vector lengths don't match.\n")
        return 0

    dist = 0
    for i in range(len(a)):
        dist += ((a[i] - b[i])**2)

    dist = dist**(1/2)
    
    return(dist)
        
# returns Cosine Similarity between vectors and b
def cosim(a,b):

    if len(a) != len(b):
      #  print("In cosim, vector lengths don't match.\n")
        return 0

    numerator = 0
    denominator_a = 0
    denominator_b = 0
    for i in range(len(a)):
        numerator += a[i] * b[i]
        denominator_a += a[i]**2
        denominator_b += b[i]**2

    denominator_a = denominator_a ** (1/2)
    denominator_b = denominator_b ** (1/2)

    denominator = denominator_a * denominator_b

    dist = float(numerator / denominator)

    return(dist)

def greyscale_to_bin(dataset):
     # Extract pixel values and convert to float
    pixels = np.array([[float(pixel) for pixel in sample[1]] for sample in dataset])
    
    # Normalize the data
    pixels /= 255.0
    binary = (pixels > 0.5).astype(int)
    
    # Reconstruct the dataset with reduced features
    processed_dataset = []
    for i, sample in enumerate(dataset):
        processed_sample = [sample[0], binary[i].tolist()]
        processed_dataset.append(processed_sample)
    
    return processed_dataset



# make the image have 200 features instead of 784 to speed up training.
def pca_preprocessing(dataset):
    # Extract pixel values and convert to float
    pixels = np.array([[float(pixel) for pixel in sample[1]] for sample in dataset])
    
    # Normalize the data
    pixels /= 255.0
    
    # Apply PCA
    pca = PCA(n_components=200)
    data = pca.fit_transform(pixels)
    
    # Reconstruct the dataset with reduced features
    processed_dataset = []
    for i, sample in enumerate(dataset):
        processed_sample = [sample[0], data[i].tolist()]
        processed_dataset.append(processed_sample)
    
    return processed_dataset

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train,query,metric, method):
    
    # Decide how to compute distance
    if metric == "euclidean":
        distance_func = euclidean
    elif metric == "cosim":
        distance_func = cosim
    else:
        print("Metric argument in KNN is invalid.")
        return []
    
    valid = read_data('mnist_valid.csv')
    k = 1 # number of neighbors to look at


    if method == 1:
        train = greyscale_to_bin(train)
        query = greyscale_to_bin(query)
        valid = greyscale_to_bin(valid)

    if method == 2:    
        train = pca_preprocessing(train)
        query = pca_preprocessing(query)
        valid = pca_preprocessing(valid)

    train = [(label, [int(feature) for feature in features]) for label, features in train]
    query = [(label, [int(feature) for feature in features]) for label, features in query]
    valid = [(label, [int(feature) for feature in features]) for label, features in valid]

    accuracies = []
    while k < 50: # use the validation set to find the best value of k. then use that k to test on test set


        labels = [] # holds the classifications of the queries
        for q in valid:
            distances = []
            for instance in train:
                if metric == "euclidean":
                    if (len(q[1]) != len(instance[1])):
                        print("here bug")
                    distance = distance_func(q[1], instance[1])
                elif metric == "cosim":
                    if len(q[1]) != len(instance[1]):
                        print("here bug")
                    similarity = distance_func(q[1], instance[1])
                    distance = 1 - similarity
                distances.append([distance, instance[0]]) # list of [distance, training label]

            distances.sort(key=lambda x: x[0]) # sort by distance
            k_nearest_neighbors = distances[:k] # get the KNN

            # Now that we have KNN, take vote among them to see what we will classify this point as.
            labels_to_votes = dict()
            for distance_and_label in k_nearest_neighbors:
                label = distance_and_label[1]
                labels_to_votes[label] = labels_to_votes.get(label, 0) + 1 # iterate over KNN and sum the number of times each label occurs

            prediction = max(labels_to_votes, key=labels_to_votes.get)
            labels.append(prediction)

        #print(f"New Acc: {get_accuracy(labels, [v[0] for v in valid])}")

        accuracies.append([get_accuracy(labels, [v[0] for v in valid]), labels])
        k += 1
    
    best_k = max(range(k-2, 0 , -1), key=lambda i: accuracies[i][0]) + 1
    print(f"Best K for {method}: {best_k}")

    # now that we found the best k for the valid set, we check results with the test set
    query_labels = []
    for qu in query:
            query_distances = []
            for i in train:
                if metric == "euclidean":
                    if len(q[1]) != len(instance[1]):
                        print("bug")
                    q_distance = distance_func(qu[1], i[1])
                elif metric == "cosim":
                    if len(q[1]) != len(instance[1]):
                        print("bug")
                    q_similarity = distance_func(qu[1], i[1])
                    q_distance = 1 - q_similarity
                query_distances.append([q_distance, i[0]]) # list of [distance, training label]

            query_distances.sort(key=lambda x: x[0]) # sort by distance
            knn_list = query_distances[:best_k] # get the KNN

            # Now that we have KNN, take vote among them to see what we will classify this point as.
            labels_to_votes_query = dict()
            for distance_and_label_query in knn_list:
                query_label = distance_and_label_query[1]
                labels_to_votes_query[query_label] = labels_to_votes_query.get(query_label, 0) + 1 # iterate over KNN and sum the number of times each label occurs

            pred = max(labels_to_votes_query, key=labels_to_votes_query.get)
            query_labels.append(pred)

    return(query_labels)

# Add this helper function
def normalize(vector):
    norm = math.sqrt(sum(x**2 for x in vector))
    return [x/norm for x in vector] if norm > 0 else vector

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric, method):
    # Decide how to compute distance
    if metric == "euclidean":
        distance_func = euclidean
    elif metric == "cosim":
        distance_func = cosim
    else:
        print("Metric argument in KNN is invalid.")
        return []
    
    valid = read_data('mnist_valid.csv')

    k = 1 # number of means we will examine
    iterations = 50 # how many times we will update the means

    if method == 1:
        train = greyscale_to_bin(train)
        valid = greyscale_to_bin(valid)
        query = greyscale_to_bin(query)

    if method == 2:
        train = pca_preprocessing(train)
        valid = pca_preprocessing(valid)
        query = pca_preprocessing(query)

    features = [[float(pixel) for pixel in sample[1]] for sample in train]
    labels = [sample[0] for sample in train]
    all_means = []
    all_cluster_labels = []

    accuracies = []
    while k < 50: 
        #print(k)

        # randomly select k points from the training data as initial means
        means = random.sample(features, k)

        # improve the means
        for _ in range(iterations):

            # set up our dict of clusters
            clusters = {i: [] for i in range(len(means))} #index:mean

            # group all data by nearest mean and store it with its cluster
            for point, label in zip(features,labels):
                distances = []
                for mean in means:
                    if metric == "euclidean":
                        if len(point) != len(mean):
                            print("here bug")
                        distance = distance_func(point, mean)
                    elif metric == "cosim":
                        if len(point) != len(mean):
                            print("here bug")
                        similarity = distance_func(point, mean)
                        distance = 1 - similarity # similiarity is in range [-1,1] so if we subtract from one we get [0,2] for distance
                    distances.append(distance)
                nearest_mean_index = distances.index(min(distances)) # 0-k
                clusters[nearest_mean_index].append((point, label))

            # update the current means to be the centroid of their clusters
            updated_means = []
            for i in range(len(means)):
                c = clusters[i] #(point,label) in c
                if c:
                    new_mean = find_centroid([point for point, _ in c])
                    if metric == "cosim":
                        new_mean = normalize(new_mean)  # Ensure the new mean is normalized for cosine similarity
                else:
                    # Attempt 1: pick a random new point
                    #new_mean = random.choice(features)# Find the point farthest from all current means
                    # Attempt 2: find the farthest point away from the current means
                    largest_min_distance = float('-inf')
                    farthest_point = None
                    for point in features:
                        min_distance = min([distance_func(point, mean) for mean in means])
                        if min_distance > largest_min_distance:
                            largest_min_distance = min_distance
                            farthest_point = point
                    new_mean = farthest_point
                updated_means.append(new_mean)

            means = updated_means
        
        # find the majority label for each cluster
        cluster_labels = {}
        for i, c in clusters.items():
            if c:
                counts = {} # label:count
                for _, label in c: #holding (point, label) in c
                    counts[label] = counts.get(label, 0) + 1
                cluster_labels[i] = max(counts, key=counts.get)
            else:
                # idk what to do here, hopefully never happens?
                # This means that one of the final clusters has no points in it. seems unlikely
                cluster_labels[i] = 'X'

        all_means.append(means)
        all_cluster_labels.append(cluster_labels)

    
        # This section will take the query data, assign it to a cluster, and then the 0-9 digit
        # associated with that cluster
        valid_features = [[float(pixel) for pixel in sample[1]] for sample in valid]
        valid_labels = [sample[0] for sample in valid]
        preds = []
        for q in valid_features:
            distances = []
            for mean in means:
                if metric == 'euclidean':
                    if len(q) != len(mean):
                        print(len(q))
                        print(len(mean))  
                    distance = distance_func(q, mean)
                elif metric == 'cosim':
                    if len(q) != len(mean):
                        print(len(q))
                        print(len(mean))
                    similarity = distance_func(q, mean)
                    distance = 1-similarity
                distances.append(distance)
            nearest_mean_index = distances.index(min(distances))
            preds.append(cluster_labels[nearest_mean_index])
    
        accuracies.append([get_accuracy(preds, [v[0] for v in valid]), valid_labels, cluster_labels])
        k += 1
        #print(f"New Acc: {get_accuracy(preds, [v[0] for v in valid])}")
    
    best_k = max(range(k-2, 0 , -1), key=lambda i: accuracies[i][0])
    print(f"Best K: {best_k+1}")

    best_means = all_means[best_k]
    best_cluster_labels = all_cluster_labels[best_k]

    query_features = [[float(pixel) for pixel in sample[1]] for sample in query]
    preds_q = []
    for qu in query_features:
        distances_q = []
        for mean_q in best_means:
            if metric == 'euclidean':
                if len(qu) != len(mean_q):
                    print("this bug")
                    print(len(qu))
                    print(len(mean_q))
                distance_q = distance_func(qu, mean_q)
            elif metric == 'cosim':
                similarity_q = distance_func(qu, mean_q)
                distance_q = 1-similarity_q
            distances_q.append(distance_q)
        nearest_mean_index_q = distances_q.index(min(distances_q))
        preds_q.append(best_cluster_labels[nearest_mean_index_q])


    return preds_q

# Take all the points in a cluster.
# Calculate the collective midpoint.
def find_centroid(cluster):
    if not cluster:
        return None
    
    number_of_features = len(cluster[0])
    number_of_points = len(cluster)

    # Sum all of the features in each index. init to zero
    sum_feature_values = [0 for n in range(number_of_features)]
    
    # do the sum
    for point in cluster:
        for n in range(number_of_features):
            sum_feature_values[n] += point[n]

    # divide by number of features to get avg
    mean = []
    for feature_sum in sum_feature_values:
        mean.append(feature_sum / number_of_points)

    # return the midpoint of a group of points
    return mean

def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    #show('mnist_valid.csv','pixels')
    x = 4
    
if __name__ == "__main__":
    main()
    