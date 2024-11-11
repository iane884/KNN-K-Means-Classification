import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from collections import defaultdict
from math import sqrt

# function to load data with genre and demographics included
def load_data(filepath):
    # initialize dicts for ratings, genres, demographics
    ratings_data = defaultdict(dict)
    genres = {}
    demographics = {}

    with open(filepath, 'r') as file:
        next(file)  # skip header row
        for line in file:
            # parse each line into variables
            user, movie, score, title, genre, age, gender, job = line.strip().split('\t')
            user, movie, score, age = int(user), int(movie), float(score), int(age)
            
            # store rating details
            ratings_data[user][movie] = score
            genres[movie] = genre
            
            # pull user demographic info if not already stored
            if user not in demographics:
                demographics[user] = {'age': age, 'gender': gender, 'occupation': job}
    
    return ratings_data, genres, demographics

# function to split user ratings into training/test sets
def separate_ratings(user_data, test_ratio=0.2):
    # liked movies = rating of at least 4
    liked = [movie for movie, score in user_data.items() if score >= 4]
    others = [movie for movie in user_data if movie not in liked]

    # calculate test set size
    liked_size = max(1, int(len(liked) * test_ratio)) if liked else 0
    others_size = int(len(others) * test_ratio)

    # randomly select movies for test set
    test_liked = set(random.sample(liked, liked_size)) if liked_size > 0 else set()
    test_others = set(random.sample(others, others_size)) if others_size > 0 else set()

    # combine test movies and create training and test sets
    test_set = test_liked.union(test_others)
    train_data = {movie: score for movie, score in user_data.items() if movie not in test_set}
    test_data = {movie: score for movie, score in user_data.items() if movie in test_set}

    return train_data, test_data

# function to calculate similarity between 2 users based on Euclidean distance metric 
def euclidean(user_ratings, other_ratings):
    # movies shared by both users
    shared_movies = set(user_ratings.keys()).intersection(other_ratings.keys())
    if not shared_movies:
        return 0
    # calculate squared differences for shared movies
    squared_diffs = sum((user_ratings[movie] - other_ratings[movie]) ** 2 for movie in shared_movies)
    # similarity = inverted differences
    return 1 / (1 + sqrt(squared_diffs))

# function to calculate similarity between 2 users based on cosine metric 
def cosine(user_ratings, other_ratings):
    shared_movies = set(user_ratings.keys()).intersection(other_ratings.keys())
    if not shared_movies:
        return 0
    
    # initialize arrays for shared movie ratings  
    u_vector = np.array([user_ratings[movie] for movie in shared_movies])
    o_vector = np.array([other_ratings[movie] for movie in shared_movies])

    # calculate cosine similarity
    num = np.dot(u_vector, o_vector)
    denom = np.linalg.norm(u_vector) * np.linalg.norm(o_vector)
    return num / denom if denom != 0 else 0


# # function to calculate similarity between 2 users based on cosine metric 
def pearson(user_ratings, other_ratings):
    shared_movies = set(user_ratings.keys()).intersection(other_ratings.keys())
    if len(shared_movies) < 2:
        return 0
    u_vector = np.array([user_ratings[movie] for movie in shared_movies])
    o_vector = np.array([other_ratings[movie] for movie in shared_movies])

    # return 0 if vectors are constant
    if np.std(u_vector) == 0 or np.std(o_vector) == 0:
        return 0

    # calculate pearson similarity, handle NaNs
    correlation, _ = pearsonr(u_vector, o_vector)
    return correlation if not np.isnan(correlation) else 0

# function to calculate similarity between genres from overlap
def genre_similarity(movie_id, user_movies, genres):
    # find genres user likes
    user_genres = set(genres.get(movie, "") for movie in user_movies if movie in genres)
    # find genre of recommended movie
    movie_genre = set([genres.get(movie_id, "")])
    # calculate overlap as ratio of matching genres to all user genres
    matches = user_genres.intersection(movie_genre)
    return len(matches) / max(len(user_genres), 1) if user_genres else 0

# function to calculate demographic similarity between 2 users based on age, gender, occupation
def demographic_similarity(user1, user2, demographics):
    # return max similarity if demographic data missiong
    if user1 not in demographics or user2 not in demographics:
        return 1
    
    # extract demographic data for user1 and user2
    d1, d2 = demographics[user1], demographics[user2]
    similarity_score = 0

    # increase similarity score if users in same age range
    if abs(d1['age'] - d2['age']) <= 10:
        similarity_score += 0.25

    # increase score if users have same gender
    if d1['gender'] == d2['gender']:
        similarity_score += 0.25

    # increase score if users have same occupation
    if d1['occupation'] == d2['occupation']:
        similarity_score += 0.25

    return similarity_score

# function for collaborative filter
def collaborative_filter(data, target, metric, neighbors, top_n, genres, demographics):
    similarities = []

    for user in data:
        if user != target:
            # calculate base similarity for each metric
            if metric == "euclidean":
                base_similarity = euclidean(data[target], data[user])
            elif metric == "cosine":
                base_similarity = cosine(data[target], data[user])
            elif metric == "pearson":
                base_similarity = pearson(data[target], data[user])
            else:
                base_similarity = 0

            # adjust for demographic similarity
            dem_sim = demographic_similarity(target, user, demographics)
            combined_score = base_similarity * dem_sim
            similarities.append((user, combined_score))

    # sort neighbors by similarity score, select more similar users
    similar_users = sorted(similarities, key=lambda x: x[1], reverse=True)[:neighbors]

    # aggregate ratings for recommendations, adjusting for genre overlap
    recommendations = defaultdict(float)
    for user, score in similar_users:
        for movie, rating in data[user].items():
            if movie not in data[target]:
                # calculate genre overlap
                genre_overlap = genre_similarity(movie, data[target], genres)
                recommendations[movie] += score * rating * (1 + genre_overlap)

    # rank recommendations, return top N recs
    ranked_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [movie for movie, _ in ranked_recs]

# function to evaludate recommendation quality based on precision, recall, F1-score
def evaluate_recs(actual, predicted):
    # calculate number of correct predictions
    correct_predictions = len(set(actual).intersection(predicted))
    precision = correct_predictions / len(predicted) if predicted else 0
    recall = correct_predictions / len(actual) if actual else 0
    # calculate f1-score
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0
    return precision, recall, f1

# main function to load data, split ratings into training/test sets, run collaborative filter
def main():
    # load data
    data, genres, demographics = load_data('movielens.txt')

    # set target user and K/M parameters
    target_user = 1
    K = 150
    M = 10

    # split ratings into training/test sets
    user_ratings = data[target_user]
    train_ratings, test_ratings = separate_ratings(user_ratings)

    print(f"Total movies rated by user {target_user}: {len(user_ratings)}")
    print(f"Training size: {len(train_ratings)}, Test size: {len(test_ratings)}")

    # find true liked movies in test set
    true_likes = [movie for movie, score in test_ratings.items() if score >= 4]
    if not true_likes:
        print("No liked movies in the test set.")
        return

    # replace target user rating's with training set ratings
    data[target_user] = train_ratings

    # run collaboratige filter (with additional genre/demographic features)
    recs = collaborative_filter(data, target_user, "cosine", K, M, genres, demographics)


    # evaludate recommendations based on precision, recall, F1-score
    precision, recall, f1 = evaluate_recs(true_likes, recs)
    print(f"Recommendations for user {target_user}: {recs}")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

if __name__ == "__main__":
    main()
