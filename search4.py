import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
import string
from gensim.models import word2vec
from sklearn.cluster import KMeans
from sklearn import preprocessing

def remove_numbers(s):
    s = ''.join([i for i in s if not i.isdigit()])
    return s
#vectorizing the summaries
def generateVectors(books):
    summaries = books["summary"].to_list()
    isbns = books["isbn"].to_list()

    summary_vectors = {}
    for i in range(len(summaries)):
        summaries[i] = remove_numbers(summaries[i])

    tokenized_summaries = [summary.translate(str.maketrans('', '', string.punctuation)).split() for summary in summaries]
    model = word2vec.Word2Vec(tokenized_summaries, vector_size=100, min_count=1)
    word_vectors = model.wv
    #calculating total vector for every summary
    shape = np.array(word_vectors.get_vector(tokenized_summaries[i][0])).shape
    for i in range(len(isbns)):
        temp_vector_1 = np.zeros(shape)
        for word in tokenized_summaries[i]:
            if word != "":
                temp_vector_2 = np.array(word_vectors.get_vector(word))
                temp_vector_1 += temp_vector_2
        summary_vectors[isbns[i]] = (temp_vector_1 / len(tokenized_summaries[i]))

    return summary_vectors, isbns
#clustering with kmeans
def kMeans(summary_vectors):
    
    kmeans = KMeans()

    clusters = kmeans.fit(preprocessing.normalize(summary_vectors))
    return clusters

def main():

    
    books = pd.read_csv('BX-Books.csv')
    ratings = pd.read_csv('BX-Book-Ratings.csv')
    users = pd.read_csv('BX-Users.csv')
    special_characters = "!@#$%^&*()-+?_=,<>/"""
    locations =[]
    user_map = users[['uid', 'location']].merge(ratings[['uid', 'isbn', 'rating']], how='inner', on='uid')
    #getting the country from location field
    user_map['location'] = user_map['location'].str.replace(r'^.*?,(.*?), ', '')
    for loc in user_map['location']:
        if not any(c in special_characters for c in loc) and loc!='' and loc not in locations:
            locations.append(loc)
    user_map = user_map[user_map['location'].isin(locations)]
    user_map = user_map.merge(books[['isbn', 'category']], how='inner', on='isbn')
    #creating summary vectors
    summary_vectors, isbns = generateVectors(books)
    summary_vectors = pd.DataFrame.from_dict(summary_vectors).fillna(0).T
    #clustering
    clusters = kMeans(summary_vectors)
    #shaping the dataframe 
    cluster_map = pd.DataFrame()
    cluster_map['isbn'] = isbns
    cluster_map['cluster'] = clusters.labels_
    cluster_map = cluster_map.merge(user_map[['location', 'rating', 'isbn']], how='inner', on='isbn')
    cluster_map = cluster_map.groupby(['cluster', 'location'])['rating'].mean().reset_index().sort_values(['cluster','rating'],ascending=False)
    #printing mean ratings per country for each cluster
    print(cluster_map)


if __name__ == "__main__":
    main()
