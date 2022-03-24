from elasticsearch import Elasticsearch
import numpy as np
import pandas as pd
import string
from gensim.models import word2vec
from keras import losses
from keras.models import Sequential
from keras.layers import Dense, Softmax
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from search2 import search2

CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
FLAGS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def remove_numbers(s):
    s = ''.join([i for i in s if not i.isdigit()])
    return s

def generateVectors(books):
    summaries = books["summary"].to_list()
    isbns = books["isbn"].to_list()

    summary_vectors = {}
    for i in range(len(summaries)):
        summaries[i] = remove_numbers(summaries[i])

    tokenized_summaries = [summary.translate(str.maketrans('', '', string.punctuation)).split() for summary in summaries]
    model = word2vec.Word2Vec(tokenized_summaries, vector_size=100, min_count=1)
    word_vectors = model.wv

    shape = np.array(word_vectors.get_vector(tokenized_summaries[i][0])).shape
    for i in range(len(isbns)):
        temp_vector_1 = np.zeros(shape)
        for word in tokenized_summaries[i]:
            if word != "":
                temp_vector_2 = np.array(word_vectors.get_vector(word))
                temp_vector_1 += temp_vector_2
        summary_vectors[isbns[i]] = (temp_vector_1 / len(tokenized_summaries[i]))

    return summary_vectors

def get_pred_flags(predictions):
    return [np.argmax(prediction) for prediction in predictions]

def NNetwork(isbns, summary_vectors, es):
    data = []
    label = []
    query = {
        "match": {
            "uid":11676
        },
        "match": {
            "uid":166123
        }
    }
    
    resp = es.search(index='ratings', query=query)
    results = pd.json_normalize(resp['hits']['hits'])
    for i in range(len(results['_source.isbn'])):
        if results['_source.isbn'][i] in isbns:
            data.append(summary_vectors[results['_source.isbn'][i]])
            label.append(results['_source.rating'][i])

    trainData, testData, trainflags, testflags = train_test_split(data, label, test_size=0.25)

    model = Sequential()
    model.add(Dense(32, input_dim=len(trainData[0]), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10))

    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

    model.fit(np.array(trainData), np.array(trainflags), epochs=20, verbose=1)

    probabilityModel = Sequential([model, Softmax()])

    return probabilityModel

def get_ratings_average(ratings):
    ratings_avg = ratings.groupby(by='isbn').mean()
    ratings_avg = ratings_avg.drop('uid', axis=1).reset_index()

    return ratings_avg

def final_rating(response, userId, predictor, summary_vectors, ratings, isbns):
    final_result = []
    ratings_avg = get_ratings_average(ratings)
    for i in range(len(response)):
        if response['_source.isbn'][i] in isbns:
            isbn = response['_source.isbn'][i]
            book_score = response['_score'][i]
            book_ratings_avg = float(ratings_avg.loc[ratings_avg['isbn'] == isbn].iloc[0]['rating']) if isbn in ratings_avg.isbn else -1
            try:
                user_rating = ratings[(ratings['isbn'] == str(isbn)) & (ratings['isbn'] == str(isbn))]['rating'].item()
            except:
                user_rating = '-'
            # -----------------------------------------calculate predicted RATING-----------------------------------------
            prediction_array = predictor.predict(np.array([summary_vectors[isbn]]))
            predicted_label = get_pred_flags(prediction_array)
            user_predicted_rating = CLASSES[predicted_label[-1]]
            # ------------------------------------------------------------------------------------------------------------
            new_record = {
                "Title": response['_source.book_title'][i],
                "Score": book_score
            }
            new_record["Score"] += book_ratings_avg if book_ratings_avg != -1 else 0
            new_record["Score"] += user_rating if user_rating != "-" else user_predicted_rating
            new_record["User_true_r"] = user_rating
            new_record["User_predicted_r"] = user_rating if user_rating != '-' else user_predicted_rating
            final_result.append(new_record)
    
    df = pd.DataFrame(final_result)
    df.sort_values("Score", inplace=True, ascending=False)
    df = df.reset_index(drop=True)
    return df

def main():
    es = Elasticsearch("http://localhost:9200")

    books = pd.read_csv('BX-Books.csv')
    ratings = pd.read_csv('BX-Book-Ratings.csv')

    summary_vectors = generateVectors(books)
    isbns = books["isbn"].to_list()
    
    response, userId = search2()
    
    predictor = NNetwork(isbns, summary_vectors, es)
    
    print(final_rating(response, userId, predictor, summary_vectors, ratings, isbns))

if __name__ == "__main__":
    main()
