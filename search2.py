from elasticsearch import Elasticsearch
import pandas as pd

# create new custom metric function for score, rating, avg
def metric(x):
    return x[0]*0.5 + x[1]*0.25 + x[2]*0.25

def search2():
    es = Elasticsearch("http://localhost:9200")

    # sample search = 2,0195153448,classical myths
    books= pd.DataFrame()

    #-----Elasticsearch metric-----
    while books.empty:
        book = input("What is the book you want to search for?\n")
        query = {
            "match": {
                "book_title":book
            }
        }

        resp = es.search(index='books', query=query)
        books = pd.json_normalize(resp['hits']['hits'])
    scores = books["_score"]
    isbn = books["_source.isbn"]
    books = books.set_index("_source.isbn")

    results= pd.DataFrame()

    #-----User Rating-----
    search_user_rating = [0]*len(scores)
    while results.empty:
        userId = input("What is your ID?\n")
        query = {
            "match": {
                "uid":userId
            }
        }
        resp = es.search(index='ratings', query=query)
        results = pd.json_normalize(resp['hits']['hits'])
    

    for i in range(len(isbn)):
        for j in range(len(results["_source.isbn"])):
            if isbn[i] == results["_source.isbn"][j]:
                search_user_rating[i] = results["_source.rating"][j]

    #-----Other Users Avg Rating-----
    users_avg_rating = [0]*len(scores)
    for i in range(len(isbn)):
        query = {
            "match": {
                "isbn":isbn[i]
            }
        }
        resp = es.search(index='ratings', query=query)
        results = pd.json_normalize(resp['hits']['hits'])
        temp = 0
        for j in range(len(results["_source.rating"])):
            temp += results["_source.rating"][j]
        users_avg_rating[i] = temp/len(results["_source.rating"])

    print("-----------------------")
    print("Original order:")
    print(books[["_score", "_source.book_title"]])

    for i in range(len(books)):
        books["_score"][i] = metric([scores[i], search_user_rating[i], users_avg_rating[i]])

    books = books.sort_values(by=['_score'], ascending=False)
    print("-----------------------")
    print("Updated order:")
    print(books[["_score", "_source.book_title"]])
    books = books.reset_index()
    return books, userId

def main():
    search2()

if __name__ == "__main__":
    main()