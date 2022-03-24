from elasticsearch import Elasticsearch
import pandas as pd

def main():
    es = Elasticsearch("http://localhost:9200")

    text = input("What is the book you want to search for?\n")
    
    query = {
        "match": {
            "book_title":text
        }
    }

    resp = es.search(index='books', query=query)
    results = pd.json_normalize(resp['hits']['hits'])
    print("The results of the search are:")
    print(results[:]["_source.book_title"])

if __name__ == "__main__":
    main()
