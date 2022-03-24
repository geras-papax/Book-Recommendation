from elasticsearch import Elasticsearch, helpers
import pandas as pd
import json

def main():
    es = Elasticsearch("http://localhost:9200")
    #insert csv to dataframe
    df = pd.read_csv('BX-Books.csv')
    #dataframe to json
    j = df.to_json(orient='records')
    #json to dict to bulk insert
    json_data = json.loads(j)
    #delete index if exists
    es.indices.delete(index='books', ignore=[400, 404])
    #bulk insert of the data
    helpers.bulk(es, json_data, index='books')
    print("Data successfully loaded!")

if __name__ == "__main__":
    main()