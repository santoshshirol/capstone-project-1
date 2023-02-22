import streamlit as st
import string
import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from sentence_transformers import SentenceTransformer, util
import templates as tp
import numpy as np


def getData(search_string):
    print("Elastic Search connection: "+str(client.ping()))
    token_vector = get_vector(search_string,search_string)
    query = {
        "field": "product_vector",
        "query_vector": token_vector,
        "k": 10,
        "num_candidates": 100
    }
    data  = client.search(index="myproducts", knn=query, source=["product_description"])['hits']['hits']
    print("Data from ES: ")
    print(data)
    return data




# nltk library Stop words list
stop_words = {'hasnt', 'for', 'ma', 'up', 'should', 'which', 'now', 'her', 'so', 'these', 'don', 'll', 'youd', 'against', 'doing', 'my', 'mightnt', 'him', 'but', 'is', 'dont', 'shouldve', 'arent', 'then', 'during', 't', 'above', 'once', 'shouldn', 'we', 'themselves', 're', 'was', 'needn', 'herself', 'has', 'be', 'as', 'from', 'until', 'between', 'his', 'hadn', 'mustn', 'under', 'too', 'through', 'mustnt', 'can', 'ours', 'theirs', 'me', 'you', 'shouldnt', 'she', 'over', 'or', 'isn', 'in', 'your', 'haven', 'ourselves', 'again', 'further', 'when', 'no', 'o', 'he', 'what', 'himself', 'all', 'after', 'will', 'been', 'have', 'not', 'being', 'other', 'having', 'few', 'both', 'than', 'that', 'it', 'some', 'about', 'their', 'whom', 'its', 'are', 'had', 'out', 'into', 'where', 've', 'our', 'the', 'youve', 'them', 'nor', 'just', 'while', 'am', 'down', 'd', 'm', 'of', 'doesnt', 'why', 'hers', 'shant', 'wasn', 'havent', 'hadnt', 'aren', 'wouldnt', 'who', 'by', 'here', 'shan', 'didn', 'such', 'own', 'below', 'neednt', 'same', 'if', 'off', 'myself', 'a', 'each', 'this', 'thatll', 'how', 'youre', 'does', 'yourselves', 'do', 'very', 'isnt', 'any', 'wont', 'werent', 'those', 'because', 'yourself', 'y', 'won', 'did', 'at', 'couldnt', 'more', 'its', 'there', 'with', 'on', 'itself', 'only', 'an', 'before', 'mightn', 'yours', 'ain', 'they', 'wasnt', 'were', 'doesn', 'shes', 'weren', 'most', 'didnt', 'hasn', 'youll', 'wouldn', 'to', 's', 'i', 'couldn', 'and'}


# make all text lowercase
def text_lowercase(text):
    return text.lower()


# remove stopwords
def remove_stopwords(text):
    words = text.split()
    text = [i for i in words if not i in stop_words]
    text = ' '.join(text)
    return text


# remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


# preprocess text string
def preprocessing(text):
    text = text_lowercase(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    return text

model = SentenceTransformer('all-MiniLM-L6-v2')
client = Elasticsearch(['http://localhost:9200'], http_auth=("elastic", "NzAQmqmHd7-ouB+fo9-4"))
def get_vector(product_description,category=""):
    productandcategory = product_description+category
    product_details = preprocessing(productandcategory)
    products  = [productandcategory]
    products_embeddings = model.encode(products)
    _ = list(products_embeddings.flatten())
    encod_np_array = np.array(_)
    products_embeddings_list = encod_np_array.tolist()
    #print(len(products_embeddings_list))
    return products_embeddings_list
    
    
def main():
    st.title('Semantic Search')
    search = st.text_input('Enter search words:')
    print("search string: "+search)
    data = [{"product_description" : ''}]
    if search:
        data = getData(search)
        results = data
   
    # render popular tags as filters
   
        # search results
        for i in range(len(data)):
            result = data[i]
            print(result)
            res = result['_source']
            print("Call of res")
            print(res)
            res['product_description'] = res['product_description']
            print(" Call of result['product_description']")
            print(res['product_description'])
            st.write(tp.search_result(i, res['product_description']), unsafe_allow_html=True)

if __name__ == '__main__':
    main()