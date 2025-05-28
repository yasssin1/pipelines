#os import
import os
#used for text to vector conversion
from langchain_ollama import OllamaEmbeddings
#used for embedding managment, can be used for RAG
import faiss

#for data managment, can be used to attribute metadata
from langchain_core.documents import Document

#for data analasys, mainly reading the csv database file
import pandas
#using panda to read the database
import numpy as np
import json

#embedding model setup, model can be changed
embeding = OllamaEmbeddings(model="mxbai-embed-large")

#database info
db_location = "./data/faiss_langchain_db"
document_location = "./data/document_store.json"
collection_name = "Facts_database"
documents = []
ids = []
embeddings = [] #initialising embeddings list, so we can turn 

#using pandas to read the database
dataframe = pandas.read_csv("./pipelines/facts.csv")

#document creation
if os.path.exists(document_location):
    with open(document_location, "r", encoding="utf-8") as f:
        doc_data = json.load(f)
        documents = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in doc_data
        ]
else:
    #converting the csv into a document
    for i, row in dataframe.iterrows():
        document = Document(
            page_content=row["Fact"] + " is " + row["fact check"],
            metadata={"reliability": row["Reliability"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i)) #saving backup of all ids
        documents.append(document)

    os.makedirs(os.path.dirname(document_location), exist_ok=True)
    with open(document_location, "w", encoding="utf-8") as f:
        json.dump([
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            } for doc in documents
        ], f, indent=2)


# vectoring stuff
if os.path.exists(db_location):
    # Load existing index
    index = faiss.read_index(db_location)
else:

    #embedding the document
    for document in documents:
        embedding = np.array(embeding.embed_documents([document.page_content])).reshape(1, -1)
        embeddings.append(embedding)
    
    os.makedirs(os.path.dirname(db_location), exist_ok=True)
    #initiating a index
    # d = 1024  # dimension of the embeddings, maybe change
    #dynamically getting the dimentions in case of embedder change
    sample_embedding = embeding.embed_documents([documents[0].page_content])
    d = len(sample_embedding[0])
    index = faiss.IndexFlatL2(d) #creating index, L2 is the type of distance (euclidian here)
    # print(np.array(embeddings).shape)
    index.add(np.vstack(embeddings).astype("float32"))

    faiss.write_index(index, db_location) #saving the db


if os.path.exists(db_location):
    index = faiss.read_index(db_location)
    # Sanity check
    if index.ntotal != len(documents):
        raise ValueError(f"FAISS index size {index.ntotal} does not match documents count {len(documents)}")

#setup of vector for usage with the ai model by turning the vector store into a retriever that sends data to the ai
def retriever(query, k=1):
    query_vector = embeding.embed_documents(query)  #get the query embedding
    query_vector = np.array(query_vector).astype("float32").reshape(1, -1)  #1 row unknown collumns, -1 auto configs collumns

    #perform the search on the faiss index
    distances, indexes  = index.search(query_vector, k) #has 2 returns distances is the distance between query and retrieved vector
    #we only need the index to search our documents list
    #collect the retrieved documents based on indexes 
    retrieved_documents = []
    for idx in indexes [0]:
        if 0 <= idx < len(documents):
            retrieved_documents.append(documents[int(idx)]) #retrieving ids and matching them to the documentjson
        else:
            print(f"Warning: retrieved index {idx} out of range.") #case of empty results
    
    return retrieved_documents
