#used for text to vector conversion
from langchain_ollama import OllamaEmbeddings
#used for embedding managment, can be used for RAG
from langchain_chroma import Chroma
#for data managment, can be used to attribute metadata
from langchain_core.documents import Document
#os import
import os
#for data analasys, mainly reading the csv database file
import pandas
#using panda to read the database
dataframe = pandas.read_csv("./pipelines/raw data/facts.csv")

#embedding model setup, model can be changed
embeding = OllamaEmbeddings(model="mxbai-embed-large")

#database info
db_location = "./chrome_langchain_db"
collection_name = "Facts_database"
documents = []
ids = []

# Check if db already exists
if os.path.exists(db_location) and os.listdir(db_location):
    # Load existing vector store
    facts_vector = Chroma(
        collection_name=collection_name,
        persist_directory=db_location,
        embedding_function=embeding
    )
else:
    #using pandas to read the database

    #converting the csv into a document
    for i, row in dataframe.iterrows():
        document = Document(
            page_content=row["Fact"] + " is " + row["fact check"],
            metadata={"reliability": row["Reliability"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

    #initiating a vector store
    facts_vector = Chroma(
        collection_name=collection_name,
        persist_directory=db_location, #THIS SAVES THE DATA
        embedding_function=embeding
    )
    facts_vector.add_documents(documents=documents, ids=ids)
#facts_vector.persist() can be used to keep vector store between sessions DOESMNT WORK!

#setup of vector for usage with the ai model by turning the vector store into a retriever that sends data to the ai
retriever = facts_vector.as_retriever(
    search_kwargs={"k": 3} #keyword for closest vector, with a parameter of 1 indicating the number of results to return
)