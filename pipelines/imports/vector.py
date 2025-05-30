#os import
import os
#used for text to vector conversion
from langchain_ollama import OllamaEmbeddings
#used for embedding managment, can be used for RAG
import faiss
#for data managment, can be used to attribute metadata
from langchain_core.documents import Document

import numpy as np
#file reading
import pipelines.imports.data_reader as data_reader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


#embedding model setup, model can be changed
embeding = OllamaEmbeddings(model="mxbai-embed-large")
#initialising values
data_path="./pipelines/raw data/reference"
db_location = "./data/faiss_langchain_db"
documents = []
id_list = []

if os.path.exists(db_location):
    # Load existing vector store
    vector_store = FAISS.load_local(db_location, embeding, allow_dangerous_deserialization=True)
else:
    #reading all data
    for i, filename in enumerate(os.listdir(data_path)):
        file_path = os.path.join(data_path, filename)
        if os.path.isfile(file_path):
            try:
                type = os.path.splitext(filename)[1].lower()
                if type == '.txt':
                    text = data_reader.read_txt(file_path)
                elif type == '.pdf':
                    text = data_reader.read_pdf(file_path)
                elif type == '.docx':
                    text = data_reader.read_docx(file_path)
                document = Document(
                page_content=text,
                    #storing ids in metadata helps against errors when reconstructing file, if list order changes indexes willl be different
                    metadata={"source": file_path, "id": str(i)}, 
                )
                id_list.append(str(i)) #saving copy of all ids
                documents.append(document)
            except Exception as e:
                print(f"Failed to read {filename}: {e}")


    #creating index
    index = faiss.IndexFlatL2(len(embeding.embed_query("test embed")))
    #initialising the vector store
    vector_store = FAISS(
        embedding_function=embeding,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(documents=documents, ids=id_list)
    vector_store.save_local(db_location)

def retriever(query, k=1):
    #searching for results
    results = vector_store.similarity_search(
    str(query),
    k,
    # filter={"source": "tweet"}, used to filter
    )
    return results