import os
data_path="./pipelines/raw data/reference"

def retrieve_data(id):
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if os.path.isfile(file_path) and filename.startswith(str(id)+"_"):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
    return "reply **ONLY** with the facts bellow"