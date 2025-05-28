# web-ui RAG pipeline

A Python script that uses Langchain process a csv database and chain it to an `Ollama3.2` model, embed data and store it to a FAISS index using the `mxbai-embed-large` model.

## requirements
Requires **Python3.11**, other versions may present compatibility issues

## setup

Create a virtual environment:

```bash
python -m venv venv
```

Enter the virtual environment:

```bash
./venv/Scripts/activate
```

Install the required libraries using pip:

```bash
pip install -r requirements.txt
```

## usage

Run the `boot.bat` file to use.
>**note:** do not close the terminals as that closes the servers.

swaping the `vector.py` file with either `vector_chroma` or `vector_faiss` will swap the vector library used