# web-ui RAG pipeline

A Python script that uses Langchain process a csv database and chain it to an `Ollama3.2` model, embed data and store it to a FAISS index using the `mxbai-embed-large` model.  
this branch returns a simple true or false to all queries without any further explaining.

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
## Usage

Run the `boot.bat` file to use.

> **Note:** Do not close the terminals as that will close the servers.

Swapping the `vector.py` file with either `vector_chroma.py` or `vector_faiss.py` will swap the vector library used.
