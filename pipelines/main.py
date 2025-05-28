from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pipelines.imports.vector import retriever
from typing import List, Union, Generator, Iterator

class Pipeline:
    def __init__(self):
        self.basic_rag_pipeline = None
        self.chain = None

    async def on_startup(self):


        model = OllamaLLM(model="llama3.2")

        template = """
        You are a fact-checking bot, your job is to take facts and respond with one of the following: 
        "true", "false", "partially true", "partially false", "unsure"

        Do not include any explanations or extra context, just one of the above words

        Always consult the following database before responding: {facts}
        A fact within the database takes precedence over any other information.

        Here is the question to answer: {question}
        """


        #creating a promt out of the previous text
        prompt = ChatPromptTemplate.from_template(template)
        #chaining the prompt to the ollama model
        self.chain = prompt | model
        

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.
        print("messages:")
        print(messages)
        print(user_message)

        question = user_message
        facts = retriever(question, k=1)

        if not facts:  # Handle case where no relevant facts are found
            return "I'm not sure. I couldn't find any relevant facts to check."


        #extract the relevant content from the retrieved documents
        fact_texts = [doc.page_content for doc in facts]
        
        #turn info to string before passing
        facts_str = "\n".join(fact_texts)
        print("fetched data:")
        print(facts_str)
        response = self.chain.invoke({"facts":facts_str, "question": question})

        return response
