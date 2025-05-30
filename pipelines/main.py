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
        Your job is to answer the given question **only** based on the information provided below.

        You MUST respond with exactly one of the following answers:  
        "response: true."  
        or  
        "response: false."

        After your response, provide an explanation that:

        - Is a clear rewriting of the information supporting your answer,
        - Does NOT mention or refer to the source, text, or resources explicitly,
        - Directly supports why your answer is true or false based solely on the provided information,
        - Avoids any speculation, guesswork, or information not in the provided data.

        Here is the question:  
        {question}

        Here is the information you must use to answer:  
        {resources}
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
        resources = retriever(question, k=1)

        if not resources:  #failsafe
            return "I'm not sure. I couldn't find any relevant facts to check."
        resource_str = "\n".join(f"{res.metadata['source']}\n{res.page_content}" for res in resources) #transforming document to string
        #debug
        print("fetched resources:")
        # print("\n\n".join("\n".join(r.page_content.splitlines()[:4]) for r in resources))
        print(resource_str)

        response = self.chain.invoke({"question": question, "resources": resource_str})

        return response
        