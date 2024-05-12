from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from queue import Queue, Empty
from threading import Thread
from pathlib import Path
from typing import Any
import gradio as gr

'''
This file is responsible for creating the environment for users to interact with the ChatBot
'''

# Create a queue and a variable to notify the UI that the text generation is ended
q = Queue()
job_done = object()

class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: Any) -> None:
        return self.q.empty()

temp = []
def answer(question: str, doc: Path):
    '''
    Method which takes the user query and starts llm generation in a separate thread
    '''
    def task():
        global temp
        load = PyPDFLoader(doc)
        context = load.load_and_split()
        embedding_function = HuggingFaceEmbeddings(model_name='thenlper/gte-large')
        
        if temp == context:
            vectorstore = Chroma(persist_directory="lectures", embedding_function=embedding_function)
        else:
            vectorstore = Chroma.from_documents(persist_directory="lectures", documents=context, embedding=embedding_function)
            vectorstore.persist()
            
        
        
        # Make prompt for LLM role
        template = """
        You are a student in university and you are given lecture slides as context to learn a topic. 
        You MUST Only Use the following piece of context given through the lecture slides to help answer questions given at the end. 
        If the question is not related to the scope of the lecture given then you MUST say you haven't learnt it yet. 
        You MUST not answer outside the scope of the context. 
        You MUST answer in a concise and factual manner without overexplaining.
        Context: {context}

        ---------------------------------
        <|user|>
        ##Question: {question}


        <|assistant|>
        Answer:
        """
        
        # prepare prompt template
        rag_prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"])
        
        chain_type_kwargs = {"prompt": rag_prompt}

        # initialise RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs=chain_type_kwargs,
        )
    
        # response = llm(template)
        response = qa_chain({"query": question})
        q.put(job_done)
        temp = context
  
    t = Thread(target=task)
    t.start()

callbacks = [QueueCallback(q), StreamingStdOutCallbackHandler()]

# Load LLM
model_path = "Models\zephyr-7b-beta.Q4_K_M.gguf"
llm = LlamaCpp(model_path=model_path,  
               n_ctx=50000, 
               n_batch=512, 
               max_tokens=-1, 
               verbose=True, 
               stop=["##", "Question","<|user|>"],
               callbacks=callbacks,
               streaming=True)

# Creating interface for user interaction with chatbot
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    user_file = gr.File(label="Document")
    user_question = gr.Textbox(label="Question", interactive=True)
    
    submit = gr.Button("Submit")
    clear = gr.Button("Clear")
    
    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history, user_file):
        question = history[-1][0]
        print("Question: ", question)
        history[-1][1] = ""
        answer(question=question, doc=user_file)
        while True:
          try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
              break
            history[-1][1] += next_token
            yield history
          except Empty:
            continue
        
    submit.click(user, 
             [user_question, chatbot], 
             [user_question, chatbot], 
             queue=False).then(bot, [chatbot, user_file], chatbot)
    
    clear.click(lambda: None, None, chatbot, queue=False)
    
# Launch Chatbot Interface
if __name__ == "__main__":
    demo.launch(share=True)

