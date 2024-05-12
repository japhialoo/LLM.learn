from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langsmith import Client
import datetime
import uuid
import os

'''
This file contains the code to generate the evaluation data for the Lecture Material Experiment
'''

# initialise environmental values
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
print('Enter your LangChain API Key: ', end='')
key = input()
os.environ["LANGCHAIN_API_KEY"] = key

# initialise client
client = Client()

directories = ["evalSlides", "evalTranscripts"]

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
Answer:"""
        
# initialise questions and marking schemes
examples = [
    ('Describe the process that occurs when an operating system makes a discretionary access control decision for a principal performing an operation on an object. You may use either Windows or Linux as an example. [5 Marks]', 
     '''Understanding of DAC (1 mark): Award 1 mark for a clear explanation of discretionary access control (DAC) and its role in operating systems.
        Identification of Relevant Factors (1 mark): Award 1 mark for mentioning key factors involved in the DAC decision-making process, such as user identity, object ownership, and permissions.
        Explanation of Decision Process (1 mark): Award 1 mark for describing how the operating system verifies the user's permissions based on these factors to determine whether access should be granted or denied.
        Example Utilization (1 mark): Award 1 mark for providing a relevant example, either from Linux or Windows, that illustrates the described process in action.
        Clarity and Coherence (1 mark): Assess the clarity and coherence of the response, including language use and organization. Award 1 mark for a well-structured and clear explanation.
        Total Marks Available: 5'''),
    
    ('A computer network uses a signature-based intrusion detection system such as Snort to monitor network packets for possible attacks. How effective will this system be for previously unknown threats? What effects, if any, would the pervasive use of encryption have on the ability of a system like this to function well? [4 Marks]',
     '''Understanding of Signature-Based Intrusion Detection (1 mark): Award 1 mark for demonstrating an understanding of how signature-based intrusion detection systems operate to monitor network packets for potential attacks.
        Assessment of Effectiveness for Unknown Threats (1 mark): Award 1 mark for recognizing the limitation of signature-based systems in detecting previously unknown threats.
        Consideration of Encryption's Impact (1 mark): Award 1 mark for explaining how the pervasive use of encryption would hinder the effectiveness of a signature-based system, particularly in terms of analyzing encrypted data for attack signatures.
        Clarity and Coherence (1 mark): Assess the clarity and coherence of the response, including language use and organization. Award 1 mark for a well-structured and clear explanation.
        Total Marks Available: 4''')
]

# load llm
model_path = "Models\Starling-LM-7B-beta-Q4_K_M.gguf"
llm = LlamaCpp(model_path=model_path, 
            n_ctx=50000, 
            n_batch=512, 
            max_tokens=-1, 
            verbose=True, 
            stop=["##", "Question","<|user|>"],
            callbacks=[StreamingStdOutCallbackHandler()],
            streaming=True)

dataset_name = "Training Evaluation Data"

if client.has_dataset(dataset_name=dataset_name):
    dataset = client.read_dataset(dataset_name=dataset_name)
else:
    dataset = client.create_dataset(dataset_name=dataset_name)

for q, a in examples:
    client.create_example(inputs={"question": q}, 
                        outputs={"answer": a}, 
                        dataset_id=dataset.id)


for dir in directories:
    # load embedding function and vector stores
    embedding_function = HuggingFaceEmbeddings(model_name='thenlper/gte-large')
    vectorstore = Chroma(persist_directory=dir, embedding_function=embedding_function)

    project_name = f"Training on {dir} {str(uuid.uuid4())}"
    rag_prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"])

    # initialise RetrievalQA chain
    chain_type_kwargs = {"prompt": rag_prompt}

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )
    
    client.run_on_dataset(project_name=project_name,
                        dataset_name=dataset_name,
                        llm_or_chain_factory= qa_chain,
                        concurrency_level=1)