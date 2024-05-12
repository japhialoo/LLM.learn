from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma

'''
This file creates the vectorstore for lecture material experiment
'''

# load pdf
loader = PyPDFLoader("transcriptData.pdf")
pages = loader.load_and_split()

# load embedding function and create vector store
embedding_function = HuggingFaceEmbeddings(model_name='thenlper/gte-large')
vectorstore = Chroma.from_documents(documents=pages, embedding=embedding_function, persist_directory="evalTranscripts")
vectorstore.persist()

# load pdf
loader = PyPDFLoader("lectureData.pdf")
pages = loader.load_and_split()

# load embedding function and create vector store
embedding_function = HuggingFaceEmbeddings(model_name='thenlper/gte-large')
vectorstore = Chroma.from_documents(documents=pages, embedding=embedding_function, persist_directory="evalSlides")
vectorstore.persist()
