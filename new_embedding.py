from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader


import os

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


vector_store = PineconeVectorStore(
            index_name="law-index", embedding=embeddings
        )


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=10000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)


loader = PyPDFLoader("married-women.pdf") 
pages = loader.load_and_split()

loader = CSVLoader(file_path="diabetes.csv")
datas = loader.load()

docsearch = PineconeVectorStore.from_documents(datas, embeddings, index_name="law-index")

