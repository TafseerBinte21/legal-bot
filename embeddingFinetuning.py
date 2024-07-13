from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import CSVLoader
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize vector store
vector_store = PineconeVectorStore(
    index_name="law-index", embedding=embeddings
)

# Define text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# Load and preprocess CSV data
csv_loader = CSVLoader(file_path="lawcsv.csv", encoding='utf-8')
csv_datas = csv_loader.load()

# Preprocess the CSV data if needed
# Example: cleaning or structuring the data
def preprocess_data(data):
    # Implement any preprocessing steps here
    processed_data = []
    for item in data:
        # Add your preprocessing logic here
        processed_data.append(item)
    return processed_data

# Apply preprocessing
processed_csv_datas = preprocess_data(csv_datas)

# Split the processed data into chunks
documents = []
for data in processed_csv_datas:
    chunks = text_splitter.split_text(data)
    documents.extend(chunks)

# Create embeddings for each chunk
embeddings = embeddings.embed_documents(documents)

# Index the documents in Pinecone
docsearch = PineconeVectorStore.from_documents(documents, embeddings, index_name="law-index")

# Further fine-tuning or queries can be done here
