from src.helper import load_pdf, text_split, download_huggingface_embeddings, create_pinecone_vector_store
from src.constants import index_name, namespace
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Load data from PDF
extracted_data = load_pdf()

# Split in chunks
text_chunks = text_split(extracted_data)

#Initialize embeddings
embeddings = download_huggingface_embeddings()

# Initialize Pinecone
vectorstore = create_pinecone_vector_store(PINECONE_API_KEY, index_name, embeddings, namespace)

# Upload data to pinecone
vectorstore.add_documents(text_chunks)
