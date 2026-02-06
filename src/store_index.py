from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "llm-med-bot"

if index_name not in pc.list_indexes().names():
    print("Creating Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=1024,  # for intfloat/e5-large-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    
    extracted_data = load_pdf_file(data="Data/")
    text_chunks = text_split(extracted_data)
    embeddings = download_hugging_face_embeddings()
    
    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=embeddings
    )
    print("Documents stored.")
else:
    print("Index exists. Loading...")
    embeddings = download_hugging_face_embeddings()
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

print("Ready.")