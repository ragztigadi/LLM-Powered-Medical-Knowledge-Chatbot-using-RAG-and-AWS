from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Extract the Data From PDF file

def load_pdf_file(data):
    loader = DirectoryLoader(data,
                             glob= "*pdf",
                             loader_cls=PyPDFLoader)
    documents = loader.load()

    return documents

# Split the Data into the Text Chunks

def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks= text_splitter.split_documents(extracted_data)
    return text_chunks

# Download the Hugging Face Embeddings Model

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2") 
    return embeddings