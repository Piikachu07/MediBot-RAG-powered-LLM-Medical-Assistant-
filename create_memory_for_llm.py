from langchain_community.document_loaders import  PyPDFLoader, DirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Load raw PDF(s)

DATA_PATH = "Data/"
def load_pdf_files(data):
    load = DirectoryLoader(data, glob = "*.pdf",
                           loader_cls = PyPDFLoader)
    documents = load.load()
    return documents 

documents = load_pdf_files(data = DATA_PATH)


# Create Chunks

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 700, chunk_overlap = 70)

    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data= documents)


# Create Vector Embeddings

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

    return embedding_model

embedding_model = get_embedding_model()

# Store embeddings in FAISS

DB_FAISS_PATH = "vectorstore/df_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)