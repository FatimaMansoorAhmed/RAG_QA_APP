from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_vector_store(chunks):
    """Converts text chunks into vectors and stores them locally."""
    # This model runs locally on your CPU for FREE
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # We persist the data so it doesn't disappear when you close the app
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore