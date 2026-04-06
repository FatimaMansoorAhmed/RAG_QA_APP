from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(file_path):
    """Loads a PDF and splits it into searchable chunks."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # We use a 1000 character window with a 100 character 'memory' overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)