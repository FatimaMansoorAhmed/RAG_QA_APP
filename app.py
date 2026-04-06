import streamlit as st
import os
from utils import process_pdf
from ingest import build_vector_store
from query import get_qa_chain
from dotenv import load_dotenv
load_dotenv()
st.set_page_config(page_title="Pro PDF RAG", layout="centered")
st.title("🛡️ Pro Document QA")

# --- CHANGE 1: Get key from Environment ---
# On your local PC, it looks for an Environment Variable.
# On Streamlit Cloud, it looks in "Secrets".
groq_key = os.getenv("GROQ_API_KEY")

# Sidebar now only handles the file
with st.sidebar:
    st.header("Settings")
    pdf_file = st.file_uploader("Upload Document", type="pdf")

# Check if the key exists before proceeding
if not groq_key:
    st.error("Error: GROQ_API_KEY not found in environment variables.")
    st.stop()

if pdf_file:
    # 1. Save and Process
    with open("temp_upload.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())
    
    with st.status("Analyzing document..."):
        chunks = process_pdf("temp_upload.pdf")
        vectorstore = build_vector_store(chunks)
        # Pass the groq_key we got from the environment
        qa_chain = get_qa_chain(vectorstore, groq_key)
    
    # 2. Chat UI
    user_input = st.text_input("What would you like to know?")
    
    if user_input:
        with st.spinner("Thinking..."):
            response = qa_chain({"query": user_input})
            
            st.markdown("### Answer")
            st.success(response["result"])
            
            st.markdown("### Sources")
            for doc in response["source_documents"]:
                st.caption(f"Page {doc.metadata['page']+1}: {doc.page_content[:200]}...")
else:
    st.info("Please upload a PDF to begin.")