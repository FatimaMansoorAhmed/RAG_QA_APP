from langchain_groq import ChatGroq
# CHANGE THIS: Move from langchain to langchain_core
from langchain_core.prompts import PromptTemplate 
# CHANGE THIS: Move from langchain to langchain_classic
from langchain_classic.chains import RetrievalQA
def get_qa_chain(vectorstore, api_key):
    """Sets up the RAG chain with a strict system prompt."""
    llm = ChatGroq(
        groq_api_key=api_key, 
        model_name="llama-3.3-70b-versatile", 
        temperature=0
    )

    template = """Answer the question based ONLY on the provided context.
    If the answer isn't in the context, say "I don't have enough information."
    Always mention the page number in your answer.

    Context: {context}
    Question: {question}
    Answer:"""

    prompt = PromptTemplate.from_template(template)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )