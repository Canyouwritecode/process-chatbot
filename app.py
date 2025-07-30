import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

st.set_page_config(page_title="📚 ProcessBot", layout="centered")
st.title("📚 ProcessBot")
st.markdown("Ask questions about your uploaded SOPs or internal docs.")

@st.cache_resource
def load_chain():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("faiss_index", embeddings)
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=retriever
    )
    return qa_chain

query = st.text_input("Enter your question:")
if query:
    chain = load_chain()
    with st.spinner("Thinking..."):
        result = chain.run(query)
    st.success(result)
