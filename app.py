import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="AI PDF Assistant", layout="wide")
st.title("📄 AI PDF Assistant")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Google Gemini API Key:", type="password")
    st.markdown("[Get a free API key here](https://google.com)")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.get_buffer())
            
        st.success("File uploaded successfully! Processing...")
        
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        
        system_prompt = (
            "You are an expert academic assistant. Use the following pieces of retrieved context "
            "to answer the question. If you don't know the answer, say that you don't know.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        os.remove(temp_path)
        st.info("PDF indexed! Ask your questions below.")
        
        user_query = st.text_input("Ask something about the document:")
        if user_query:
            with st.spinner("Analyzing document..."):
                response = rag_chain.invoke({"input": user_query})
                st.markdown("### 🤖 Answer:")
                st.write(response["answer"])
else:
    st.warning("Please enter your Gemini API key in the sidebar to activate the application.")
