import streamlit as st
import os

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="AI PDF Assistant", layout="wide")
st.title("📄 AI PDF Assistant")

# 1. Look for API key automatically in Local Environment or Cloud Secrets
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

# 2. Only build the sidebar input box if the API key isn't found automatically
if not api_key:
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "Enter Google Gemini API Key:",
            type="password"
        )

# 3. Halt the application if no key is provided anywhere
if not api_key:
    st.warning("Please enter your Gemini API key in the sidebar.")
    st.stop()

# Set key to environment for downstream components
os.environ["GOOGLE_API_KEY"] = api_key

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:

    temp_path = uploaded_file.name

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    try:
        with st.spinner("Processing PDF..."):

            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            splits = splitter.split_documents(docs)

            embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001"
            )

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings
            )

            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 20}
            )

            llm = ChatGoogleGenerativeAI(
                model="gemini-3.6-flash",
                temperature=0.2
            )
            
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are an expert PDF assistant.

Use the provided context to answer questions.

If the answer is not found in the document,
say "The document does not contain that information."

Context:
{context}
""",
                    ),
                    ("human", "{input}"),
                ]
            )

            qa_chain = create_stuff_documents_chain(
                llm,
                prompt
            )

            rag_chain = create_retrieval_chain(
                retriever,
                qa_chain
            )

        st.success("PDF indexed successfully!")

        query = st.text_input(
            "Ask a question about the PDF:"
        )

        if query:
            with st.spinner("Thinking..."):
                result = rag_chain.invoke(
                    {"input": query}
                )

            st.markdown("### Answer")
            st.write(result["answer"])

    except Exception as e:
        st.error(f"Error: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
