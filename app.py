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

# Set page configuration with a modern title
st.set_page_config(page_title="AI PDF Assistant Pro", layout="wide")

# Custom CSS styling for premium layout spacing and sleek card elements
st.markdown("""
    <style>
        /* Smooth borders and transitions for interactive user components */
        .stButton>button {
            border-radius: 8px !important;
            transition: all 0.2s ease-in-out !important;
            background-color: #4F46E5 !important;
            color: white !important;
            border: none !important;
        }
        .stButton>button:hover {
            background-color: #4338CA !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2) !important;
        }
        /* Custom card wrapper block styling */
        .workspace-card {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Main Hero Header Title Block
st.markdown('# 📄 AI PDF Assistant Pro')
st.markdown('<p style="color: #6B7280; font-size: 1.05rem; margin-top: -12px;">An elegant, high-fidelity RAG tool optimized for rapid document navigation.</p>', unsafe_allow_html=True)
st.divider()

# Initialize dynamic stream storage arrays for clean thread handling
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

# Split UI architecture into a professional horizontal grid dashboard layout
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="workspace-card">', unsafe_allow_html=True)
    st.subheader("📁 Control Center")
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Loads context directly into local memory graph blocks."
    )
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:

    temp_path = uploaded_file.name

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    try:
        # Cache pipeline objects inside session states to prevent computation loops
        if "rag_chain" not in st.session_state:
            with col1:
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

                    st.session_state.rag_chain = create_retrieval_chain(
                        retriever,
                        qa_chain
                    )

            with col1:
                st.success("PDF indexed successfully!")

        # Dynamic chat dialogue space rendered in column 2
        with col2:
            st.subheader("💬 Active Conversation")
            
            # Print past question/answer snapshots into polished conversation boxes
            for chat in st.session_state.chat_history:
                with st.chat_message(chat["role"]):
                    st.markdown(chat["content"])

            # Native, smooth floating input block container anchored gracefully at the base
            if query := st.chat_input("Ask a question about the PDF:"):
                
                # Append user prompt immediately to active thread memory log arrays
                st.session_state.chat_history.append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)

                # Process retrieval chains inside standard engine loaders
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        result = st.session_state.rag_chain.invoke(
                            {"input": query}
                        )
                        response_text = result["answer"]
                        st.markdown(response_text)
                
                # Commit AI answer to active thread memory logs
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})

    except Exception as e:
        st.error(f"Error: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
