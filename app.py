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

# 1. Premium Dark-Mode Oriented Wide Layout Config
st.set_page_config(page_title="AI PDF Assistant Pro", layout="wide")

# Advanced Premium Dark Mode CSS Injection to fix margins and color contrasts
st.markdown("""
    <style>
        /* Smooth text styling and grid normalization */
        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, sans-serif;
        }
        .block-container {
            padding-top: 1.5rem;
            max-width: 1200px;
        }
        
        /* Modern Title Styling with subtle violet-blue gradients */
        .premium-title {
            background: linear-gradient(90deg, #A78BFA, #60A5FA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.6rem;
            font-weight: 800;
            margin-bottom: 0.1rem;
        }
        
        /* Subtitle Thread Badge styling */
        .thread-badge {
            background-color: rgba(96, 165, 250, 0.12);
            color: #60A5FA;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(96, 165, 250, 0.25);
        }

        /* Card Container Design Pattern for the Upload Space */
        .control-card {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }

        /* Seamless Dark-Mode optimization for File Upload Boxes */
        div[data-testid="stFileUploader"] {
            border: 2px dashed #475569 !important;
            background-color: #0F172A !important;
            border-radius: 12px !important;
            padding: 12px !important;
        }

        /* Premium Sidebar Buttons Formatting */
        .stButton>button {
            border-radius: 10px !important;
            transition: all 0.2s ease-in-out !important;
            background-color: #4F46E5 !important;
            color: white !important;
            border: none !important;
            font-weight: 600 !important;
        }
        .stButton>button:hover {
            background-color: #4338CA !important;
            box-shadow: 0 0 15px rgba(79, 70, 229, 0.35) !important;
            transform: translateY(-1px) !important;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE HISTORY ARCHITECTURE ----------------- #
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {}  
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.session_state.all_chats[st.session_state.current_chat] = st.session_state.chat_history
# ---------------------------------------------------------------------- #

# Sidebar Navigation Architecture
# Sidebar Navigation Architecture (100% Native Streamlit Elements)
with st.sidebar:
    st.title("🚀 Workspace Pro")
    st.caption("Manage conversational context threads and engine rules dynamically.")
    
    # Action Container Box
    with st.container():
        col_new, col_clear = st.columns(2)
        with col_new:
            if st.button("➕ New Thread", use_container_width=True):
                new_id = f"Chat {len(st.session_state.all_chats) + 1}"
                st.session_state.all_chats[new_id] = []
                st.session_state.current_chat = new_id
                st.session_state.chat_history = []
                if "rag_chain" in st.session_state:
                    del st.session_state.rag_chain
                st.rerun()
        with col_clear:
            if st.button("🗑️ Reset All", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.all_chats[st.session_state.current_chat] = []
                st.rerun()
        
    st.divider()
    
    # Expandable Dialogues Tree for better cleanliness
    with st.expander("📜 Stored Conversations History", expanded=True):
        for chat_title in list(st.session_state.all_chats.keys()):
            if chat_title == st.session_state.current_chat:
                # Highlight active chat using native colorful status indicators
                st.success(f"🔥 Active: {chat_title}")
            else:
                if st.button(f"💬 Open {chat_title}", key=f"nav_{chat_title}", use_container_width=True):
                    st.session_state.current_chat = chat_title
                    st.session_state.chat_history = st.session_state.all_chats[chat_title]
                    st.rerun()

    st.divider()
    
    # Engine Status Indicators Box
    with st.container():
        st.subheader("⚡ System Diagnostics")
        
        # Native metric blocks to provide dashboard visual feedback loops
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(label="Active Logs", value=len(st.session_state.chat_history))
        with metric_col2:
            st.metric(label="Total Threads", value=len(st.session_state.all_chats))
            
        # Status light based on document index state
        if "rag_chain" in st.session_state:
            st.status("💡 Context Matrix Online", state="complete")
        else:
            st.status("💤 Standing By For Document...", state="running")

    st.divider()
    st.subheader("🔑 Engine Configuration")
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

    if not api_key:
        api_key = st.text_input("Enter Google Gemini API Key:", type="password")


if not api_key:
    st.warning("Please enter your Gemini API key in the sidebar.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# --- Main Board Layout Design Elements ---
st.markdown('<h1 class="premium-title">📄 AI PDF Assistant Pro</h1>', unsafe_allow_html=True)
st.markdown(f'<span class="thread-badge">● Active Thread: {st.session_state.current_chat}</span>', unsafe_allow_html=True)

# --- Integrated Control Center Module ---
st.markdown('<div class="control-card">', unsafe_allow_html=True)
st.markdown("### 📂 Control Center")
uploaded_file = st.file_uploader(
    "Upload your reference PDF file structure",
    type=["pdf"],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# --- Pipeline Vector Processing & Conversation Feed Area ---
if uploaded_file is not None:
    temp_path = uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    try:
        if "rag_chain" not in st.session_state:
            with st.status("🔮 Analyzing and indexing document vector spaces...", expanded=True) as status:
                st.write("Extracting PDF text structures...")
                loader = PyPDFLoader(temp_path)
                docs = loader.load()

                st.write("Splitting document blocks into logical chunk systems...")
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = splitter.split_documents(docs)

                st.write("Generating embedding vectors & building vector database indexes...")
                embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
                vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

                llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an expert PDF assistant.\n\nUse the provided context to answer questions.\n\nIf the answer is not found in the document, say 'The document does not contain that information.'\n\nContext:\n{context}\n"),
                    ("human", "{input}"),
                ])

                qa_chain = create_stuff_documents_chain(llm, prompt)
                st.session_state.rag_chain = create_retrieval_chain(retriever, qa_chain)
                status.update(label="✅ Knowledge Base Synchronized!", state="complete", expanded=False)

        # ------------------ PRE-RENDER MODERN CHAT INTERFACE ------------------
        # Creating a dedicated visual window container for messages
        with st.container():
            st.write("### 💬 Conversation Thread")
            
            # Print past question/answer logs cleanly into dedicated user/assistant message bubbles
            for chat in st.session_state.chat_history:
                with st.chat_message(chat["role"]):
                    st.markdown(chat["content"])

        # ------------------ PREMIUM FLOATING CHAT ENTRY ------------------
        # This replaces the static box with an elegant, anchored bottom interface
        if query := st.chat_input("Message your document assistant or ask clarifying details..."):
            
            # 1. Print and save user input immediately
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # 2. Open an aligned processing container bubble for the assistant's stream
            with st.chat_message("assistant"):
                with st.spinner("Analyzing context matrices..."):
                    result = st.session_state.rag_chain.invoke({"input": query})
                    response_text = result["answer"]
                    st.markdown(response_text)
            
            # 3. Complete context logging cycle and reload state seamlessly
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            st.session_state.all_chats[st.session_state.current_chat] = st.session_state.chat_history
            st.rerun()

    except Exception as e:
        st.error(f"Error: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
else:
    st.info("👋 Welcome! Please upload a PDF document inside the Control Center module above to activate the chat framework.")
