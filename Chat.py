import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_mistralai import ChatMistralAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_local_storage import LocalStorage
from dotenv import load_dotenv
import tempfile
import hashlib
import os
import logging
import time

# Load environment variables
load_dotenv()

# Initialize Local Storage
local_storage = LocalStorage()

def safe_delete_chat_history():
    try:
        local_storage.deleteItem("chat_history")
    except KeyError:
        logger.info("SESSION: No chat_history key found in local storage; nothing to delete.")
    except Exception as e:
        logger.warning(f"SESSION: Failed to delete chat_history from local storage: {e}")

# ─── LOGGING SETUP ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("PDFChatbot")
logger.info("=" * 60)
logger.info("APP STARTED — Streamlit script execution begin")
logger.info("=" * 60)

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Chatbot", layout="wide", page_icon="🤖")

# Hide Streamlit's default top-right menu and footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

logger.info("UI: Page config set, title rendered")

# ─── SESSION STATE ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    saved_history = local_storage.getItem("chat_history")
    
    if saved_history and isinstance(saved_history, list) and len(saved_history) > 0:
        st.session_state.messages = saved_history
        logger.info(f"SESSION: Loaded {len(saved_history)} messages from local storage")
    else:
        st.session_state.messages = []
        logger.info("SESSION: Starting with empty messages")
else:
    logger.info(f"SESSION: Using existing {len(st.session_state.messages)} messages from session state")

if "chain" not in st.session_state:
    st.session_state.chain = None

if "current_provider" not in st.session_state:
    st.session_state.current_provider = None

# ─── SIDEBAR: APP BRANDING ───────────────────────────────────────────────────
st.sidebar.markdown("### 🤖 PDF Chatbot")
st.sidebar.caption("Powered by Mistral + LangChain")

# New Chat button — prominent at the top
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chain = None
        st.session_state.current_provider = None
        safe_delete_chat_history()
        logger.info("ACTION: New chat started, local storage cleared")
        st.rerun()
with col2:
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chain = None
        st.session_state.current_provider = None
        safe_delete_chat_history()
        logger.info("ACTION: All cleared, local storage cleared")
        st.rerun()

# ─── SIDEBAR: PDF UPLOAD ────────────────────────────────────────────────────
st.sidebar.header("📄 Upload PDF")
uploaded_files = st.sidebar.file_uploader(
    "Upload your PDF files", type=["pdf"], accept_multiple_files=True
)
logger.info(f"UI: Sidebar rendered | Files uploaded: {len(uploaded_files) if uploaded_files else 0}")

st.sidebar.divider()

# ─── SIDEBAR: MODE TOGGLES ──────────────────────────────────────────────────
st.sidebar.header("⚙️ Chat Settings")

# Toggle 1: RAG vs Direct Chat
chat_mode = st.sidebar.radio(
    "💡 Chat Mode",
    ["🔍 RAG (PDF Context)", "💬 Direct Chat"],
    index=0,
    help="RAG uses your uploaded PDF to answer questions. Direct Chat talks to the AI without any PDF context."
)
is_rag_mode = chat_mode == "🔍 RAG (PDF Context)"
logger.info(f"CONFIG: Chat mode = {'RAG' if is_rag_mode else 'Direct Chat'}")

# Toggle 2: Local Ollama vs Cloud Mistral API
mistral_api_key = os.getenv("MISTRAL_API_KEY", "")
has_api_key = len(mistral_api_key) > 5

llm_provider = st.sidebar.radio(
    "🌐 LLM Provider",
    ["🖥️ Local Ollama (Mistral 7B)", "☁️ Mistral Cloud API (Fast)"],
    index=0,
    help="Local runs on your machine (slower). Cloud uses Mistral's free API (much faster, needs internet)."
)
use_cloud = llm_provider == "☁️ Mistral Cloud API (Fast)"

if use_cloud and not has_api_key:
    st.sidebar.error("⚠️ No MISTRAL_API_KEY found in .env file!")
    logger.warning("CONFIG: Cloud API selected but no API key found")

logger.info(f"CONFIG: LLM provider = {'Cloud Mistral API' if use_cloud else 'Local Ollama'}")

st.sidebar.divider()


# ─── HELPER: GET LLM ────────────────────────────────────────────────────────
def get_llm(use_cloud_api):
    """Returns the appropriate LLM based on the selected provider."""
    if use_cloud_api and has_api_key:
        logger.info("  LLM: Initializing Mistral Cloud API (model=open-mistral-7b)")
        return ChatMistralAI(
            model="open-mistral-7b",
            api_key=mistral_api_key,
            temperature=0.3
        )
    else:
        logger.info("  LLM: Initializing Local Ollama (model=mistral:7b-instruct-q2_K)")
        return OllamaLLM(model="mistral:7b-instruct-q2_K")

# ─── HELPER: FILE HASHING ───────────────────────────────────────────────────
def get_file_hash(files):
    """Generate a hash based on the names & sizes of all uploaded files."""
    hash_str = "".join([f"{f.name}-{f.size}" for f in files])
    return hashlib.md5(hash_str.encode()).hexdigest()

# ─── FAISS INDEX PATH ────────────────────────────────────────────────────────
FAISS_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")
logger.info(f"CONFIG: FAISS index path = {FAISS_INDEX_PATH}")

# ─── DETECT PROVIDER CHANGE ─────────────────────────────────────────────────
provider_key = "cloud" if use_cloud else "local"
if st.session_state.current_provider and st.session_state.current_provider != provider_key:
    if st.session_state.chain is not None:
        st.session_state.chain = None
        logger.info(f"PROVIDER CHANGE: {st.session_state.current_provider} → {provider_key}, rebuilding chain")

# ─── FILE HASH CHECK (prevents unnecessary FAISS rebuilds) ──────────────────
if uploaded_files:
    current_hash = get_file_hash(uploaded_files)
    if "file_hash" not in st.session_state or st.session_state.file_hash != current_hash:
        st.session_state.chain = None
        st.session_state.file_hash = current_hash
        logger.info(f"FILE CHANGE DETECTED: New hash {current_hash}. Resetting chain.")

# ─── PDF PROCESSING PIPELINE ────────────────────────────────────────────────
if uploaded_files and st.session_state.chain is None:

    logger.info("=" * 60)
    logger.info("RAG PIPELINE: Starting PDF processing")
    logger.info("=" * 60)
    pipeline_start = time.time()

    with st.spinner("🔄 Processing your PDF(s)... Please wait."):

        # Step 1: Load PDFs
        status_text = st.empty()
        status_text.info("📄 Step 1/5: Loading PDF documents...")
        step_start = time.time()
        documents = []

        for uploaded_file in uploaded_files:
            logger.info(f"  LOAD: Reading file '{uploaded_file.name}' ({uploaded_file.size} bytes)")

            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                loader = PyPDFLoader(tmp_path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                logger.info(f"  LOAD: Extracted {len(loaded_docs)} pages from '{uploaded_file.name}'")

                os.remove(tmp_path)
            except Exception as e:
                logger.error(f"  LOAD ERROR: Failed to load '{uploaded_file.name}': {e}")
                st.error(f"❌ Failed to load '{uploaded_file.name}': {e}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        logger.info(f"  LOAD: Total pages loaded = {len(documents)} | Time: {time.time() - step_start:.2f}s")

        # Step 2: Split text into chunks
        status_text.info(f"✂️ Step 2/5: Splitting {len(documents)} pages into chunks...")
        step_start = time.time()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)
        logger.info(f"  SPLIT: Created {len(docs)} chunks (chunk_size=1000, overlap=100) | Time: {time.time() - step_start:.2f}s")

        # Step 3: Generate embeddings
        status_text.info(f"🧠 Step 3/5: Generating embeddings for {len(docs)} chunks...")
        step_start = time.time()
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info(f"  EMBED: HuggingFace model loaded (all-MiniLM-L6-v2) | Time: {time.time() - step_start:.2f}s")

        # Step 4: Create and save FAISS vector store
        status_text.info("💾 Step 4/5: Storing vectors in FAISS database...")
        step_start = time.time()
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        index_size = sum(os.path.getsize(os.path.join(FAISS_INDEX_PATH, f)) for f in os.listdir(FAISS_INDEX_PATH))
        logger.info(f"  FAISS: Vector store created & saved ({index_size} bytes on disk) | Time: {time.time() - step_start:.2f}s")

        # Step 5: Build the conversational chain
        status_text.info("🔗 Step 5/5: Connecting to Mistral model...")
        step_start = time.time()
        llm = get_llm(use_cloud)

        # Dynamic memory size for RAG memory buffer
        auto_k = 5 if use_cloud else 2

        memory = ConversationBufferWindowMemory(
            k=auto_k,
            memory_key="chat_history",
            return_messages=True
        )
        logger.info(f"  MEMORY: ConversationBufferWindowMemory initialized (k={auto_k})")

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
            memory=memory
        )
        logger.info(f"  CHAIN: Built successfully | Time: {time.time() - step_start:.2f}s")

        st.session_state.chain = chain
        st.session_state.current_provider = provider_key
        status_text.empty()

    total_time = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info(f"RAG PIPELINE: COMPLETE | Total time: {total_time:.2f}s")
    logger.info("=" * 60)

    st.sidebar.success("✅ PDF processed successfully!")
    st.sidebar.info(f"📁 FAISS vectors saved to:\n`{FAISS_INDEX_PATH}`")


# ─── CHAT HEADER ─────────────────────────────────────────────────────────────
mode_label = "🔍 RAG" if is_rag_mode else "💬 Direct"
provider_label = "☁️ Cloud" if use_cloud else "🖥️ Local"

header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown("### 🤖 Chat")
with header_col2:
    st.markdown(f"<div style='text-align:right; padding-top:8px;'><code>{mode_label} • {provider_label}</code></div>", unsafe_allow_html=True)

# ─── WELCOME SCREEN (when chat is empty) ────────────────────────────────────
if len(st.session_state.messages) == 0:
    st.markdown("---")
    st.markdown("#### 👋 Welcome! How can I help you today?")
    st.markdown("")

    wcol1, wcol2, wcol3 = st.columns(3)
    with wcol1:
        st.markdown("##### 📄 RAG Mode")
        st.markdown("Upload a PDF in the sidebar, then ask questions about its content.")
    with wcol2:
        st.markdown("##### 💬 Direct Chat")
        st.markdown("Switch to Direct Chat mode to talk to Mistral without any PDF.")
    with wcol3:
        st.markdown("##### ☁️ Cloud API")
        st.markdown("Switch to Cloud API for **instant responses** during your project review!")

    st.markdown("---")

# ─── CHAT UI ─────────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ─── CHAT INPUT ──────────────────────────────────────────────────────────────
input_placeholder = "Ask about your PDF..." if is_rag_mode else "Chat with Mistral AI..."
if prompt := st.chat_input(input_placeholder):

    logger.info(f"USER QUERY: '{prompt}' | Mode: {'RAG' if is_rag_mode else 'Direct'} | Provider: {'Cloud' if use_cloud else 'Local'}")
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    # ─── RAG MODE: Use the PDF retrieval chain ───────────────────────────
    if is_rag_mode:
        if st.session_state.chain:
            logger.info("LLM: Sending query via RAG chain...")
            query_start = time.time()

            with st.spinner("🔍 Searching PDF & generating answer..."):
                try:
                    # ✅ MUST pass chat_history explicitly for langchain_classic
                    response = st.session_state.chain.invoke({
                        "question": prompt + " (Answer in English only)",
                        "chat_history": []
                    })
                    answer = response["answer"]

                    query_time = time.time() - query_start
                    logger.info(f"LLM: Response received | Time: {query_time:.2f}s")
                    logger.info(f"LLM: Answer preview: '{answer[:100]}...'")
                except Exception as e:
                    logger.error(f"LLM ERROR: API call failed -> {str(e)}")
                    answer = f"⚠️ **API Error:** {str(e)}. Please wait a few seconds and try again, or switch to **Local Ollama**."
        else:
            answer = "⚠️ Please upload a PDF first to use RAG mode!"
            logger.warning("NO CHAIN: RAG mode but no PDF uploaded")

    # ─── DIRECT CHAT MODE: Talk to the LLM directly ─────────────────────
    else:
        logger.info("LLM: Sending query directly (no RAG)...")
        query_start = time.time()

        with st.spinner("💬 Generating response..."):
            try:
                llm = get_llm(use_cloud)
                auto_k = 5 if use_cloud else 2

                # Get the last `k` pairs of history (excluding the prompt we just appended)
                past_messages = st.session_state.messages[:-1]
                recent_history = past_messages[-(auto_k * 2):] if len(past_messages) > 0 else []

                if use_cloud and has_api_key:
                    # Cloud Mistral uses LangChain Messages
                    chat_history = []
                    for msg in recent_history:
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            chat_history.append(AIMessage(content=msg["content"]))

                    # Add current prompt
                    chat_history.append(HumanMessage(content=prompt))

                    logger.info(f"LLM: Attached {len(chat_history)-1} previous messages as context (k={auto_k})")
                    result = llm.invoke(chat_history)
                    answer = result.content
                else:
                    # Local Ollama uses a formatted string prompt
                    history_str = ""
                    if recent_history:
                        history_str = "Conversation History:\n"
                        for msg in recent_history:
                            role = "User" if msg["role"] == "user" else "AI"
                            history_str += f"{role}: {msg['content']}\n"
                        history_str += "\nCurrent Question:\n"

                    full_prompt = history_str + prompt
                    logger.info(f"LLM: Attached {len(recent_history)} previous messages as context (k={auto_k})")
                    answer = llm.invoke(full_prompt)

                query_time = time.time() - query_start
                logger.info(f"LLM: Direct response received | Time: {query_time:.2f}s")
                logger.info(f"LLM: Answer preview: '{str(answer)[:100]}...'")
            except Exception as e:
                logger.error(f"LLM ERROR: API call failed -> {str(e)}")
                answer = f"⚠️ **API Error:** {str(e)}. Please wait a few seconds and try again, or switch to **Local Ollama**."

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    # Save the updated messages to local storage
    local_storage.setItem("chat_history", st.session_state.messages)
    logger.info("SESSION: Chat history updated & saved to local storage")

logger.info("-" * 60)
logger.info("APP: Script execution complete (waiting for next interaction)")
logger.info("-" * 60)