import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# === Setup ===
# Set up the Streamlit page configuration
st.set_page_config(page_title="Portfolio Assistant", page_icon="🤖")
st.title("🤖 Portfolio AI Assistant")

# === Load and Split Document ===
# Function to load a PDF and split it into chunks
def load_and_split_document(file_path):
    """Loads a PDF document and splits it into manageable chunks for processing."""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        return chunks
    except Exception as e:
        # Handle cases where the file path is invalid or the file is missing
        st.error(f"Error loading document: {e}. Please check the file path.")
        st.stop()

# === Create Embeddings and Vectorstore ===
# Function to create a vector store from document chunks
def setup_vectorstore(chunks):
    """Creates a vector store using a HuggingFace embedding model."""
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cpu"}, # Force embeddings to run on CPU for deployment compatibility
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# === Create Conversation Chain ===
# Function to create a conversational chain using the vector store and LLM
def create_conversation_chain(vectorstore):
    """Initializes and returns a conversational retrieval chain."""
    # Use st.secrets to securely access the GROQ API key
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain

# === Main Logic ===
# Use os.path.join for a robust file path that works on both Windows and Linux
file_path = os.path.join("data", "Kushrajsinh_Zala_Resume_2025.pdf")

# Initialize session state for vector store, conversation chain, and chat history
if "vectorstore" not in st.session_state:
    # Check if the PDF file exists before trying to load it
    if not os.path.exists(file_path):
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it's in the 'data' folder in your GitHub repository.")
        st.stop()
    
    chunks = load_and_split_document(file_path)
    st.session_state.vectorstore = setup_vectorstore(chunks)

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = create_conversation_chain(st.session_state.vectorstore)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show previous messages from the chat history
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Handle user input
user_input = st.chat_input("Know Me Better......")

if user_input:
    # Append the user's message to the chat history and display it
    st.session_state.chat_history.append({'role': "user", "content": user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # Get the AI's response and display it
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            response = st.session_state.conversation_chain.invoke({'question': user_input})
            ai_response = response['answer']
            st.markdown(ai_response)
    
    # Append the AI's response to the chat history
    st.session_state.chat_history.append({'role': 'ai', "content": ai_response})

