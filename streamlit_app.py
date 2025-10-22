import streamlit as st
import os

# --- OLD (STABLE) LANGCHAIN IMPORTS ---
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Sarah Bessadi | Digital CV", page_icon="🚀", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Hides the default Streamlit header, footer, and hamburger menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom chat bubble styles */
    .st-emotion-cache-1c7y2kd { /* Assistant chat bubble */
        background-color: #262730; 
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #262730;
    }

    .st-emotion-cache-4oy321 { /* User chat bubble */
        background-color: transparent;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #BF40BF; /* Purple border for the user */
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
with st.container():
    st.title("🤖 Sarah Bessadi's AI Ambassador")
    st.subheader("Dual Master's Graduate in Engineering & International Business")
    st.info(
        "Welcome! I am an AI agent trained on Sarah's professional profile. "
        "Feel free to ask me anything about her skills, experience, or ambitions."
    )
    st.markdown(
        "[View LinkedIn Profile](https://www.linkedin.com/in/sarah-bessadi/) | "
        "**Email:** bessadisarah@gmail.com"
    )

# --- CACHED FUNCTION TO LOAD THE CHAIN ---
@st.cache_resource
def load_chain():
    """
    Loads and caches the full LangChain RAG pipeline.
    This function will only run once.
    """
    try:
        # 1. Get OpenAI API Key
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            st.error("OpenAI API Key not found. Please set it in your Streamlit Cloud secrets.")
            return None

        # 2. Load the data
        # Assumes 'profile.txt' is in the *same folder* as 'streamlit_app.py'
        loader = TextLoader('./profile.txt', encoding='utf-8')
        documents = loader.load()

        # 3. Split the document
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # 4. Create embeddings (using the old import path)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # 5. Create FAISS vector store
        vectorstore = FAISS.from_documents(docs, embeddings)

        # 6. Set up the LLM (using the old import path and old openai version)
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
            temperature=0.7 
        )
        
        # 7. Create memory
        # This will hold the chat history *inside* the chain object
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # 8. Create the old, simple chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory  # The chain will manage its own memory
        )
        
        return chain

    except Exception as e:
        st.error(f"An error occurred while loading the AI model: {e}")
        st.exception(e) 
        return None

# --- MAIN APPLICATION LOGIC ---
chain = load_chain()

if chain:
    # Initialize chat history *for display only*
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you learn more about Sarah's profile?"}
        ]

    # Display chat messages from display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about Sarah..."):
        
        # Add user message to display history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                
                # This is the old, simple way to call the chain.
                # The memory is handled internally by the 'chain' object.
                # We don't need to pass the history manually.
                result = chain({"question": prompt})
                
                response = result.get('answer', 'Sorry, I encountered an issue.')
                st.markdown(response)
        
        # Add AI response to display history
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.error(
        "The chatbot could not be loaded. "
        "Please check the 'Manage app' logs for errors "
        "and ensure your OPENAI_API_KEY secret is set."
    )
