import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
import os

st.markdown("""
<style>
    /* Hides the default Streamlit header, footer, and hamburger menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom chat bubble styles */
    .st-emotion-cache-1c7y2kd { /* Assistant chat bubble */
        background-color: #262730; /* A darker gray for the assistant */
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

# --- UI Configuration ---
st.set_page_config(page_title="Sarah Bessadi | Digital CV", page_icon="🚀", layout="wide")

# --- Header Section ---
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

# --- Caching the LangChain Chain for performance ---
@st.cache_resource
def load_chain():
    """
    Loads the LangChain conversational retrieval chain.
    This version uses the powerful OpenAI GPT-3.5 model via API.
    """
    try:
        # Check for the OpenAI API key
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            st.error("OpenAI API Key not found. Please set the OPENAI_API_KEY secret for the app to work.")
            return None

        # 1. Load the data from the text file
        loader = TextLoader('./profile.txt', encoding='utf-8')
        documents = loader.load()

        # 2. Split the document into manageable chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # 3. Create embeddings (this part remains the same)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # 4. Create a FAISS vector store
        vectorstore = FAISS.from_documents(docs, embeddings)

        # 5. Set up the OpenAI Language Model (LLM)
        # This is the new, more reliable part.
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
            temperature=0.5 # Lower temperature for more factual answers
        )

        # 6. Create the Conversational Retrieval Chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=False
        )
        return chain
    except Exception as e:
        st.error(f"An error occurred while loading the AI model: {e}")
        return None

# --- Main Application Logic ---
chain = load_chain()

if chain:
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(
            {"role": "assistant", "content": "Hello! How can I help you learn more about Sarah's profile?"}
        )

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about Sarah..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_history = [(msg["role"], msg["content"]) for msg in st.session_state.messages if msg["role"] != 'assistant']
                result = chain({"question": prompt, "chat_history": chat_history})
                response = result['answer']
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.error("The chatbot could not be loaded. Please check the logs in the terminal for errors.")