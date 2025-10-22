import streamlit as st
import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI  # Moved import to top
from langchain.text_splitter import CharacterTextSplitter

# --- UI Configuration ---
# This should be the *first* Streamlit command
st.set_page_config(page_title="Sarah Bessadi | Digital CV", page_icon="🚀", layout="wide")

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
        padding: 1rem; /* Corrected padding property */
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

        # 3. Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # 4. Create a FAISS vector store
        vectorstore = FAISS.from_documents(docs, embeddings)

        # 5. Set up the OpenAI Language Model (LLM)
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
            temperature=0.7 
        )

        # --- This is the new, correct chain logic ---

        # 6. Create the history-aware retriever
        retriever = vectorstore.as_retriever()
        
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. DO NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # 7. Create the question-answering chain
        qa_system_prompt = (
            "You are Sarah Bessadi's AI Ambassador. Your personality is professional, "
            "yet charismatic and a little fun. Your mission is to share Sarah's "
            "professional story in a captivating way. Be direct, but feel free to "
            "use a relevant emoji here and there. Never make up information."
            "\n\n"
            "Use the following pieces of retrieved context to answer the question:"
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # 8. Create the final retrieval chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # 9. Return the *correct* chain variable
        return rag_chain

    except Exception as e:
        st.error(f"An error occurred while loading the AI model: {e}")
        return None

# --- Main Application Logic ---
chain = load_chain()

if chain:
    # Initialize session state for *LangChain message objects*
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! How can I help you learn more about Sarah's profile?")
        ]

    # Display chat messages from history
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    # Accept user input
    if prompt := st.chat_input("Ask a question about Sarah..."):
        # Add user message to history
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Call the chain using the new 'invoke' method and correct history format
                result = chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                response = result['answer']
                st.markdown(response)
        
        # Add AI response to history
        st.session_state.chat_history.append(AIMessage(content=response))
else:
    st.error("The chatbot could not be loaded. Please check the app's 'Manage app' logs for errors.")
