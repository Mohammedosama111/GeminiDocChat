import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import tempfile

def get_text_from_sources(sources, source_type):
    """
    Extract text from a list of uploaded files or a URL.
    """
    text = ""
    if source_type == 'file':
        for source in sources:
            # Create a temporary file to handle in-memory files
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{source.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(source.getvalue())
                tmp_file_path = tmp_file.name

            if source.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
                pages = loader.load_and_split()
                text += "\n".join([page.page_content for page in pages])
            elif source.name.endswith('.docx'):
                loader = Docx2txtLoader(tmp_file_path)
                pages = loader.load()
                text += "\n".join([page.page_content for page in pages])
            
            os.remove(tmp_file_path) # Clean up the temporary file

    elif source_type == 'url' and sources:
        # WebBaseLoader expects a list of URLs
        loader = WebBaseLoader([sources])
        pages = loader.load()
        text += "\n".join([page.page_content for page in pages])
        
    return text

def get_text_chunks(text):
    """
    Split text into manageable chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """
    Create a FAISS vector store from text chunks.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """
    Create a conversational retrieval chain.
    """
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def main():
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    st.set_page_config(page_title="Chat with your Documents", page_icon=":books:")

    st.header("Chat with your Documents :books:")
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.subheader("Your sources")
        docs = st.file_uploader(
            "Upload your PDFs or DOCX files here",
            accept_multiple_files=True,
            type=['pdf', 'docx']
        )
        url = st.text_input("Or paste a URL here")

        if st.button("Process"):
            # Check if API key is available
            if not os.getenv("GOOGLE_API_KEY"):
                st.error("Gemini API key is not set. Please create a .env file with your key as GOOGLE_API_KEY.")
                return

            with st.spinner("Processing..."):
                raw_text = ""
                if docs:
                    raw_text = get_text_from_sources(docs, 'file')
                elif url:
                    raw_text = get_text_from_sources(url, 'url')

                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.chat_history = [] # Reset chat history
                    st.success("Processing complete! You can now ask questions.")
                else:
                    st.warning("Please upload a document or provide a URL.")

    # Display chat messages from history
    for message in st.session_state.chat_history:
        with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
            st.write(message.content)

    # Accept user input
    if user_question := st.chat_input("Ask a question about your documents:"):
        if st.session_state.conversation:
            # Add user message to chat history
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            # Display user message
            with st.chat_message("user"):
                st.write(user_question)

            # Get assistant response
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({'question': user_question})
                ai_message = response['chat_history'][-1]
                st.session_state.chat_history.append(ai_message)
                # Display assistant response
                with st.chat_message("assistant"):
                    st.write(ai_message.content)
        else:
            st.warning("Please process a document or URL first.")

if __name__ == '__main__':
    main() 