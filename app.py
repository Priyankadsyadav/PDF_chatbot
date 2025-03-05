import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# Ensure API key is set
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY is missing! Please set the API key before running the application.")
    st.stop()
else:
    os.environ["GROQ_API_KEY"] = "API_key"  # Only set if it exists

# Set up Streamlit UI
st.title("PDF-based Chatbot")
st.sidebar.header("Upload your PDF")

uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_pdf_path = os.path.join("temp.pdf")
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load and extract text from the PDF
    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # Embed and store in FAISS vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()
    
    # Setup conversational chain with memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,output_key="answer")
    llm = ChatGroq(model="gemma2-9b-it") 
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, return_source_documents=True, verbose=True)
    
    # Chat UI
    st.write("### Ask a question based on the uploaded PDF")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_query" not in st.session_state:
        pass

    user_query = st.text_input("Your question:", key="user_query")
    if user_query:
        result = qa_chain.invoke(user_query)
        response = result['answer']
        st.session_state.chat_history.append((user_query, response))
        
    # Display chat history
    for query, answer in st.session_state.chat_history:
        st.write(f"**You:** {query}")
        st.write(f"**Bot:** {answer}")
    
    # Clean up temporary file
    os.remove(temp_pdf_path)
