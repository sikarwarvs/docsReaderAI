import streamlit as st
import os
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from docx import Document

# Load environment variables
load_dotenv()

# Configure Google Generative AI
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from DOCX files
def get_docx_text(docx_docs):
    text = ""
    for docx in docx_docs:
        try:
            doc = Document(io.BytesIO(docx.read()))
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            st.error(f"Error processing DOCX file: {e}")
            return None
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save vector store
def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("No text chunks found. Exiting.")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    try:
        vector_store.save_local("faiss_index")
        st.success("FAISS index saved successfully.")
    except Exception as e:
        st.error(f"Error saving FAISS index: {e}")
        return None

    return vector_store

# Function to get conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the provided context, just say, "Answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n{context}\n
    Question:\n{question}\n

    Answer: 
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and provide response
def user_input(user_question, vector_store):
    docs = vector_store.similarity_search(user_question)
    
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}
    , return_only_outputs= True)

    st.write("Reply: ", response["output_text"])

# Main function
def main():
    st.set_page_config("Module1 chatting with docx files")
    st.header("Module1 Chat with docx using AI")

    user_question = st.text_input("Ask a question")

    # Check if user question is provided
    if user_question:
        # Load vector store from FAISS index
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        # Provide response based on user question and vector store
        user_input(user_question, vector_store)
   
    # Sidebar for uploading DOCX files
    with st.sidebar:
        st.title("Menu:")
        docx_docs = st.file_uploader("Upload Your DOCX Files and click the Submit Button", accept_multiple_files=True, type=['docx'])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if docx_docs:
                    # Extract text from uploaded DOCX files
                    raw_text = get_docx_text(docx_docs)
                    if raw_text:
                        # Split text into chunks
                        text_chunks = get_text_chunks(raw_text)
                        if text_chunks:
                            # Create and save vector store
                            get_vector_store(text_chunks)
                else:
                    st.error("Please upload at least one DOCX file.")

# Run the main function
if __name__== "__main__":
    main()
