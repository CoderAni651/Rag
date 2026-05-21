import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import asyncio
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.embeddings.base import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


# ---------------- FIX ASYNC ISSUE ---------------- #

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# ---------------- SIMPLE EMBEDDINGS ---------------- #

class SimpleEmbeddings(Embeddings):

    def embed_documents(self, texts):
        return [[float(len(text))] * 10 for text in texts]

    def embed_query(self, text):
        return [float(len(text))] * 10


# ---------------- PDF TEXT EXTRACTION ---------------- #

def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted

    return text


# ---------------- TEXT CHUNKING ---------------- #

def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    return chunks


# ---------------- VECTOR STORE ---------------- #

def get_vector_store(text_chunks):

    embeddings = SimpleEmbeddings()

    vector_store = DocArrayInMemorySearch.from_texts(
        text_chunks,
        embedding=embeddings
    )

    return vector_store


# ---------------- QA CHAIN ---------------- #

def get_conversational_chain(api_key):

    prompt_template = """
    Answer the question as detailed as possible from the provided context.

    If the answer is not in the provided context,
    say:
    "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=api_key
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt
    )

    return chain


# ---------------- USER QUESTION ---------------- #

def user_input(user_question, api_key, pdf_docs):

    if not api_key:
        st.warning("Please enter Google API Key")
        return

    if not pdf_docs:
        st.warning("Please upload PDF files")
        return

    # Extract text
    raw_text = get_pdf_text(pdf_docs)

    # Chunking
    text_chunks = get_text_chunks(raw_text)

    # Vector DB
    vector_store = get_vector_store(text_chunks)

    # Search docs
    docs = vector_store.similarity_search(user_question)

    # QA chain
    chain = get_conversational_chain(api_key)

    response = chain(
        {
            "input_documents": docs,
            "question": user_question
        },
        return_only_outputs=True
    )

    answer = response["output_text"]

    # Store history
    st.session_state.conversation_history.append(
        {
            "Question": user_question,
            "Answer": answer,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )

    # Display
    st.markdown("## Question")
    st.write(user_question)

    st.markdown("## Answer")
    st.write(answer)

    st.success("Done")


# ---------------- MAIN APP ---------------- #

def main():

    st.set_page_config(
        page_title="RAG PDF Chatbot",
        page_icon="📚"
    )

    st.header("Chat with PDF using Gemini 📚")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Sidebar
    with st.sidebar:

        st.title("Menu")

        api_key = st.text_input(
            "Enter Google API Key",
            type="password"
        )

        st.markdown(
            "Get API key from: https://ai.google.dev/"
        )

        pdf_docs = st.file_uploader(
            "Upload PDF Files",
            accept_multiple_files=True
        )

        if st.button("Clear History"):
            st.session_state.conversation_history = []
            st.success("History Cleared")

    # User input
    user_question = st.text_input(
        "Ask a Question from the PDF Files"
    )

    if user_question:

        user_input(
            user_question,
            api_key,
            pdf_docs
        )

    # Show conversation history
    if st.session_state.conversation_history:

        st.markdown("---")
        st.markdown("## Conversation History")

        for chat in reversed(st.session_state.conversation_history):

            st.markdown(f"### Question")
            st.write(chat["Question"])

            st.markdown(f"### Answer")
            st.write(chat["Answer"])

            st.caption(chat["Timestamp"])

        # Download CSV
        df = pd.DataFrame(st.session_state.conversation_history)

        csv = df.to_csv(index=False)

        b64 = base64.b64encode(csv.encode()).decode()

        href = f'''
        <a href="data:file/csv;base64,{b64}"
        download="conversation_history.csv">
        Download Conversation CSV
        </a>
        '''

        st.sidebar.markdown(
            href,
            unsafe_allow_html=True
        )


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    main()


# Rag
