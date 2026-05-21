import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai

# ---------------- PDF TEXT ---------------- #

def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted

    return text


# ---------------- SIMPLE CHUNKING ---------------- #

def chunk_text(text, chunk_size=3000):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunks.append(text[i:i + chunk_size])

    return chunks


# ---------------- SIMPLE SEARCH ---------------- #

def find_relevant_chunks(chunks, question):

    question_words = question.lower().split()

    scored_chunks = []

    for chunk in chunks:

        score = 0

        chunk_lower = chunk.lower()

        for word in question_words:

            if word in chunk_lower:
                score += 1

        scored_chunks.append((score, chunk))

    scored_chunks.sort(reverse=True)

    top_chunks = [chunk for score, chunk in scored_chunks[:3]]

    return "\n".join(top_chunks)


# ---------------- GEMINI RESPONSE ---------------- #

def ask_gemini(api_key, context, question):

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {question}

    If answer is not found, say:
    "Answer not available in context."
    """

    response = model.generate_content(prompt)

    return response.text


# ---------------- MAIN ---------------- #

def main():

    st.set_page_config(page_title="PDF Chatbot")

    st.title("Chat with PDF")

    api_key = st.sidebar.text_input(
        "Enter Google API Key",
        type="password"
    )

    pdf_docs = st.sidebar.file_uploader(
        "Upload PDFs",
        accept_multiple_files=True
    )

    user_question = st.text_input(
        "Ask a question"
    )

    if st.button("Submit"):

        if not api_key:
            st.warning("Enter API key")
            return

        if not pdf_docs:
            st.warning("Upload PDFs")
            return

        if not user_question:
            st.warning("Enter question")
            return

        with st.spinner("Processing..."):

            raw_text = get_pdf_text(pdf_docs)

            chunks = chunk_text(raw_text)

            relevant_context = find_relevant_chunks(
                chunks,
                user_question
            )

            answer = ask_gemini(
                api_key,
                relevant_context,
                user_question
            )

            st.success("Done")

            st.markdown("## Answer")

            st.write(answer)


if __name__ == "__main__":
    main()
