import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os

st.set_page_config(page_title="Smart PDF/CSV QA with Ollama", layout="centered")

st.title("📄 Smart PDF/CSV QA with Ollama (Local & Free)")

# Choose model
model = st.text_input("🧠 Choose Ollama Model", value="llama3")

# Upload file
uploaded_file = st.file_uploader("📤 Upload a PDF or CSV file", type=["pdf", "csv"])

# Ask question
query = st.text_input("❓ Ask a question based on the file")

# Setup Ollama
def ask_ollama(prompt, model_name):
    llm = Ollama(model=model_name)
    return llm.invoke(prompt)

# Handle CSV
def handle_csv(file, query, model):
    try:
        df = pd.read_csv(file)
        sample_rows = df.head(10).to_csv(index=False)
        prompt = (
            f"You're a data expert. Here is a CSV sample:\n\n{sample_rows}\n\n"
            f"Based on this, answer this question: {query}\n"
            f"Give only the answer, no code."
        )
        return ask_ollama(prompt, model)
    except Exception as e:
        return f"❌ Execution error: {e}"

# Handle PDF
def handle_pdf(file, query, model):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
        texts = text_splitter.split_documents(pages)
        sample_text = texts[0].page_content[:1500]

        prompt = (
            f"Here is an excerpt from a PDF:\n\n{sample_text}\n\n"
            f"Answer this question based on the document: {query}\n"
            f"Give only the answer, no code."
        )
        return ask_ollama(prompt, model)
    except Exception as e:
        return f"❌ Execution error: {e}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# Process on submit
if uploaded_file and query:
    with st.spinner("🧠 Thinking..."):
        if uploaded_file.name.endswith(".csv"):
            response = handle_csv(uploaded_file, query, model)
        elif uploaded_file.name.endswith(".pdf"):
            response = handle_pdf(uploaded_file, query, model)
        else:
            response = "Unsupported file type."

        st.markdown("### ✅ Answer")
        st.success(response)
