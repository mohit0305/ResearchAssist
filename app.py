import os
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from litellm import completion
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools import ArxivQueryRun
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

os.environ["STREAMLIT_DISABLE_FILE_WATCHER"] = "true"
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
token = os.getenv("HUGGINGFAVE_TOKEN")

if token:
    from huggingface_hub import login
    login(token=token)

# Use HuggingFace-compatible wrapper for FAISS integration
text_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
arxiv_tool = ArxivQueryRun()


def extract_text_from_pdf(uploaded_files):
    all_text = ""
    for uploaded_file in uploaded_files:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            all_text += page.extract_text() or ""
    return all_text


def process_text_and_store(all_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(all_text)
    documents = [{"page_content": chunk, "metadata": {"source": "pdf", "chunk_id": i}} for i, chunk in enumerate(chunks)]

    vectorstore = FAISS.from_documents(
        documents=[d["page_content"] for d in documents],
        embedding=embedding
    )
    return vectorstore


def semantic_search(query, vectorstore, top_k=2):
    results = vectorstore.similarity_search(query, k=top_k)
    return results


def generate_response(query, context):
    prompt = f"Query: {query}\nContext: {context}\nAnswer:"
    response = completion(
        model="gemini/gemini-1.5-flash",
        messages=[{"content": prompt, "role": "user"}],
        api_key=gemini_api_key
    )
    return response['choices'][0]['message']['content']


def main():
    st.title("RAG-powered Research Paper Assistant")

    option = st.radio("Choose an option:", ("Upload PDFs", "Search arXiv"))

    if option == "Upload PDFs":
        uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
        if uploaded_files:
            st.write("Processing uploaded files...")
            all_text = extract_text_from_pdf(uploaded_files)
            vectorstore = process_text_and_store(all_text)
            st.success("PDF content processed and stored successfully!")

            query = st.text_input("Enter your query:")
            if st.button("Execute Query") and query:
                results = semantic_search(query, vectorstore)
                context = "\n".join([res.page_content for res in results])
                response = generate_response(query, context)
                st.subheader("Generated Response:")
                st.write(response)

    elif option == "Search arXiv":
        query = st.text_input("Enter your search query for arXiv:")
        if st.button("Search ArXiv") and query:
            arxiv_results = arxiv_tool.invoke(query)
            st.session_state["arxiv_results"] = arxiv_results
            st.subheader("Search Results:")
            st.write(arxiv_results)

            vectorstore = process_text_and_store(arxiv_results)
            st.session_state["vectorstore"] = vectorstore
            st.success("arXiv paper content processed and stored successfully!")

        if "arxiv_results" in st.session_state and "vectorstore" in st.session_state:
            query = st.text_input("Ask a question about the paper:")
            if st.button("Execute Query on Paper") and query:
                results = semantic_search(query, st.session_state["vectorstore"])
                context = "\n".join([res.page_content for res in results])
                response = generate_response(query, context)
                st.subheader("Generated Response:")
                st.write(response)


if __name__ == "__main__":
    main()
