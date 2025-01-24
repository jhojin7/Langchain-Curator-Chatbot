import streamlit as st
import openai
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]
DOCUMENTS_DIR = st.secrets["DOCUMENTS_DIR"]


# # Load the notion content located in the notion_content folder
# loader = NotionDirectoryLoader("notion_content")
# documents = loader.load()
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document

from pathlib import Path

# Initialize OpenAI embedding model
embeddings = OpenAIEmbeddings()

documents_dir = Path(DOCUMENTS_DIR)
docs = []
for doc_path in documents_dir.glob("*.md"):
    print(doc_path)
    loader = TextLoader(str(doc_path), mode="elements")
    documents = loader.load()
    print(len(documents), documents[0])

    markdown_splitter = RecursiveCharacterTextSplitter(
        separators=["#", "##", "###", "\n\n", "\n", "."],
        # chunk_size=1500,
        # chunk_overlap=100,
    )
    docs.extend(markdown_splitter.split_documents(documents))
    # Convert all chunks into vectors embeddings using OpenAI embedding model
    # Store all vectors in FAISS index and save locally to 'faiss_index'
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index_textloader")
print("Local FAISS index has been successfully saved.")
