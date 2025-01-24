import streamlit as st
import openai
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain_core.documents import Document

# Load OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]
DOCUMENTS_DIR = st.secrets["DOCUMENTS_DIR"]
SAVE_LOCAL_PATH = "faiss_index_nosplit"


# # Load the notion content located in the notion_content folder
# loader = NotionDirectoryLoader("notion_content")
# documents = loader.load()
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    TextLoader,
    Docx2txtLoader,
    DirectoryLoader,
)

from langchain_core.documents import Document

from pathlib import Path

# Initialize OpenAI embedding model
embeddings = OpenAIEmbeddings()
documents_dir = Path(DOCUMENTS_DIR)
docs = DirectoryLoader(documents_dir).load()
# docs = []
# for doc_path in documents_dir.glob("*.md"):
#     print(doc_path)
#     loader = TextLoader(str(doc_path), mode="elements")
#     documents = loader.load()
#     print(len(documents), documents[0])
#     docs.extend(documents)
#     # markdown_splitter = RecursiveCharacterTextSplitter(
#     #     separators=["#", "##", "###", "\n\n", "\n", "."],
#     #     # chunk_size=1500,
#     #     # chunk_overlap=100,
#     # )
#     # docs.extend(markdown_splitter.split_documents(documents))

db = FAISS.from_documents(docs, embeddings)
db.save_local(SAVE_LOCAL_PATH)
print("Local FAISS index has been successfully saved.")
