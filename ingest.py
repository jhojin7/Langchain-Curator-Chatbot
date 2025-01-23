import streamlit as st
import openai
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# # Load the notion content located in the notion_content folder
# loader = NotionDirectoryLoader("notion_content")
# documents = loader.load()
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document


loader = UnstructuredMarkdownLoader(
    "/Users/hojinjang/Downloads/CUBO_DB/맛집 정보 분석 - 쌤쌤쌤.md",
    mode="elements",
)
documents = loader.load()
print(len(documents), documents[0])

# Split Notion content into smaller chunks
markdown_splitter = RecursiveCharacterTextSplitter(
    separators=["#", "##", "###", "\n\n", "\n", "."], chunk_size=1500, chunk_overlap=100
)
docs = markdown_splitter.split_documents(documents)

# Initialize OpenAI embedding model
embeddings = OpenAIEmbeddings()

# Convert all chunks into vectors embeddings using OpenAI embedding model
# Store all vectors in FAISS index and save locally to 'faiss_index'
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")

print("Local FAISS index has been successfully saved.")
