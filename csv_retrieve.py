import openai
import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import KonlpyTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import time

LOADER = TextLoader
TEXT_SPLITTER = RecursiveCharacterTextSplitter


def create_retriever(
    csv_path,
    top_k=3,
    save_path: Path = None,
):
    embeddings = OpenAIEmbeddings()
    if not save_path:
        save_path = Path("faiss_cache_dir", csv_path.name)
    if not save_path.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    with st.spinner("Loading documents from CSV file..."):
        loader = CSVLoader(csv_path, encoding="utf-8-sig")
        documents = loader.load_and_split(text_splitter=TEXT_SPLITTER())
        print(f"Loaded {len(documents)} documents from CSV file.")
        if not documents:
            raise ValueError("No documents loaded from CSV file.")

    with st.spinner("Creating FAISS vectorstore..."):
        try:
            faiss_vectorstore = FAISS.load_local(save_path, embeddings)
        except Exception as e:
            faiss_vectorstore = FAISS.from_documents(documents, embeddings)
            faiss_vectorstore.save_local(save_path)
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})
        print("Created FAISS vectorstore.")

    with st.spinner("Creating BM25Retriever vectorstore..."):
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 2

    with st.spinner("Creating Ensamble Retriever..."):
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )
        print("Created Ensemble Retriever.")

    st.success("Retrievers created successfully.")
    return ensemble_retriever


CHAT_PROMPT_TEMPLATE = """Context: {context}
Question: {question}
Answer:
"""


# Helper function to format documents
def format_docs(docs):
    print("===DEBUG===")
    for doc in docs:
        print(doc)
    print("===/DEBUG===")

    return "\n\n".join(doc.page_content for doc in docs)


def rag_from_csv(
    retriever,
    chat_prompt_template=CHAT_PROMPT_TEMPLATE,
    model_name="gpt-3.5-turbo",
    system_prompt="",
    temp=None,
    api_key=None,
):
    # Prompt template
    prompt = ChatPromptTemplate.from_template(chat_prompt_template)
    llm = ChatOpenAI(model=model_name, temperature=temp, api_key=api_key)
    print(llm)
    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    rag_chain.verbose = True
    print("RAG chain created.")
    return rag_chain
