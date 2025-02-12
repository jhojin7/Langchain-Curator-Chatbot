import openai
import streamlit as st
from pathlib import Path
from langchain_community import retrievers
from langchain_community.retrievers import TFIDFRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.document_loaders import BaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader
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
import pickle


openai.api_key = st.secrets["OPENAI_API_KEY"]
LOADER = TextLoader
TEXT_SPLITTER = RecursiveCharacterTextSplitter


metadata_colnames = "가게명,카테고리,주소,이미지URL1,이미지URL2,이미지URL3,별점,방문자리뷰수,블로그리뷰수,영업시간,전화번호,편의시설".split(
    ","
)
colnames = "가게명,카테고리,주소,메뉴1,메뉴1_가격,메뉴2,메뉴2_가격,메뉴3,메뉴3_가격,메뉴4,메뉴4_가격,방문자리뷰1,방문자리뷰2,방문자리뷰3,블로그리뷰1,블로그리뷰2,블로그리뷰3".split(
    ","
)


def build_retriever(documents: list[str], retriever, k=2):
    built_retriever = retriever.from_documents(documents)
    built_retriever.k = k
    return built_retriever


def build_documents(loader: BaseLoader, path: Path, text_splitter=None):
    documents = loader.load_and_split(text_splitter=text_splitter)
    return documents


def create_retriever(
    csv_path=None,
    top_k=3,
    save_path: Path = None,
):
    embeddings = OpenAIEmbeddings()
    bm25_pkl_path = Path("cache", "bm25_retriever.pkl")

    if not save_path:
        save_path = Path("faiss_cache_dir", csv_path.name)
    if not save_path.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    with st.spinner("Loading documents from CSV file..."):
        loader = CSVLoader(
            csv_path,
            encoding="utf-8-sig",
            content_columns=colnames,
            # metadata_columns=metadata_colnames
        )
        documents = build_documents(
            loader,
            csv_path,
            text_splitter=TEXT_SPLITTER(separators=["\n"], keep_separator=True),
        )
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
        if bm25_pkl_path.exists():
            bm25_retriever = pickle.load(open(bm25_pkl_path, "rb"))
            print("Loaded BM25Retriever from cache.")
        else:
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = 2
            pickle.dump(bm25_retriever, open(bm25_pkl_path, "wb"))
            print("Created BM25Retriever.")

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
