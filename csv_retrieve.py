import openai
import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import KonlpyTextSplitter
from pathlib import Path

TEXT_SPLITTER = KonlpyTextSplitter


def create_retriever(
    csv_path,
    top_k=5,
    save_path: Path = None,
):
    embeddings = OpenAIEmbeddings()
    if not save_path:
        save_path = Path("faiss_cache_dir", csv_path.name)
    if not save_path.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        vectorstore = FAISS.load_local(
            save_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        print(e)

        with st.spinner("Loading documents from CSV file..."):
            loader = CSVLoader(csv_path, encoding="utf-8-sig")
            # documents = loader.load()
            documents = loader.load_and_split(
                text_splitter=TEXT_SPLITTER(
                    # chunk_size=500,
                    # chunk_overlap=100,
                )
            )
            print("Loaded documents from CSV file.")
        with st.spinner("Creating FAISS vectorstore..."):
            vectorstore = FAISS.from_documents(documents, embeddings)
            vectorstore.save_local(save_path)
            print("Created FAISS vectorstore.")
    print("Loaded FAISS vectorstore.")
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever


CHAT_PROMPT_TEMPLATE = """Context: {context}
Question: {question}
Answer:
"""


# Helper function to format documents
def format_docs(docs):
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
    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
