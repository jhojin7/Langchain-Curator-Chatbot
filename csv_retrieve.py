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

try:
    # 1. Load CSV
    file_path = "./data/네이버맛집리스트_20250124.0150.csv"
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()

    # 2. Create a retriever (using similarity search)
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("faiss_csv")
except Exception as e:
    print(e)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        "faiss_csv", embeddings, allow_dangerous_deserialization=True
    )
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Create the RAG chain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)


from pathlib import Path


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
        loader = CSVLoader(csv_path)
        documents = loader.load()
        print("Loaded documents from CSV file.")
        vectorstore = FAISS.from_documents(documents, embeddings)
        print("Created FAISS vectorstore.")
        vectorstore.save_local(save_path)
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
    temp=0.2,
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


# 4. Interactive chat loop
def csv_chatbot():
    chain = rag_from_csv()
    print("CSV Chatbot: Hello! I can answer questions about the data in the CSV file.")
    print("Type 'quit' to exit.")

    while True:
        user_query = input("\nYour question: ")

        if user_query.lower() == "quit":
            print("Goodbye!")
            break

        # Generate response
        response = chain.invoke(user_query)
        print("\nChatbot:", response)


if __name__ == "__main__":
    csv_chatbot()
