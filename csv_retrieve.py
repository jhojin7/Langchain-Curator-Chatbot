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

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful chatbot that recommends places to eat based the user's question.
Recommend 3-4 restaurants that meet user's requirements.
Use ONLY the context provided to answer the question.
If you cannot find the answer in the context, say so.
Please provide your answer in markdown format.
List each recommended restaurant as a separate heading.
Each restaurant MUST include its name, address, and a brief 3-4 sentence description of why it is recommended.
Do not recommend anything that is not in the context.

This is an example of how the answer should be formatted:
## 봉화묵집
**주소:** 서울 성북구 아리랑로19길 46-2
**설명:** 이 곳은 메밀묵, 손칼국수, 손만두 등을 판매하는 한식 음식점으로, 전통적이고 정직한 음식을 제공합니다. 방문자들의 리뷰에는 깔끔하고 건강한 식사를 원할 때 방문하기 좋다는 평가가 많이 나왔습니다. 또한, 묵사발과 만두 등의 메뉴가 맛있게 준비되어 있습니다.

Context: {context}

Question: {question}
"""
)


# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def rag_from_csv():
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
