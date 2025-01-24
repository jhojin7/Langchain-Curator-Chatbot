import streamlit as st
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

openai.api_key = st.secrets["OPENAI_API_KEY"]
FAISS_PATH = "faiss_index_nosplit"

SYSTEM_TEMPLATE = """ 당신은 최고의 맛집 큐레이터로서 사용자가 묻는 내용을 검토하여, 자체 데이터베이스에서 검색된 연관 내용을 함께 검토하여 최고의 맛집을 추천해주는 역할을 합니다.
모르는 경우 답변할 수 없다고 말해주세요.
사용자의 질문은 <question>, </question> 태그로 감싸져 있습니다. 데이터베이스에서 검색된 내용들은 `<docs>, </docs>` 태그로 감싸져 있습니다.
각 문서는 `<doc>, </doc>` 태그로 감싸져 있습니다.
문서의 내용은 `<content>, </content>` 태그로 감싸져 있습니다.
문서의 메타데이터는 `<metadata>, </metadata>` 태그로 감싸져 있습니다.
답변은 마크다운 형식으로 작성해주세요. 추천하는 맛집들에 대해서 각 맛집마다 별도의 리스트 아이템으로 작성해주세요.
각 맛집은 이름, 주소를 포함하고, 그 아래 불릿 포인트 리스트로 왜 추천했는지 간단하게 3-4 문장으로 설명해주세요.
문서에 없는 내용을 답변하거나, 문서에 없는 맛집을 절대 추천해선 안됩니다. 
문서 중에서 사용자가 찾고자 하는 내용이 하나도 없다면, "죄송하지만 질문하신 내용에 대한 맛집을 찾을 수 없었어요." 라는 답변을 해주세요. 

<question>{question}</question>
<docs>{context}</docs>
"""


# https://python.langchain.com/docs/versions/migrating_chains/retrieval_qa/
def format_docs(docs: list[Document]):
    _output = ""
    for doc in docs:
        _output += f"<doc><content>{doc.page_content}</content><metadata>{doc.metadata}<metadata></doc>"
    print(_output)
    return _output


@st.cache_resource
def load_qa_chain():
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
    prompt = PromptTemplate(
        input_variables=["question", "context"], template=SYSTEM_TEMPLATE
    )

    llm = ChatOpenAI(temperature=0.2)
    qa_chain = (
        {
            "context": vector_store.as_retriever(search_kwargs={"k": 3}) | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain


@st.cache_resource
def load_chain():
    """
    The `load_chain()` function initializes and configures a conversational retrieval chain for
    answering user questions.
    :return: The `load_chain()` function returns a ConversationalRetrievalChain object.
    """

    # Load OpenAI embedding model
    embeddings = OpenAIEmbeddings()

    # Load OpenAI chat model
    llm = ChatOpenAI(temperature=0.2)

    # Load our local FAISS index as a retriever
    vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Create memory 'chat_history'
    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history")

    # Create the Conversational Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        get_chat_history=lambda h: h,
        verbose=True,
    )

    # Add system prompt to chain
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=SYSTEM_TEMPLATE
    )
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(
        prompt=QA_CHAIN_PROMPT
    )

    return chain
