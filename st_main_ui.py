import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import MessagesState, StateGraph


def build_chat(message: str, docs: list[str]):
    c = st.container()
    c.markdown(message)
    if not docs:
        return c

    c.container(border=True)
    c.expander(label="찾은 문서 더 보기", expanded=False)
    d = c.container()
    for i, doc in enumerate(docs, 1):
        d.expander(label=f"문서 #{i}", expanded=False).text(doc)
    return c


def generate_response(query: str, messages: MessagesState, graph: StateGraph):
    chainConfig = st.session_state["chainConfig"]
    graph.invoke({"messages": messages}, config=chainConfig)
    _messages = graph.get_state(config=chainConfig).values["messages"]

    response_msg = _messages[-1].content
    retrieved_docs = _messages[-2].content
    return response_msg, retrieved_docs


def st_main_ui():
    CHAT_RUNNABLE: StateGraph = st.session_state["chain"]

    for message in st.session_state["messages"]:
        if message["role"] == "assistant":
            with st.chat_message(message["role"]):
                build_chat(message["content"], message.get("docs", []))
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat logic
    if query := st.chat_input(placeholder="떡볶이 맛집 추천해줘."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            response_msg, retrieved_docs = generate_response(
                query, st.session_state.messages, CHAT_RUNNABLE
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": response_msg, "docs": retrieved_docs}
            )
            build_chat(response_msg, retrieved_docs)
