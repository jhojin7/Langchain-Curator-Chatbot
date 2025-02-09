import streamlit as st
from langchain_core.runnables import RunnablePassthrough


def st_main_ui():
    CHAT_RUNNABLE: RunnablePassthrough = st.session_state["chain"]

    # Display chat messages from history on app rerun
    # Custom avatar for the assistant, default avatar for user
    for message in st.session_state["messages"]:
        if message["role"] == "assistant":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat logic
    if query := st.chat_input(placeholder="떡볶이 맛집 추천해줘."):
        # Add user message to chat history
        msg = {"role": "user", "content": query}
        st.session_state.messages.append(msg)
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response = CHAT_RUNNABLE.invoke(
                {"messages": st.session_state.messages},
                config=st.session_state["chainConfig"],
            )
            response_msg = response["messages"][-1].content
            message_placeholder.markdown(response_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": response_msg}
            )
