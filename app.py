import time
import streamlit as st
from csv_retrieve import rag_from_csv

# Configure streamlit page
st.set_page_config(page_title="맛집 큐레이션 챗봇 데모")
from st_sidebar_ui import st_sidbar_ui

CHAT_INIT: bool = st_sidbar_ui()

if "messages" not in st.session_state.keys():
    st.info("챗봇 초기화에 실패했습니다. 사이드바에서 채팅을 초기화해주세요.")
else:
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
    query = st.chat_input("떡볶이 맛집 추천해줘.")

    if query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response = st.session_state["chain"].invoke(query)
            print(response)
            print(st.session_state.messages)
            message_placeholder.markdown(response)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
