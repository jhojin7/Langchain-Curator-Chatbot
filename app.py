import time
import streamlit as st
from utils import load_chain, load_qa_chain

# Configure streamlit page
st.set_page_config(page_title="맛집 큐레이션 챗봇 데모")

# Initialize LLM chain in session_state
if "chain" not in st.session_state:
    st.session_state["chain"] = load_qa_chain()

# Initialize chat history
if "messages" not in st.session_state:
    # Start with first message from assistant
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "안녕하세요! 맛집 큐레이션 챗봇입니다. 무엇을 도와드릴까요?",
        }
    ]

# Display chat messages from history on app rerun
# Custom avatar for the assistant, default avatar for user
for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat logic
if query := st.chat_input("떡볶이 맛집 추천해줘."):
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
