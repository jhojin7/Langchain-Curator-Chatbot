import time
import streamlit as st
from csv_retrieve import rag_from_csv

# Configure streamlit page
st.set_page_config(page_title="맛집 큐레이션 챗봇 데모")

# Initialize system message in session_state
st.session_state[
    "system_message"
] = """You are a helpful chatbot that recommends places to eat based the user's question.
Recommend 3-4 restaurants that meet user's requirements.
Use ONLY the context provided to answer the question.
If you cannot find the answer in the context, say so.
Please provide your answer in markdown format.
List each recommended restaurant as a separate heading.
Each restaurant MUST include its name, address, and a brief 3-4 sentence description of why it is recommended.
Do not recommend anything that is not in the context.

This is an example of how the answer should be formatted:
## 봉화묵집
- **주소:** 서울 성북구 아리랑로19길 46-2
- **설명:** 이 곳은 메밀묵, 손칼국수, 손만두 등을 판매하는 한식 음식점으로, 전통적이고 정직한 음식을 제공합니다. 방문자들의 리뷰에는 깔끔하고 건강한 식사를 원할 때 방문하기 좋다는 평가가 많이 나왔습니다. 또한, 묵사발과 만두 등의 메뉴가 맛있게 준비되어 있습니다.
"""

# Display system message box at the top
system_prompt = st.text_area(
    "시스템 메시지",
    value=st.session_state["system_message"],
    height=200,
)
if "system_message" not in st.session_state:
    st.session_state["system_message"] = system_prompt

# Initialize LLM chain in session_state
if "chain" not in st.session_state:
    st.session_state["chain"] = rag_from_csv()

if "messages" not in st.session_state:
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
