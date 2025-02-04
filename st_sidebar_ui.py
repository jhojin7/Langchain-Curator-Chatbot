import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from csv_retrieve import rag_from_csv, create_retriever
from pathlib import Path

sample_system_prompt = """You are a helpful chatbot that recommends places to eat based the user's question.
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


def write_file(st_uploaded_file: UploadedFile) -> Path:
    if not st_uploaded_file:
        return None
    file_name = st_uploaded_file.name
    file_path = Path("cache", file_name)
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(st_uploaded_file.getvalue())
    return file_path


def st_sidbar_ui() -> bool:
    st.sidebar.header("Configuration")
    csv_file = st.sidebar.file_uploader(
        "Upload CSV File",
        type=["csv"],
        # accept_multiple_files=True
    )
    uploaded_csv_path = write_file(csv_file)
    print(csv_file)
    print(uploaded_csv_path)

    system_prompt = st.sidebar.text_area(
        "System Prompt", sample_system_prompt, height=200
    )

    temperature = st.sidebar.number_input(
        "LLM Temperature", value=0.3, step=0.05, min_value=0.0, max_value=1.0
    )
    openai_api_key_default = (
        st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else ""
    )
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key", type="password", value=openai_api_key_default
    )
    openai_model_name = st.sidebar.text_input("OpenAI 모델 이름", value="gpt-3.5-turbo")
    openai_model_name = st.sidebar.selectbox(
        "OpenAI 모델 이름", ["gpt-3.5-turbo", "gpt-4o-mini", "o3-mini"]
    )

    if st.sidebar.button("Initialize Chat Session"):
        _retriever = create_retriever(csv_path=uploaded_csv_path)
        if not _retriever:
            st.sidebar.error("Failed to create the retriever.")
        elif not csv_file:
            st.sidebar.error("Please upload a CSV file.")
        elif not openai_api_key:
            st.sidebar.error("Please enter the OpenAI API Key.")
        else:
            # Call the backend /initialize endpoint
            files = {"csv_file": csv_file}
            # Note: Form fields are sent as strings.
            data = {
                "system_prompt": system_prompt,
                "temperature": str(temperature),
                "openai_api_key": openai_api_key,
            }
            st.session_state["data"] = data

            st.session_state["retriever"] = _retriever
            st.session_state["chain"] = rag_from_csv(retriever=_retriever)
            print(st.session_state)

        # # Initialize LLM chain in session_state
        # if "chain" not in st.session_state:
        #     st.session_state["chain"] = rag_from_csv()

        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": "안녕하세요! 맛집 큐레이션 챗봇입니다. 무엇을 도와드릴까요?",
                }
            ]
    return True
