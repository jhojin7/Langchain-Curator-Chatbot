import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from csv_retrieve import rag_from_csv, create_retriever
from pathlib import Path

sample_system_prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
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
        "CSV 파일 업로드",
        type=["csv"],
        # accept_multiple_files=True
    )
    uploaded_csv_path = write_file(csv_file)
    print(csv_file)
    print(uploaded_csv_path)

    system_prompt = st.sidebar.text_area(
        "시스템 프롬프트", sample_system_prompt, height=200
    )

    temperature = st.sidebar.number_input(
        "LLM Temperature", value=0.1, step=0.05, min_value=0.0, max_value=1.0
    )
    openai_api_key_default = (
        st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else ""
    )
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key", type="password", value=openai_api_key_default
    )
    openai_model_name = st.sidebar.selectbox(
        "OpenAI 모델 이름", ["gpt-3.5-turbo", "gpt-4o-mini", "o1-mini"]
    )

    if st.sidebar.button("채팅 시작"):
        try:
            _retriever = create_retriever(csv_path=uploaded_csv_path)
        except Exception as e:
            st.sidebar.error(f"파일 업로드에 실패했습니다: {e}")
            return False

        if not csv_file:
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
            st.session_state["chain"] = rag_from_csv(
                retriever=_retriever,
                model_name=openai_model_name,
                temp=temperature,
                api_key=openai_api_key,
            )
            print(st.session_state)

        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": "안녕하세요! 맛집 큐레이션 챗봇입니다. 무엇을 도와드릴까요?",
                }
            ]
    return True
