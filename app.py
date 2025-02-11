import streamlit as st
from st_sidebar_ui import st_sidbar_ui
from st_main_ui import st_main_ui

st.set_page_config(page_title="맛집 큐레이션 챗봇 데모")
st.title("맛집 큐레이션 챗봇 데모")


def intro_section():
    c = st.container(border=True)
    c.markdown("## 🤖 챗봇 소개")
    return c


st_sidbar_ui()
if "messages" not in st.session_state.keys():
    st.info("챗봇 초기화에 실패했습니다. 사이드바에서 채팅을 초기화해주세요.")
    intro_section()
else:
    st_main_ui()
