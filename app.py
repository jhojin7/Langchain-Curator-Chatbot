import streamlit as st
from st_sidebar_ui import st_sidbar_ui
from st_main_ui import st_main_ui

st.set_page_config(page_title="맛집 큐레이션 챗봇 데모")
# "미쉐린 가이드에 오른 맛집 중, 외국인에게 인기 있는 곳은 어디인가요?"


def intro_section():
    c = st.container(border=True)
    c.title("🍰 맛집 큐레이션 챗봇 데모")
    c.markdown(
        """1. 왼쪽 사이드바에서 `시스템 프롬프트, temperature, OpenAI API Key(선택)`를 선택하고 `채팅시작`을 눌러 시작해주세요.
2. 메인 화면에서 챗봇과 대화해보세요.
"""
    )
    c.markdown("아직은 한정적인 데이터로 인해 모르는 내용이 있을 수도 있어요.")
    return c


intro_section()
st_sidbar_ui()
if "messages" not in st.session_state.keys():
    st.info("챗봇 초기화에 실패했습니다. 사이드바에서 채팅을 초기화해주세요.")
else:
    st_main_ui()
