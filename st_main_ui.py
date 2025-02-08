import streamlit as st


def st_main_ui():
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
        msg = {"role": "user", "content": query}
        st.session_state.messages.append(msg)
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response = (
                st.session_state["chain"]
                .invoke(
                    {"messages": st.session_state.messages},
                    config=st.session_state["chainConfig"],
                )["messages"][-1]
                .content
            )
            print(response)
            print(st.session_state.messages)
            message_placeholder.markdown(response)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
