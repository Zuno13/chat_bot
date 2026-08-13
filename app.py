import streamlit as st
from chatbot import chatbot_response

st.set_page_config(
    page_title="BVIT Student Chatbot",
    layout="centered"
)

st.title("BVIT Student Chatbot")

st.write(
    "Welcome to the BVIT Student Chatbot. "
    "Ask a question related to BVIT or MSBTE."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your message...")

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    response = chatbot_response(user_input)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    with st.chat_message("assistant"):
        st.write(response)