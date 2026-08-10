import streamlit as st
from chatbot import chatbot_response


st.set_page_config(
    page_title="BVIT Student Chatbot",
    layout="centered"
)

st.title("BVIT Student Chatbot")

st.write(
    "Welcome to the BVIT Student Chatbot. "
    "Please type a message and press Enter to start the conversation."
)

user_input = st.text_input("You:")

if user_input:

    response = chatbot_response(user_input)

    st.write("Chatbot:")

    st.text_area(
        "",
        response,
        height=150
    )