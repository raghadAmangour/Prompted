import streamlit as st

st.title("Hello Streamlit Cloud! 🚀")
name = st.text_input("What is your name?")

if name:
    st.write(f"Welcome, {name}! Your cloud app is working perfectly.")
    st.balloons()
