import streamlit as st

st.set_page_config(page_title="Prompted", page_icon="🤖", layout="wide")

st.title("🤖 Prompted: AI Support Ticket Triage Agent")
st.write("---")

# 1. Customer Input Section
st.subheader("📥 Customer Input")
customer_name = st.text_input("Customer Name")
complaint_text = st.text_area("Enter complaint details here...", height=150)
submit_btn = st.button("Analyze & Route Ticket", use_container_width=True)

st.write("---")


