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

# 2. AI Outputs Structural Framework
st.subheader("🧠 AI Triage Outputs")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 ML Classification")
    st.info("Issue Type: [Pending]\n\nUrgency Level: [Pending]")
    
with col2:
    st.markdown("### 📝 Generative AI")
    st.info("Summary: [Pending]\n\nSuggested Response: [Pending]")
    
with col3:
    st.markdown("### 🤖 AI Agent Route")
    st.info("Assigned Department: [Pending]\n\nNext Best Action: [Pending]")
