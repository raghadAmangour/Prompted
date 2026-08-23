import streamlit as st

st.set_page_config(page_title="Prompted", page_icon="🤖", layout="wide")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation:", 
    ["🏠 Home", "📥 Ticket Submission & Triage"]
)

# 1. Home Page
if page == "🏠 Home":
    st.title("🤖 Prompted Platform")
    st.subheader("AI Support Ticket Triage Agent")
    st.write("A smart system designed to help customer support teams handle and route customer complaints efficiently.")
    
    st.markdown("""
    ### 🎯 Core Features (Planned):
    * **Machine Learning:** Classifies the complaint type and predicts issue urgency/priority.
    * **Generative AI:** Summarizes long complaints and generates automated initial draft responses.
    * **AI Agent:** Determines the appropriate department and coordinates the next action.
    """)

# 2. Ticket Submission & Triage Page
elif page == "📥 Ticket Submission & Triage":
    st.title("📥 Submit & Process Complaint")
    
    # Customer Input Section
    st.write("### 1. Customer Input")
    customer_name = st.text_input("Customer Name")
    complaint_text = st.text_area("Enter complaint details here...", height=150)
    submit_btn = st.button("Analyze & Route Ticket")
    
    st.write("---")
    
    # AI Outputs Structural Framework (Placeholder for the next phase)
    st.write("### 🧠 AI Triage Outputs (Prototype Context)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📊 ML Classification")
        st.info("Issue Type: (Pending Integration)\n\nUrgency Level: (Pending Integration)")
        
    with col2:
        st.subheader("📝 Generative AI")
        st.info("Summary: (Pending Integration)\n\nSuggested Response: (Pending Integration)")
        
    with col3:
        st.subheader("🤖 AI Agent Route")
        st.info("Assigned Department: (Pending Integration)\n\nNext Best Action: (Pending Integration)")
