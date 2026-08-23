import streamlit as st

st.set_page_config(page_title="Prompted", page_icon="🤖", layout="wide")

st.title("🤖 Prompted: AI Support Ticket Triage Agent")
st.write("---")

# الفصل بين واجهة العميل وواجهة موظف الشركة
tab1, tab2 = st.tabs(["👤 Customer View (What the client sees)", "💼 Support Team View (What the company sees)"])

# ----------------- واجهة العميل -----------------
with tab1:
    st.subheader("📥 Submit a Complaint")
    st.write("This is the only screen visible to your customer.")
    
    c_name = st.text_input("Your Name", key="cust_name")
    c_text = st.text_area("Describe your issue here...", height=150, key="cust_text")
    
    if st.button("Submit Ticket", use_container_width=True):
        st.success("Thank you! Your ticket has been submitted successfully. Our team will review it shortly.")

# ----------------- واجهة الموظف والذكاء الاصطناعي -----------------
with tab2:
    st.subheader("🧠 Internal AI Triage & Routing Dashboard")
    st.write("This background analysis is hidden from the customer and only visible to the support team.")
    
    # محاكاة لشكوى وصلت من عميل لرؤية النتيجة
    st.info("📊 **Current Active Ticket:** Simulated data to demonstrate the AI workflow.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 ML Classification")
        st.info("**Issue Type:** [Pending]\n\n**Urgency Level:** [Pending]")
        
    with col2:
        st.markdown("### 📝 Generative AI")
        st.info("**Summary:** [Pending]\n\n**Suggested Draft Response:** [Pending]")
        
    with col3:
        st.markdown("### 🤖 AI Agent Route")
        st.info("**Assigned Department:** [Pending]\n\n**Next Action:** [Pending]")
