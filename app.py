import streamlit as st

st.set_page_config(page_title="Prompted", page_icon="🤖", layout="wide")

st.title("🤖 Prompted: AI Support Ticket Triage Agent")
st.write("---")

# نظام محاكاة تسجيل الدخول للفصل بين العميل والموظف
st.sidebar.title("🔐 Access Control")
user_role = st.sidebar.selectbox("Select User Role (For Demo Only):", ["Customer", "Support Employee"])

# 👤 1. واجهة العميل (Customer Portal)
if user_role == "Customer":
    st.subheader("📥 Submit a Complaint")
    customer_name = st.text_input("Your Name", placeholder="e.g., Sarah Ahmed")
    complaint_text = st.text_area("Describe your issue here...", height=150)
    
    if st.button("Submit Ticket", use_container_width=True):
        st.success("Thank you! Your ticket has been submitted to our AI Triage system.")

# 💼 2. واجهة الموظف والشركة (Internal Dashboard)
elif user_role == "Support Employee":
    st.subheader("⚙️ Internal Support Dashboard & AI Triage")
    st.write("This secure dashboard is hidden from customers and used only by the company team.")
    
    # محاكاة تذكرة نشطة ليرى المدرب كيف سيعمل النظام مستقبلاً
    st.info("📊 **Active Ticket Under Review:** Showing how incoming text is routed by AI.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 ML Classification")
        st.info("**Issue Type:** [Pending ML Model]\n\n**Urgency Level:** [Pending ML Model]")
        
    with col2:
        st.markdown("### 📝 Generative AI")
        st.info("**Summary:** [Pending LLM]\n\n**Suggested Draft:** [Pending LLM]")
        
    with col3:
        st.markdown("### 🤖 AI Agent Route")
        st.info("**Target Department:** [Pending Agent]\n\n**Next Action:** [Pending Agent]")
