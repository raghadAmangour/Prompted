import streamlit as st
import pandas as pd

st.set_page_config(page_title="Prompted", page_icon="🤖", layout="wide")

st.title("🤖 Prompted: AI Support Ticket Triage Agent")
st.write("---")

# نظام محاكاة تسجيل الدخول للفصل بين العميل والموظف
st.sidebar.title("🔐 Access Control")
user_role = st.sidebar.selectbox("Select User Role:", ["Customer", "Support Employee"])

# بيانات افتراضية للشكاوى تظهر في صفحة الموظف كمثال للمدرب
mock_tickets = [
    {"Ticket ID": "T-101", "Customer": "Sarah Ahmed", "Complaint": "My credit card was charged twice for the same subscription.", "Status": "New"},
    {"Ticket ID": "T-102", "Customer": "John Doe", "Complaint": "The mobile application crashes every time I try to upload my profile picture.", "Status": "New"},
]

# 👤 1. واجهة العميل (Customer Portal)
if user_role == "Customer":
    st.subheader("📥 Submit a Complaint")
    customer_name = st.text_input("Your Name", placeholder="e.g., Sarah Ahmed")
    complaint_text = st.text_area("Describe your issue here...", height=150)
    
    if st.button("Submit Ticket", use_container_width=True):
        st.success("Thank you! Your ticket has been submitted to our AI Triage system.")

# 💼 2. واجهة الموظف والشركة (Internal Dashboard)
elif user_role == "Support Employee":
    st.subheader("💼 Internal Support Dashboard & AI Triage")
    
    # الجزء الأول: قائمة الشكاوى الواردة
    st.write("### 📋 Incoming Tickets Queue")
    df = pd.DataFrame(mock_tickets)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.write("---")
    
    # الجزء الثاني: اختيار شكوى معينة لرؤية تحليل الـ AI لها
    st.write("### 🔍 Select a Ticket to Review AI Triage")
    selected_id = st.selectbox("Choose Ticket ID to process:", df["Ticket ID"])
    
    # جلب تفاصيل الشكوى المختارة للعرض
    selected_ticket = df[df["Ticket ID"] == selected_id].iloc[0]
    st.text_area("Original Customer Complaint:", value=selected_ticket["Complaint"], disabled=True, height=70)
    
    st.write("#### 🧠 AI Automated Analysis Results")
    
    # توزيع مخرجات الـ AI في أعمدة منظمة
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 📊 ML Classification")
        st.info("**Predicted Category:** [Pending ML Model]\n\n**Urgency Score:** [Pending ML Model]")
        
    with col2:
        st.markdown("##### 📝 Generative AI")
        st.info("**Automated Summary:** [Pending LLM]\n\n**Draft Response:** [Pending LLM]")
        
    with col3:
        st.markdown("##### 🤖 AI Agent Route")
        st.info("**Target Department:** [Pending Agent]\n\n**Next Best Action:** [Pending Agent]")

    # زر اتخاذ إجراء للموظف
    st.button("Approve AI Triage & Route Ticket", use_container_width=True)
