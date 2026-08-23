import streamlit as st
import pandas as pd

st.set_page_config(page_title="Prompted", page_icon="🤖", layout="wide")

st.title("🤖 Prompted: AI Support Ticket Triage Agent")
st.write("---")

# نظام التحكم في نوع المستخدم
st.sidebar.title("🔐 Access Control")
user_role = st.sidebar.selectbox("Select User Role:", ["Customer", "Support Employee"])

# بيانات افتراضية للشكاوى (مشكلتين مختلفتين تماماً)
mock_tickets = [
    {"Ticket ID": "T-101", "Customer": "Sarah Ahmed", "Complaint": "My credit card was charged twice for the same subscription.", "Status": "New"},
    {"Ticket ID": "T-102", "Customer": "John Doe", "Complaint": "The mobile application crashes every time I try to upload my profile picture.", "Status": "New"},
]

# 👤 1. واجهة العميل
if user_role == "Customer":
    st.subheader("📥 Submit a Complaint")
    customer_name = st.text_input("Your Name", placeholder="e.g., Sarah Ahmed")
    complaint_text = st.text_area("Describe your issue here...", height=150)
    
    if st.button("Submit Ticket", use_container_width=True):
        st.success("Thank you! Your ticket has been submitted to our AI Triage system.")

# 💼 2. واجهة الموظف والذكاء الاصطناعي (صفحة واحدة متغيرة ديناميكياً)
elif user_role == "Support Employee":
    st.subheader("💼 Internal Support Dashboard & AI Triage")
    
    st.write("### 📋 Incoming Tickets Queue")
    df = pd.DataFrame(mock_tickets)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.write("---")
    
    st.write("### 🔍 Select a Ticket to Review AI Triage")
    selected_id = st.selectbox("Choose Ticket ID to process:", df["Ticket ID"])
    
    # جلب تفاصيل الشكوى المختارة
    selected_ticket = df[df["Ticket ID"] == selected_id].iloc[0]
    st.text_area("Original Customer Complaint:", value=selected_ticket["Complaint"], disabled=True, height=70)
    
    st.write("#### 🧠 AI Automated Analysis Results")
    
    # تفريغ البيانات ديناميكياً بناءً على نوع المشكلة المختارة
    if selected_id == "T-101":  # مشكلة الدفع
        ml_cat, ml_urg = "💳 Billing & Payments", "🔴 High"
        gen_sum = "Customer was double-charged for a single subscription."
        gen_draft = "Dear Sarah, we detected the double charge. A refund has been initiated."
        agent_dept, agent_act = "🏦 Finance Department", "Trigger automated refund via Stripe API."
    else:  # مشكلة تقنية
        ml_cat, ml_urg = "📱 Technical / Bug", "🟡 Medium"
        gen_sum = "Mobile app crashes specifically during profile picture upload."
        gen_draft = "Dear John, our technical team is investigating the app crash issue."
        agent_dept, agent_act = "💻 IT & Development Team", "Assign ticket to Mobile Dev QA queue."

    # عرض البيانات المتغيرة في الأعمدة الثلاثة
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### 📊 ML Classification")
        st.info(f"**Category:** {ml_cat}\n\n**Urgency:** {ml_urg}")
    with col2:
        st.markdown("##### 📝 Generative AI")
        st.info(f"**Summary:** {gen_sum}\n\n**Draft:** {gen_draft}")
    with col3:
        st.markdown("##### 🤖 AI Agent Route")
        st.info(f"**Target:** {agent_dept}\n\n**Action:** {agent_act}")

    st.write("---")
    st.write("### 🛠️ Resolution Action")
    
    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("🚀 Approve AI Triage & Route Ticket", use_container_width=True):
            st.success(f"Ticket {selected_id} successfully routed to {agent_dept}!")
    with action_col2:
        if st.button("✅ Resolve & Close Ticket Immediately", use_container_width=True):
            st.balloons()
            st.success(f"Excellent! Ticket {selected_id} is now CLOSED and resolved.")
