import streamlit as st
import pandas as pd

st.set_page_config(page_title="Prompted", page_icon="🤖", layout="wide")

st.title("🤖 Prompted: AI Support Ticket Triage Agent")
st.write("---")

# 1. إنشاء الذاكرة المؤقتة وتعبئتها بتذاكر افتراضية في البداية
if "tickets" not in st.session_state:
    st.session_state.tickets = [
        {"Ticket ID": "T-101", "Customer": "Sarah Ahmed", "Complaint": "My credit card was charged twice for the same subscription.", "Status": "New"},
        {"Ticket ID": "T-102", "Customer": "John Doe", "Complaint": "The mobile application crashes every time I try to upload my profile picture.", "Status": "New"},
    ]

# التحكم في نوع المستخدم
st.sidebar.title("🔐 Access Control")
user_role = st.sidebar.selectbox("Select User Role:", ["Customer", "Support Employee"])

# 👤 2. واجهة العميل (Customer Portal)
if user_role == "Customer":
    st.subheader("📥 Submit a Complaint")
    customer_name = st.text_input("Your Name", placeholder="e.g., Khaled Mohamed")
    complaint_text = st.text_area("Describe your issue here...", height=150, placeholder="Write your complaint...")
    
    if st.button("Submit Ticket", use_container_width=True):
        if customer_name and complaint_text:
            # توليد رقم تذكرة جديد تلقائياً وإضافته للذاكرة
            new_id = f"T-{101 + len(st.session_state.tickets)}"
            new_ticket = {
                "Ticket ID": new_id,
                "Customer": customer_name,
                "Complaint": complaint_text,
                "Status": "New"
            }
            st.session_state.tickets.append(new_ticket) # حفظ الشكوى في الذاكرة
            st.success(f"Thank you {customer_name}! Your ticket ({new_id}) has been submitted successfully to our AI Triage system. Switch to 'Support Employee' role to view it!")
        else:
            st.warning("Please fill in both your name and complaint details.")

# 💼 3. واجهة الموظف والذكاء الاصطناعي (الحذف والتحديث التلقائي)
elif user_role == "Support Employee":
    st.subheader("💼 Internal Support Dashboard & AI Triage")
    
    # تحقق إذا كانت جميع التذاكر قد حُلت واختفت
    if len(st.session_state.tickets) == 0:
        st.balloons()
        st.success("🎉 All tickets have been resolved! Great job team!")
    else:
        st.write("### 📋 Incoming Tickets Queue")
        df = pd.DataFrame(st.session_state.tickets)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.write("---")
        
        st.write("### 🔍 Select a Ticket to Review AI Triage")
        selected_id = st.selectbox("Choose Ticket ID to process:", df["Ticket ID"])
        
        # جلب تفاصيل الشكوى المختارة
        selected_ticket = df[df["Ticket ID"] == selected_id].iloc[0]
        st.text_area("Original Customer Complaint:", value=selected_ticket["Complaint"], disabled=True, height=70)
        
        st.write("#### 🧠 AI Automated Analysis Results")
        
        # تحليل ديناميكي ذكي بناءً على نص الشكوى المختارة
        if "credit" in selected_ticket["Complaint"].lower() or "charge" in selected_ticket["Complaint"].lower() or "money" in selected_ticket["Complaint"].lower():
            ml_cat, ml_urg = "💳 Billing & Payments", "🔴 High"
            gen_sum = "Customer is reporting a financial or transaction issue."
            gen_draft = f"Dear {selected_ticket['Customer']}, we are reviewing your billing transaction now."
            agent_dept, agent_act = "🏦 Finance Department", "Verify payment gateway logs."
        elif "crash" in selected_ticket["Complaint"].lower() or "app" in selected_ticket["Complaint"].lower() or "error" in selected_ticket["Complaint"].lower():
            ml_cat, ml_urg = "📱 Technical / Bug", "🟡 Medium"
            gen_sum = "Customer is experiencing a technical glitch or application crash."
            gen_draft = f"Dear {selected_ticket['Customer']}, our technical support is investigating the application error."
            agent_dept, agent_act = "💻 IT & Development Team", "Check system bug logs."
        else:
            ml_cat, ml_urg = "📂 General Inquiry", "🟢 Low"
            gen_sum = "General customer support request."
            gen_draft = f"Dear {selected_ticket['Customer']}, thank you for contacting us. We will reply shortly."
            agent_dept, agent_act = "👥 Customer Service Team", "Assign to general support agent."

        # عرض البيانات في الأعمدة الثلاثة
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
                # كود الحذف اللحظي والتوجيه عند الضغط
                st.session_state.tickets = [t for t in st.session_state.tickets if t["Ticket ID"] != selected_id]
                st.success(f"Ticket {selected_id} successfully routed to {agent_dept}!")
                st.rerun() # إعادة إنعاش الصفحة لمشاهدة التحديث فوراً
                
        with action_col2:
            if st.button("✅ Resolve & Close Ticket Immediately", use_container_width=True):
                # كود الحذف اللحظي والحل عند الضغط
                st.session_state.tickets = [t for t in st.session_state.tickets if t["Ticket ID"] != selected_id]
                st.success(f"Excellent! Ticket {selected_id} is now CLOSED and resolved.")
                st.rerun() # إعادة إنعاش الصفحة لمشاهدة التحديث فوراً
