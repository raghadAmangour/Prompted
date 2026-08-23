import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Prompted", page_icon="🤖", layout="wide")

# Application Header
st.title("🤖 Prompted: AI Support Ticket Triage Agent")
st.write("---")

# 1. SESSION STATE INITIALIZATION
# Creates a temporary simulated database to store tickets during the app runtime
if "tickets" not in st.session_state:
    st.session_state.tickets = [
        {"Ticket ID": "T-101", "Customer": "Sarah Ahmed", "Complaint": "My credit card was charged twice for the same subscription.", "Status": "New"},
        {"Ticket ID": "T-102", "Customer": "John Doe", "Complaint": "The mobile application crashes every time I try to upload my profile picture.", "Status": "New"},
    ]

# SIDEBAR: USER ACCESS CONTROL
# Simulated login system to separate the Customer Portal from the Internal Support Dashboard
st.sidebar.title("🔐 Access Control")
user_role = st.sidebar.selectbox("Select User Role:", ["Customer", "Support Employee"])

# 👤 2. CUSTOMER PORTAL
# This section simulates the frontend environment visible ONLY to the client
if user_role == "Customer":
    st.subheader("📥 Submit a Complaint")
    customer_name = st.text_input("Your Name", placeholder="e.g., Khaled Mohamed")
    complaint_text = st.text_area("Describe your issue here...", height=150, placeholder="Write your complaint...")
    
    # Handle ticket submission
    if st.button("Submit Ticket", use_container_width=True):
        if customer_name and complaint_text:
            # Generate a new unique Ticket ID and append it to the session state array
            new_id = f"T-{101 + len(st.session_state.tickets)}"
            new_ticket = {
                "Ticket ID": new_id,
                "Customer": customer_name,
                "Complaint": complaint_text,
                "Status": "New"
            }
            st.session_state.tickets.append(new_ticket) 
            st.success(f"Thank you {customer_name}! Your ticket ({new_id}) has been submitted successfully to our AI Triage system. Switch to 'Support Employee' role to view it!")
        else:
            st.warning("Please fill in both your name and complaint details.")

# 💼 3. SUPPORT EMPLOYEE DASHBOARD
# This section simulates the backend enterprise environment with internal AI analytics
elif user_role == "Support Employee":
    st.subheader("💼 Internal Support Dashboard & AI Triage")
    
    # Check if the queue is empty (All tickets resolved)
    if len(st.session_state.tickets) == 0:
        st.balloons()
        st.success("🎉 All tickets have been resolved! Great job team!")
    else:
        # Step A: Display Active Tickets Queue
        st.write("### 📋 Incoming Tickets Queue")
        df = pd.DataFrame(st.session_state.tickets)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.write("---")
        
        # Step B: Select Ticket for Analysis
        st.write("### 🔍 Select a Ticket to Review AI Triage")
        selected_id = st.selectbox("Choose Ticket ID to process:", df["Ticket ID"])
        
        # Fetch details of the selected ticket from the dataframe
        selected_ticket = df[df["Ticket ID"] == selected_id].iloc[0]
        st.text_area("Original Customer Complaint:", value=selected_ticket["Complaint"], disabled=True, height=70)
        
        st.write("#### 🧠 AI Automated Analysis Results")
        
        # Step C: Dynamic AI Simulation Pipeline (MOCKING AI OUTPUTS)
        # Evaluates the text keywords to simulate how real ML and LLM models will respond later
        complaint_lower = selected_ticket["Complaint"].lower()
        
        if "credit" in complaint_lower or "charge" in complaint_lower or "money" in complaint_lower:
            # Simulated outputs for Financial issues
            ml_cat, ml_urg = "💳 Billing & Payments", "🔴 High"
            gen_sum = "Customer is reporting a financial or transaction error."
            gen_draft = f"Dear {selected_ticket['Customer']}, we are reviewing your billing transaction now."
            agent_dept, agent_act = "🏦 Finance Department", "Verify payment gateway logs."
            
        elif "crash" in complaint_lower or "app" in complaint_lower or "error" in complaint_lower:
            # Simulated outputs for Software/Technical bugs
            ml_cat, ml_urg = "📱 Technical / Bug", "🟡 Medium"
            gen_sum = "Customer is experiencing a technical glitch or application crash."
            gen_draft = f"Dear {selected_ticket['Customer']}, our technical support is investigating the application error."
            agent_dept, agent_act = "💻 IT & Development Team", "Check system bug logs."
            
        else:
            # Simulated fallback outputs for general inquiries
            ml_cat, ml_urg = "📂 General Inquiry", "🟢 Low"
            gen_sum = "General customer support request."
            gen_draft = f"Dear {selected_ticket['Customer']}, thank you for contacting us. We will reply shortly."
            agent_dept, agent_act = "👥 Customer Service Team", "Assign to general support agent."

        # Step D: Render the AI Framework layout using Streamlit columns
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
        
        # Step E: Handle Workflow Action Buttons
        action_col1, action_col2 = st.columns(2)
        
        # Action 1: Route Ticket
        with action_col1:
            if st.button("🚀 Approve AI Triage & Route Ticket", use_container_width=True):
                # Remove the processed ticket from the state array and refresh the view
                st.session_state.tickets = [t for t in st.session_state.tickets if t["Ticket ID"] != selected_id]
                st.success(f"Ticket {selected_id} successfully routed to {agent_dept}!")
                st.rerun() 
                
        # Action 2: Instantly Close and Resolve Ticket
        with action_col2:
            if st.button("✅ Resolve & Close Ticket Immediately", use_container_width=True):
                # Remove the processed ticket from the state array and refresh the view with celebratory balloons
                st.session_state.tickets = [t for t in st.session_state.tickets if t["Ticket ID"] != selected_id]
                st.success(f"Excellent! Ticket {selected_id} is now CLOSED and resolved.")
                st.rerun()
