import streamlit as st
import psutil
import time
from engine import generate_full_response  # Ensure your engine.py is in the same src folder

# --- UI CONFIGURATION ---
st.set_page_config(page_title="TinyLlama Private AI", layout="wide")

# --- CUSTOM CSS FOR PASTEL TOUCHES ---
st.markdown("""
    <style>
    .stApp {
        background-color: #FDFDFD;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: MLOPS DASHBOARD ---
with st.sidebar:
    st.title("⚙️ Engine Control")
    
    # Toggle for Loan Buddy Mode
    loan_buddy_mode = st.toggle("🚀 Enable Loan Buddy Mode", value=False)
    
    st.divider()
    st.subheader("📊 Live MLOps Metrics")
    # Placeholders for live updates
    ram_metric = st.sidebar.empty()
    speed_metric = st.sidebar.empty()
    
    # System Prompt Logic based on Toggle
    if loan_buddy_mode:
        system_msg = "You are a professional Loan Officer for Loan Buddy. Assist with eligibility and document queries."
        st.info("Mode: Loan Buddy Agent")
    else:
        system_msg = "You are a helpful and concise AI assistant."
        st.info("Mode: General Assistant")

# --- MAIN CHAT INTERFACE ---
st.title("🦙 TinyLlama Local CPU")
st.caption("Quantized ONNX Inference | Ryzen 3 Optimized")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Type your message here..."):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Monitor RAM right before inference
        current_ram = psutil.virtual_memory().percent
        ram_metric.metric("RAM Usage", f"{current_ram}%")

        # Call the engine and get response + speed
        with st.spinner("Generating..."):
            res_text, tps = generate_full_response(prompt, system_prompt=system_msg)
        
        # SIMULATING STREAMING EFFECT
        # (Since we decoded fully in engine, we stream the final string here)
        for chunk in res_text.split():
            full_response += chunk + " "
            time.sleep(0.05) # Adjust speed of typing
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
        
        # Update speed metric
        speed_metric.metric("Inference Speed", f"{tps:.2f} tokens/sec")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})