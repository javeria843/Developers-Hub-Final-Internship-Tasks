import os
import streamlit as st
from app import RAGChatbot  # your RAGChatbot class

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 RAG Chatbot with Gemini + ChromaDB")

# Sidebar
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Google API Key", type="password")
build_index = st.sidebar.button("Build Index")

# Initialize chatbot in session state
if "chatbot" not in st.session_state and api_key:
    st.session_state.chatbot = RAGChatbot(api_key=api_key)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload documents
st.subheader("📂 Upload Documents")
uploaded_files = st.file_uploader("Upload PDFs or TXT files", accept_multiple_files=True)

if build_index and uploaded_files:
    with st.spinner("Indexing documents..."):
        for file in uploaded_files:
            temp_path = os.path.join("temp", file.name)
            os.makedirs("temp", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            if file.name.endswith(".pdf"):
                st.session_state.chatbot.add_pdf(temp_path)
            else:
                text = file.getvalue().decode("utf-8")
                st.session_state.chatbot.add_text(text, doc_id=file.name)
        st.success("✅ Documents indexed!")

# Chat UI
st.subheader("💬 Chat with your documents")
user_input = st.text_input("Ask a question:", key="user_input")
send_button = st.button("Send")

if send_button and user_input.strip():
    with st.spinner("Thinking..."):
        answer = st.session_state.chatbot.query(user_input)
        st.session_state.messages.append(("User", user_input))
        st.session_state.messages.append(("Bot", answer))
    st.session_state.user_input = ""  # clear input

# Display conversation
for role, msg in st.session_state.messages:
    if role == "User":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")
