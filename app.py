import os
import streamlit as st
from dotenv import load_dotenv
from config import (
    Tone, CommunicationGoal, ResponseLength,
    ResponseStyle, UserPersona, DEFAULT_CONFIG,
    PERSONA_CONFIGS
)
from rag_engine import RAGEngine
from chatbot import PersonalizedChatbot

# Load environment variables
load_dotenv()

# Initialize session state
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = RAGEngine(use_openai=False)  # Use local embeddings
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = PersonalizedChatbot(st.session_state.rag_engine)

def main():
    st.title("Personalized RAG Chatbot")
    
    # Sidebar for personalization settings
    with st.sidebar:
        st.header("Personalization Settings")
        
        # Persona selection
        selected_persona = st.selectbox(
            "Select Persona",
            ["Custom", "Beginner", "Expert", "Young Learner"]
        )
        
        if selected_persona != "Custom":
            st.session_state.chatbot.update_config(PERSONA_CONFIGS[selected_persona])
            st.success(f"Applied {selected_persona} persona settings!")
        else:
            # Manual configuration
            tone = st.selectbox("Tone", [t.value for t in Tone])
            goal = st.selectbox("Communication Goal", [g.value for g in CommunicationGoal])
            length = st.selectbox("Response Length", [l.value for l in ResponseLength])
            style = st.selectbox("Response Style", [s.value for s in ResponseStyle])
            language = st.text_input("Language", "English")
            persona = st.selectbox("User Persona", [p.value for p in UserPersona])
            
            if st.button("Apply Settings"):
                config = DEFAULT_CONFIG.copy()
                config.tone = Tone(tone)
                config.communication_goal = CommunicationGoal(goal)
                config.response_length = ResponseLength(length)
                config.response_style = ResponseStyle(style)
                config.language = language
                config.user_persona = UserPersona(persona)
                st.session_state.chatbot.update_config(config)
                st.success("Settings applied!")
        
        # Document upload
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            try:
                # Save uploaded files temporarily
                temp_files = []
                for file in uploaded_files:
                    temp_path = f"temp_{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getvalue())
                    temp_files.append(temp_path)
                
                # Process documents
                st.session_state.rag_engine.process_documents(temp_files)
                st.success("Documents processed successfully!")
                
                # Clean up temporary files
                for temp_file in temp_files:
                    os.remove(temp_file)
                    
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
    
    # Chat interface
    st.header("Chat")
    
    # Display chat history
    for message in st.session_state.chatbot.conversation_history:
        with st.chat_message("user"):
            st.write(message["user"])
        with st.chat_message("assistant"):
            st.write(message["assistant"])
    
    # Chat input
    prompt = st.chat_input("Ask a question...")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            response = st.session_state.chatbot.get_response(prompt)
            st.write(response)
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chatbot.clear_history()
        st.success("Chat history cleared!")

if __name__ == "__main__":
    main() 