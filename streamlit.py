import streamlit as st
import requests
import uuid
import json
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")
st.title("My AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Render FastAPI endpoint
    streaming_url = os.getenv("STREAMING_URL")
    
    payload = {
        "message": prompt,
        "thread_id": st.session_state.thread_id
    }

    with st.chat_message("assistant"):
        
        # Generator function to stream tokens from FastAPI to Streamlit
        def generate_tokens():
            try:
                # stream=True keeps the HTTP connection open
                response = requests.post(streaming_url, json=payload, stream=True)
                response.raise_for_status()

                # Iterate through the Server-Sent Events (SSE)
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        
                        # Check if the line is an SSE data packet
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:] # Strip out the "data: " prefix
                            
                            if data_str == "[DONE]":
                                break
                            
                            try:
                                json_data = json.loads(data_str)
                                
                                if "token" in json_data:
                                    yield json_data["token"]
                                elif "error" in json_data:
                                    yield f"\n\n**Error:** {json_data['error']}"
                                    
                            except json.JSONDecodeError:
                                continue 

            except requests.exceptions.RequestException as network_error:
                yield f"Connection failure. Render service may be warming up. Error details: {network_error}"

        ai_response = st.write_stream(generate_tokens())

    st.session_state.messages.append({"role": "assistant", "content": ai_response})

    