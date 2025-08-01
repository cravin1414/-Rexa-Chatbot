import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from PIL import Image
import io
import base64

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "llama3:instruct"
    if "ollama_llm" not in st.session_state:
        st.session_state.ollama_llm = Ollama(model=st.session_state.selected_model)
    if "stop_generation" not in st.session_state:
        st.session_state.stop_generation = False
    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = []
    if "image_analyzed" not in st.session_state:
        st.session_state.image_analyzed = False

def get_available_models():
    """Get available Ollama models"""
    return ["llama2", "gemma:2b", "phi", "neural-chat","llama3:instruct","llama3:latest","nous-hermes2:latest"]

def reset_conversation():
    """Reset the conversation history"""
    st.session_state.messages = []
    st.session_state.ollama_llm = Ollama(model=st.session_state.selected_model)
    st.session_state.stop_generation = False
    st.session_state.uploaded_images = []
    st.session_state.image_analyzed = False

def stop_generation():
    """Set the stop generation flag"""
    st.session_state.stop_generation = True

def create_pdf(content, filename="output.pdf"):
    """Create a PDF file from the given content"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    text = c.beginText(40, height - 40)
    text.setFont("Helvetica", 12)
    
    lines = []
    for paragraph in content.split('\n'):
        words = paragraph.split()
        line = ''
        for word in words:
            if c.stringWidth(line + ' ' + word, "Helvetica", 12) < (width - 80):
                line += ' ' + word
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
    
    for line in lines:
        text.textLine(line)
    
    c.drawText(text)
    c.save()
    
    buffer.seek(0)
    return buffer

def analyze_image(img):
    """Analyze the uploaded image and generate a description"""
    # Convert image to bytes
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    
    # Create prompt with image context
    prompt = """
    Analyze this image in detail. Describe:
    1. The main subject and any important objects
    2. Colors and visual style
    3. Any text visible in the image
    4. Possible context or meaning
    5. Any other notable features
    
    Provide a comprehensive description that would help someone understand what the image contains.
    """
    
    # Use generate instead of chat
    response = st.session_state.ollama_llm.generate([prompt])
    return response.generations[0][0].text

def display_images():
    """Display uploaded images in the chat and analyze them"""
    for img in st.session_state.uploaded_images:
        with st.chat_message("human"):
            st.image(img, caption="Uploaded Image", use_container_width=True)
        
        if not st.session_state.image_analyzed:
            with st.spinner("Analyzing image..."):
                analysis = analyze_image(img)
                st.session_state.image_analyzed = True
                
                with st.chat_message("ai"):
                    st.markdown("**Image Analysis:**")
                    st.markdown(analysis)
                
                st.session_state.messages.append(AIMessage(content=f"Image Analysis:\n{analysis}"))

def main():
    st.set_page_config(page_title="Strongest Rexa Bot", page_icon="🤖")
    
    st.markdown("""
    <style>
        .stApp {
            background-color: #e5ddd5;
            background-image: url('https://web.whatsapp.com/img/bg-chat-tile-light_a4be512e7195b6b733d9110b408f075d.png');
            background-repeat: repeat;
        }
        .stChatInput {
            bottom: 20px;
            background-color: white;
            padding: 10px;
            border-radius: 8px;
        }
        .stChatMessage {
            padding: 8px 16px;
            border-radius: 7.5px;
            margin: 5px 0;
            max-width: 70%;
            word-wrap: break-word;
            box-shadow: 0 1px 0.5px rgba(0,0,0,0.13);
        }
        .stChatMessage[data-testid="chat-human-message"] {
            background-color: #dcf8c6;
            margin-left: auto;
            margin-right: 10px;
        }
        .stChatMessage[data-testid="chat-ai-message"] {
            background-color: white;
            margin-left: 10px;
        }
        .stChatMessageContainer {
            padding: 0 10px;
        }
        .chat-title {
            background-color: #128c7e;
            color: white;
            padding: 10px 16px;
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px 8px 0 0;
            margin-bottom: 10px;
            text-align: center;
        }
        .stMarkdown {
            font-family: Segoe UI, Helvetica Neue, Helvetica, sans-serif;
            font-size: 14px;
        }
        .sidebar .sidebar-content {
            background-color: #128c7e;
            color: white;
        }
        .stSelectbox, .stSlider, .stTextArea, .stButton>button {
            background-color: white;
            color: black;
        }
    </style>
    """, unsafe_allow_html=True)
    
    initialize_session_state()
    
    with st.sidebar:
        st.header("Configuration")
        
        available_models = get_available_models()
        new_model = st.selectbox(
            label="Select Ollama Model",
            options=available_models,
            index=available_models.index(st.session_state.selected_model),
            key="model_selectbox"
        )
        
        if new_model != st.session_state.selected_model:
            st.session_state.selected_model = new_model
            reset_conversation()
        
        temperature = st.slider(
            "Creativity (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values produce more creative but less predictable outputs"
        )
        
        system_prompt = st.text_area(
            "System Prompt",
            value=f"You are a helpful AI assistant talking to Rexa. Answer questions clearly and concisely, and remember you're talking to Rexa.",
            help="Guides the model's behavior throughout the conversation"
        )
        
        uploaded_file = st.file_uploader(
            "Upload an image", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state.uploaded_images = [image]
            st.session_state.image_analyzed = False
            st.success("Image uploaded successfully!")
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.button("Clear Conversation", on_click=reset_conversation)
        with col2:
            st.button("Stop Generation", on_click=stop_generation)
        
        st.markdown("---")
        st.markdown("💡 **Tip:** Press Shift+Enter to send your message")
        st.markdown("Made with [Ollama](https://ollama.ai/) and [Streamlit](https://streamlit.io/)")

    with st.container():
        st.markdown('<div class="chat-title">Rexa Chatbot</div>', unsafe_allow_html=True)
        
        for message in st.session_state.messages:
            if isinstance(message, HumanMessage):
                with st.chat_message("human"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("ai"):
                    st.markdown(message.content)
        
        display_images()

    if prompt := st.chat_input("Type your message here..."):
        st.session_state.stop_generation = False
        
        human_message = HumanMessage(content=prompt)
        st.session_state.messages.append(human_message)
        
        with st.chat_message("human"):
            st.markdown(prompt)
        
        with st.chat_message("ai"):
            message_placeholder = st.empty()
            full_response = ""
            
            conversation_context = [{"role": "system", "content": system_prompt}]
            for msg in st.session_state.messages:
                if isinstance(msg, HumanMessage):
                    conversation_context.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    conversation_context.append({"role": "assistant", "content": msg.content})
            
            for chunk in st.session_state.ollama_llm.stream(
                prompt,
                temperature=temperature,
                context=conversation_context
            ):
                if st.session_state.stop_generation:
                    full_response += " [Generation stopped by user]"
                    break
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            if "pdf" in prompt.lower() or "download" in prompt.lower():
                pdf_buffer = create_pdf(full_response)
                st.download_button(
                    label="Download as PDF",
                    data=pdf_buffer,
                    file_name="chat_response.pdf",
                    mime="application/pdf"
                )
        
        ai_message = AIMessage(content=full_response)
        st.session_state.messages.append(ai_message)

if __name__ == "__main__":
    main()