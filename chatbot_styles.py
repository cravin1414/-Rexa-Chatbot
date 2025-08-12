import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from PIL import Image
import io
import base64
import google.generativeai as genai
from typing import Optional
import sounddevice as sd
import numpy as np
import wave
from transformers import pipeline
from io import BytesIO
import time
import speech_recognition as sr
import tempfile

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
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = ""
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = "You are a helpful AI assistant talking to Rexa. Answer questions clearly and concisely, and remember you're talking to Rexa."
    if "generating" not in st.session_state:
        st.session_state.generating = False
    if "sidebar_collapsed" not in st.session_state:
        st.session_state.sidebar_collapsed = True
    if "response_complete" not in st.session_state:
        st.session_state.response_complete = False
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "audio_data" not in st.session_state:
        st.session_state.audio_data = None
    if "voice_assistant_enabled" not in st.session_state:
        st.session_state.voice_assistant_enabled = False
    if "speech_to_text_model" not in st.session_state:
        st.session_state.speech_to_text_model = None
    if "text_to_speech_model" not in st.session_state:
        st.session_state.text_to_speech_model = None
    if "speech_recognizer" not in st.session_state:
        st.session_state.speech_recognizer = None

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
    st.session_state.generating = False
    st.session_state.response_complete = False
    st.session_state.audio_data = None

def stop_generation():
    """Set the stop generation flag"""
    st.session_state.stop_generation = True
    st.session_state.generating = False
    st.session_state.response_complete = True

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

def setup_gemini_api(api_key: str) -> Optional[genai.GenerativeModel]:
    """Setup Gemini API with the provided key"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"Failed to setup Gemini API: {str(e)}")
        return None

def analyze_image_with_gemini(img: Image.Image, api_key: str) -> str:
    """Analyze image using Google Gemini Vision model"""
    try:
        if not api_key:
            return "❌ Please provide a Gemini API key to enable advanced image analysis."
        
        model = setup_gemini_api(api_key)
        if not model:
            return "❌ Failed to initialize Gemini model. Please check your API key."
        
        # Prepare the image
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        buffered.seek(0)
        
        # Create detailed prompt for comprehensive analysis
        prompt = """
        Analyze this image comprehensively and provide detailed information about:

        1. **Main Subject/Content**: What is the primary focus of this image? What does it show?

        2. **Objects and Elements**: List all visible objects, people, animals, or items in the image.

        3. **Scene Context**: Describe the setting, environment, or location where this image was taken.

        4. **Colors and Visual Style**: Describe the color palette, lighting, and overall visual aesthetic.

        5. **Text Content**: Identify and transcribe any text visible in the image (signs, labels, writing, etc.).

        6. **Actions/Activities**: Describe what's happening in the image - any actions, movements, or activities.

        7. **Emotions/Mood**: What emotions or mood does this image convey?

        8. **Purpose/Context**: What do you think this image is meant to represent or communicate? What might it be used for?

        9. **Technical Details**: Comment on image quality, composition, perspective, and any notable photographic aspects.

        10. **Additional Insights**: Any other interesting or notable details you observe.

        Please provide a thorough and detailed analysis covering all these aspects.
        """
        
        # Generate response
        response = model.generate_content([prompt, img])
        
        if response and response.text:
            return response.text
        else:
            return "❌ No response generated from Gemini model."
            
    except Exception as e:
        return f"❌ Error analyzing image with Gemini: {str(e)}"

def analyze_image_basic(img: Image.Image) -> str:
    """Basic image analysis without external APIs"""
    try:
        width, height = img.size
        mode = img.mode
        format_type = img.format if img.format else "Unknown"
        
        # Calculate file size
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        size_bytes = len(buffered.getvalue())
        size_mb = size_bytes / (1024 * 1024)
        
        analysis = f"""
        **📊 Basic Image Analysis:**
        
        **Technical Properties:**
        - Dimensions: {width} × {height} pixels
        - Color Mode: {mode}
        - Format: {format_type}
        - File Size: {size_mb:.2f} MB
        - Aspect Ratio: {width/height:.2f}
        
        **Visual Assessment:**
        - Resolution: {"High" if width > 1920 or height > 1080 else "Standard" if width > 640 else "Low"}
        - Image Quality: {"Good" if size_mb > 0.5 else "Compressed"}
        
        **Note:** For detailed content analysis (objects, scenes, text recognition), please provide a Gemini API key in the sidebar.
        """
        
        return analysis
        
    except Exception as e:
        return f"❌ Error in basic image analysis: {str(e)}"

def analyze_image(img: Image.Image) -> str:
    """Main image analysis function that chooses between Gemini or basic analysis"""
    api_key = st.session_state.gemini_api_key.strip()
    
    if api_key:
        # Use advanced Gemini analysis
        return analyze_image_with_gemini(img, api_key)
    else:
        # Use basic analysis
        return analyze_image_basic(img)

def display_images():
    """Display uploaded images in the chat and analyze them"""
    for img in st.session_state.uploaded_images:
        with st.chat_message("human"):
            st.image(img, caption="Uploaded Image", use_container_width=True)
        
        if not st.session_state.image_analyzed:
            with st.spinner("🔍 Analyzing image with AI..."):
                analysis = analyze_image(img)
                st.session_state.image_analyzed = True
                
                with st.chat_message("ai"):
                    st.markdown("**🤖 AI Image Analysis:**")
                    st.markdown(analysis)
                
                st.session_state.messages.append(AIMessage(content=f"AI Image Analysis:\n{analysis}"))

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone"""
    st.session_state.recording = True
    st.session_state.audio_data = None
    
    with st.spinner(f"🎤 Recording for {duration} seconds..."):
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        st.session_state.audio_data = audio
    
    st.session_state.recording = False
    return audio

def save_audio_to_wav(audio, sample_rate=16000):
    """Save audio data to temporary WAV file and return filename"""
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_filename = temp_file.name
    temp_file.close()
    
    # Write audio data to the temporary WAV file
    with wave.open(temp_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((audio * 32767).astype(np.int16))
    
    return temp_filename

def load_speech_recognizer():
    """Load speech recognizer"""
    if st.session_state.speech_recognizer is None:
        try:
            st.session_state.speech_recognizer = sr.Recognizer()
            # Adjust for ambient noise
            st.session_state.speech_recognizer.energy_threshold = 4000
            st.session_state.speech_recognizer.dynamic_energy_threshold = True
            st.session_state.speech_recognizer.pause_threshold = 0.8
            st.session_state.speech_recognizer.phrase_threshold = 0.3
            st.session_state.speech_recognizer.non_speaking_duration = 0.8
        except Exception as e:
            st.error(f"Failed to load speech recognizer: {str(e)}")
            return None
    return st.session_state.speech_recognizer

def speech_to_text_google(audio_file):
    """Convert speech to text using Google Speech Recognition (free)"""
    recognizer = load_speech_recognizer()
    if recognizer is None:
        return ""
    
    try:
        with st.spinner("🔊 Processing speech with Google Speech Recognition..."):
            # Load audio file
            with sr.AudioFile(audio_file) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Listen for audio
                audio_data = recognizer.record(source)
            
            # Try Google Speech Recognition first (free, but requires internet)
            try:
                text = recognizer.recognize_google(audio_data, language='en-US')
                return text.strip()
            except sr.UnknownValueError:
                # Try with different language if English fails
                try:
                    text = recognizer.recognize_google(audio_data)
                    return text.strip()
                except:
                    pass
            except sr.RequestError:
                # Fallback to offline recognition if no internet
                pass
            
            # Fallback to offline recognition (Sphinx)
            try:
                text = recognizer.recognize_sphinx(audio_data)
                return text.strip()
            except sr.UnknownValueError:
                return ""
            except sr.RequestError as e:
                st.error(f"Speech recognition error: {str(e)}")
                return ""
            
    except Exception as e:
        st.error(f"Error in speech recognition: {str(e)}")
        return ""
    finally:
        # Clean up temporary file
        try:
            os.unlink(audio_file)
        except:
            pass

def speech_to_text(audio_file):
    """Convert speech to text using Google Speech Recognition (enhanced version)"""
    return speech_to_text_google(audio_file)

def text_to_speech(text):
    """Convert text to speech using Hugging Face model"""
    if st.session_state.text_to_speech_model is None:
        try:
            st.session_state.text_to_speech_model = pipeline(
                "text-to-speech", 
                model="facebook/fastspeech2-en-ljspeech"
            )
        except Exception as e:
            st.error(f"Failed to load text-to-speech model: {str(e)}")
            return None, None
    
    with st.spinner("🔊 Generating speech..."):
        try:
            audio = st.session_state.text_to_speech_model(text)
            # Convert to numpy array and normalize
            audio_array = np.array(audio["audio"])
            audio_array = audio_array / np.max(np.abs(audio_array))
            return audio_array, audio["sampling_rate"]
        except Exception as e:
            st.error(f"Error in speech generation: {str(e)}")
            return None, None

def play_audio(audio_array, sample_rate):
    """Play audio using sounddevice"""
    if audio_array is not None and sample_rate is not None:
        try:
            sd.play(audio_array, samplerate=sample_rate)
            sd.wait()
        except Exception as e:
            st.error(f"Error playing audio: {str(e)}")

def voice_assistant_ui():
    """Voice assistant UI components"""
    with st.expander("🎙️ Voice Assistant", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎤 Start Recording (5s)"):
                if st.session_state.voice_assistant_enabled:
                    try:
                        audio = record_audio()
                        if audio is not None:
                            audio_file = save_audio_to_wav(audio)
                            user_text = speech_to_text(audio_file)
                            
                            if user_text and user_text.strip():
                                st.session_state.messages.append(HumanMessage(content=user_text))
                                with st.chat_message("human"):
                                    st.markdown(user_text)
                                
                                # Generate response
                                st.session_state.generating = True
                                with st.chat_message("ai"):
                                    message_placeholder = st.empty()
                                    full_response = ""
                                    
                                    # Prepare conversation context
                                    conversation_context = [{"role": "system", "content": st.session_state.system_prompt}]
                                    for msg in st.session_state.messages:
                                        if isinstance(msg, HumanMessage):
                                            conversation_context.append({"role": "user", "content": msg.content})
                                        elif isinstance(msg, AIMessage):
                                            conversation_context.append({"role": "assistant", "content": msg.content})
                                    
                                    # Stream response from Ollama
                                    try:
                                        for chunk in st.session_state.ollama_llm.stream(
                                            user_text,
                                            temperature=st.session_state.get("temperature", 0.7),
                                            context=conversation_context
                                        ):
                                            if st.session_state.stop_generation:
                                                full_response += " [Generation stopped by user]"
                                                break
                                            full_response += chunk
                                            message_placeholder.markdown(full_response + "▌")
                                    except Exception as e:
                                        full_response = f"Error: {str(e)}"
                                    
                                    message_placeholder.markdown(full_response)
                                    ai_message = AIMessage(content=full_response)
                                    st.session_state.messages.append(ai_message)
                                    
                                    # Convert response to speech
                                    if st.session_state.voice_assistant_enabled and full_response:
                                        audio_array, sample_rate = text_to_speech(full_response)
                                        play_audio(audio_array, sample_rate)
                                
                                st.session_state.generating = False
                                st.session_state.response_complete = True
                                st.rerun()
                            else:
                                st.warning("⚠️ No speech detected. Please try speaking more clearly or check your microphone.")
                    except Exception as e:
                        st.error(f"Error in voice assistant: {str(e)}")
                        st.session_state.generating = False
                        st.session_state.response_complete = True
                else:
                    st.warning("⚠️ Please enable voice responses first!")
        
        with col2:
            voice_enabled = st.checkbox(
                "Enable Voice Responses",
                value=st.session_state.voice_assistant_enabled,
                key="voice_enabled_checkbox",
                help="Enable text-to-speech for AI responses"
            )
            if voice_enabled != st.session_state.voice_assistant_enabled:
                st.session_state.voice_assistant_enabled = voice_enabled
                st.rerun()

def main():
    st.set_page_config(page_title="Strongest Rexa Bot", page_icon="🤖", layout="wide")
    
    st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        background-color: white !important;
        color: black !important;
    }
    
    /* Chat container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 6rem;
        max-width: 100%;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        max-width: 80%;
        word-wrap: break-word;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Human message styling */
    .stChatMessage[data-testid="chat-human-message"] {
        background-color: #f0f2f6;
        margin-left: auto;
        margin-right: 10px;
        color: black;
    }
    
    /* AI message styling */
    .stChatMessage[data-testid="chat-ai-message"] {
        background-color: #ffffff;
        margin-left: 10px;
        color: black;
        border: 1px solid #e1e4e8;
    }
    
    /* Chat input styling */
    .stChatInput {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 1rem;
        border-top: 1px solid #e1e4e8;
        z-index: 100;
    }
    
    /* Chat input container */
    .stChatInputContainer {
        max-width: 800px;
        margin: 0 auto;
        padding-bottom: 1rem;
    }
    
    /* Chat title styling */
    .chat-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: black;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        color: black;
    }
    
    /* Voice Assistant Styling */
    .stExpander {
        background-color: #e6f7ff !important;
        border: 1px solid #b3e0ff !important;
        border-radius: 8px !important;
        margin-bottom: 1rem !important;
    }
    
    .stExpander > div:first-child {
        background-color: #e6f7ff !important;
        color: #0066cc !important;
        font-weight: bold !important;
    }
    
    .stExpander > div:first-child:hover {
        background-color: #d0ebff !important;
    }
    
    /* Voice Assistant Buttons */
    .voice-assistant .stButton > button {
        background-color: #007bff !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
    }
    
    .voice-assistant .stButton > button:hover {
        background-color: #0069d9 !important;
    }
    
    /* Disabled chat input */
    .disabled-chat-input {
        opacity: 0.7;
        pointer-events: none;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .stChatMessage {
            max-width: 90%;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    initialize_session_state()
    
    # Hidden sidebar for settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Gemini API Key Input
        st.subheader("🔑 AI Vision Setup")
        gemini_key = st.text_input(
            "Gemini API Key (for advanced image analysis)",
            type="password",
            value=st.session_state.gemini_api_key,
            help="Get your free API key from: https://makersuite.google.com/app/apikey"
        )
        
        if gemini_key != st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = gemini_key
        
        if st.session_state.gemini_api_key:
            st.success("✅ Advanced AI image analysis enabled!")
        else:
            st.info("ℹ️ Add API key for detailed image understanding")
        
        st.divider()
        
        # Model Selection
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
        
        st.session_state.system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.system_prompt,
            help="Guides the model's behavior throughout the conversation"
        )
        
        st.divider()
        
        # Image Upload
        st.subheader("📷 Image Upload")
        uploaded_file = st.file_uploader(
            "Upload an image for AI analysis", 
            type=["jpg", "jpeg", "png", "gif", "bmp", "webp"],
            accept_multiple_files=False,
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state.uploaded_images = [image]
            st.session_state.image_analyzed = False
            st.success("📸 Image uploaded successfully!")
            
            # Show thumbnail in sidebar
            st.image(image, caption="Uploaded Image", width=200)
        
        st.divider()
        
        # Control Buttons
        col1, col2 = st.columns(2)
        with col1:
            st.button("🔄 Clear Chat", on_click=reset_conversation)
        with col2:
            st.button("⏹️ Stop", on_click=stop_generation)
        
        st.markdown("---")
        st.markdown("💡 **Tips:**")
        st.markdown("• Press Shift+Enter to send messages")
        st.markdown("• Upload images for AI analysis")
        st.markdown("• Get Gemini API key for advanced features")
        st.markdown("• Voice powered by Google Speech Recognition")
        st.markdown("")
        st.markdown("**Powered by:**")
        st.markdown("• [Ollama](https://ollama.ai/)")
        st.markdown("• [Google Gemini](https://ai.google.dev/)")
        st.markdown("• [Google Speech Recognition](https://cloud.google.com/speech-to-text)")
        st.markdown("• [Streamlit](https://streamlit.io/)")

    # Main Chat Interface
    with st.container():
        st.markdown('<div class="chat-title">🤖 Rexa AI  </div>', unsafe_allow_html=True)
        
        # Voice Assistant UI
        voice_assistant_ui()
        
        # Display conversation history
        for message in st.session_state.messages:
            if isinstance(message, HumanMessage):
                with st.chat_message("human"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("ai"):
                    st.markdown(message.content)
        
        # Display and analyze uploaded images
        display_images()

    # Chat Input - Placed at the bottom
    if st.session_state.generating:
        # Show disabled input during generation
        st.markdown("""
        <div class="disabled-chat-input">
            <input class="stChatInput" placeholder="💬 Bot is responding... (type will be enabled when done)" disabled>
        </div>
        """, unsafe_allow_html=True)
    else:
        if prompt := st.chat_input("💬 Type your message here...", key="chat_input"):
            st.session_state.stop_generation = False
            st.session_state.generating = True
            st.session_state.response_complete = False
            
            human_message = HumanMessage(content=prompt)
            st.session_state.messages.append(human_message)
            
            with st.chat_message("human"):
                st.markdown(prompt)
            
            with st.chat_message("ai"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Prepare conversation context
                conversation_context = [{"role": "system", "content": st.session_state.system_prompt}]
                for msg in st.session_state.messages:
                    if isinstance(msg, HumanMessage):
                        conversation_context.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        conversation_context.append({"role": "assistant", "content": msg.content})
                
                # Stream response from Ollama
                try:
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
                except Exception as e:
                    full_response = f"Error: {str(e)}"
                
                message_placeholder.markdown(full_response)
                
                # PDF Download option
                if "pdf" in prompt.lower() or "download" in prompt.lower():
                    pdf_buffer = create_pdf(full_response)
                    st.download_button(
                        label="📄 Download as PDF",
                        data=pdf_buffer,
                        file_name="chat_response.pdf",
                        mime="application/pdf"
                    )
                
                # Convert response to speech if voice assistant is enabled
                if st.session_state.voice_assistant_enabled:
                    audio_array, sample_rate = text_to_speech(full_response)
                    play_audio(audio_array, sample_rate)
            
            ai_message = AIMessage(content=full_response)
            st.session_state.messages.append(ai_message)
            st.session_state.generating = False
            st.session_state.response_complete = True
            st.rerun()

if __name__ == "__main__":
    main()