import streamlit as st
from streamlit_chat import message
import ollama
import time

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'bank_info' not in st.session_state:
    st.session_state.bank_info = {
        "about": "V Bank is a leading digital bank established in 2010, offering innovative financial solutions with a customer-first approach.",
        "services": "We offer savings accounts, current accounts, loans, credit cards, and investment products with competitive interest rates.",
        "locations": "V Bank operates entirely online with headquarters in New York and customer support centers in London, Singapore, and Dubai.",
        "security": "We use 256-bit encryption, multi-factor authentication, and real-time fraud monitoring to keep your accounts secure."
    }

# Custom CSS for styling
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .bank-header {
            background-color: #0056b3;
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .chat-container {
            border: 1px solid #ddd;
            border-radius: 0.5rem;
            padding: 1rem;
            background-color: white;
            height: 400px;
            overflow-y: auto;
            margin-bottom: 1rem;
        }
        .info-buttons {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        .stButton button {
            background-color: #0056b3;
            color: white;
        }
        .stTextInput input {
            border: 1px solid #0056b3;
        }
    </style>
""", unsafe_allow_html=True)

# Bank information buttons
bank_info_buttons = {
    "About V Bank": "about",
    "Our Services": "services",
    "Bank Locations": "locations",
    "Security Measures": "security"
}

# Function to generate response from Ollama
def generate_response(prompt):
    try:
        response = ollama.chat(
            model='llama2',
            messages=[{'role': 'user', 'content': prompt}],
            stream=False
        )
        return response['message']['content']
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# Function to handle bank info queries
def handle_bank_info_query(topic):
    info = st.session_state.bank_info.get(topic, "I don't have information on that topic.")
    prompt = f"""You are a helpful customer service chatbot for V Bank. 
    A customer has asked about {topic}. Here is the information you should provide: {info}.
    Provide a concise and friendly response."""
    
    return generate_response(prompt)

# Main app
def main():
    # Header
    st.markdown("<div class='bank-header'><h1>V Bank - Digital Banking Solutions</h1></div>", unsafe_allow_html=True)
    
    # Sidebar with account info
    st.sidebar.title("My Account")
    st.sidebar.subheader("Welcome, Customer!")
    st.sidebar.write("Account Number: ****1234")
    st.sidebar.write("Balance: $12,345.67")
    st.sidebar.button("View Transactions")
    st.sidebar.button("Transfer Money")
    st.sidebar.button("Pay Bills")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Banking Services")
        st.write("""
            - Open a new account in minutes
            - Apply for loans with competitive rates
            - Manage your investments
            - 24/7 customer support
        """)
        
        st.subheader("Quick Actions")
        st.button("Deposit Check")
        st.button("Send Money")
        st.button("View Statements")
    
    with col2:
        st.subheader("Rates & Offers")
        st.write("""
            - Savings APR: 3.25%
            - Mortgage Rate: 5.75%
            - Credit Card: 0% intro APR
        """)
        
        st.subheader("Contact Us")
        st.write("""
            Phone: 1-800-VBANK
            Email: support@vbank.com
            Chat: Available 24/7
        """)
    
    # Chatbot section
    st.markdown("---")
    st.subheader("V Bank Assistant")
    
    # Info buttons
    st.write("Quick information:")
    cols = st.columns(4)
    for i, (btn_text, topic) in enumerate(bank_info_buttons.items()):
        with cols[i]:
            if st.button(btn_text):
                response = handle_bank_info_query(topic)
                st.session_state.chat_history.append(("user", f"Tell me about {btn_text.lower()}"))
                st.session_state.chat_history.append(("assistant", response))
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    for i, (sender, msg) in enumerate(st.session_state.chat_history):
        if sender == "user":
            message(msg, is_user=True, key=f"user_{i}")
        else:
            message(msg, key=f"assistant_{i}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # User input
    user_input = st.text_input("Ask me anything about V Bank:", key="user_input")
    
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        
        # Check if the question matches any predefined bank info
        response = None
        for btn_text, topic in bank_info_buttons.items():
            if topic in user_input.lower() or btn_text.lower() in user_input.lower():
                response = handle_bank_info_query(topic)
                break
        
        if not response:
            # General question
            prompt = f"""You are a helpful customer service chatbot for V Bank. 
            A customer has asked: {user_input}. 
            If this is related to banking, provide a helpful response. 
            If not, politely explain that you can only assist with banking queries.
            Keep responses concise and friendly."""
            
            response = generate_response(prompt)
        
        st.session_state.chat_history.append(("assistant", response))
        st.experimental_rerun()

if __name__ == "__main__":
    main()