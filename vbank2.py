import streamlit as st
import ollama
import time
from typing import Dict, List

# Initialize Ollama client
try:
    ollama_client = ollama.Client(host='http://localhost:11434')
except Exception as e:
    st.error(f"Failed to connect to Ollama: {e}")
    st.stop()

# Bank information database
BANK_INFO = {
    "About V Bank": {
        "description": "V Bank is a leading digital banking platform established in 2010. We serve over 5 million customers worldwide with innovative financial solutions.",
        "key_points": [
            "Founded: 2010",
            "Customers: 5M+",
            "Digital-first approach",
            "Global presence in 15 countries"
        ]
    },
    "Services offered": {
        "description": "V Bank offers a comprehensive range of financial services:",
        "services": [
            "Personal Banking: Savings, Checking, CDs",
            "Business Banking: Merchant services, Business loans",
            "Loans: Personal, Auto, Mortgage, Student",
            "Investments: Retirement accounts, Brokerage services",
            "Digital Wallet: V Pay contactless payments"
        ]
    },
    "Location": {
        "description": "V Bank operates primarily online with some physical locations:",
        "locations": [
            "Headquarters: 123 Financial District, New York, NY",
            "Regional Centers: London, Singapore, Dubai",
            "ATMs: 10,000+ worldwide (Find nearest on our app)"
        ]
    },
    "Security measures": {
        "description": "Your security is our top priority with:",
        "measures": [
            "256-bit encryption for all transactions",
            "Biometric authentication (Face ID, Fingerprint)",
            "Real-time fraud monitoring",
            "Zero-liability protection",
            "24/7 account freezing capability"
        ]
    },
    "Loans": {
        "description": "V Bank offers competitive loan products:",
        "loan_types": [
            "Personal Loans: 5.99%-15.99% APR (up to $100,000)",
            "Mortgages: Fixed rates from 3.99% (15-30 year terms)",
            "Auto Loans: Starting at 2.99% APR for qualified buyers",
            "Student Loans: Refinancing available from 2.99%"
        ],
        "requirements": "Minimum credit score of 650 for most loan products"
    }
}

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize bank info context
if "bank_context" not in st.session_state:
    st.session_state.bank_context = """
    You are V-Bot, the AI assistant for V Bank. Your role is to provide accurate information about V Bank's services and products.
    Always be polite, professional, and concise in your responses. If you don't know an answer, direct the customer to call our 24/7 support line at 1-800-VBANK.
    
    Key information about V Bank:
    - Digital bank founded in 2010
    - Offers personal banking, business accounts, loans, and investment services
    - Strong focus on security with biometric authentication
    - Competitive loan rates with quick approval process
    - Mobile app available on iOS and Android
    
    Current promotions:
    - 2.5% APY on savings accounts for new customers
    - $200 bonus for opening a checking account with direct deposit
    - 0% APR balance transfers for 18 months
    """

# Function to generate response from Ollama
def generate_response(prompt: str) -> str:
    try:
        # Combine with bank context
        full_prompt = f"{st.session_state.bank_context}\n\nCustomer question: {prompt}"
        
        response = ollama_client.generate(
            model='llama3:instruct',
            prompt=full_prompt,
            options={'temperature': 0.3}
        )
        return response['response']
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}. Please try again later."

# Function to handle predefined queries
def handle_predefined_query(query: str) -> str:
    if query in BANK_INFO:
        info = BANK_INFO[query]
        response = f"**{query}**\n\n{info['description']}\n\n"
        
        for key in info:
            if key not in ['description']:
                response += f"**{key.replace('_', ' ').title()}**:\n"
                if isinstance(info[key], list):
                    for item in info[key]:
                        response += f"- {item}\n"
                else:
                    response += f"{info[key]}\n"
        return response
    return "I couldn't find information on that topic. Please try another query."

# Streamlit app layout
def main():
    st.set_page_config(page_title="V Bank - Digital Banking", page_icon="🏦", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
        .stChatFloatingInputContainer { bottom: 20px; }
        .stButton button { background-color: #4CAF50; color: white; }
        .bank-header { background-color: #0056b3; padding: 20px; border-radius: 10px; color: white; }
        .feature-card { padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); margin: 10px 0; }
        .chat-container { max-height: 500px; overflow-y: auto; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="bank-header">
        <h1 style="margin:0;">V Bank</h1>
        <p style="margin:0;">Digital Banking for the Modern World</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("## Welcome to V Bank")
        st.markdown("""
        <div class="feature-card">
            <h3>Your Digital Banking Partner</h3>
            <p>Experience banking reimagined with our cutting-edge digital platform offering:</p>
            <ul>
                <li>24/7 Account Access</li>
                <li>Instant Money Transfers</li>
                <li>AI-Powered Financial Insights</li>
                <li>Competitive Interest Rates</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>Quick Links</h3>
            <div style="display: flex; gap: 10px; margin-top: 10px;">
                <button onclick="window.location.href='#account'" style="padding: 8px 12px; border: none; border-radius: 5px; background-color: #4CAF50; color: white;">Open Account</button>
                <button onclick="window.location.href='#loans'" style="padding: 8px 12px; border: none; border-radius: 5px; background-color: #2196F3; color: white;">Apply for Loan</button>
                <button onclick="window.location.href='#contact'" style="padding: 8px 12px; border: none; border-radius: 5px; background-color: #ff9800; color: white;">Contact Us</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Predefined query buttons
        st.markdown("### Common Questions")
        cols = st.columns(2)
        with cols[0]:
            if st.button("About V Bank"):
                response = handle_predefined_query("About V Bank")
                st.session_state.messages.append({"role": "user", "content": "Tell me about V Bank"})
                st.session_state.messages.append({"role": "assistant", "content": response})
            if st.button("Our Locations"):
                response = handle_predefined_query("Location")
                st.session_state.messages.append({"role": "user", "content": "Where are your locations?"})
                st.session_state.messages.append({"role": "assistant", "content": response})
        with cols[1]:
            if st.button("Loan Options"):
                response = handle_predefined_query("Loans")
                st.session_state.messages.append({"role": "user", "content": "What loan options do you offer?"})
                st.session_state.messages.append({"role": "assistant", "content": response})
            if st.button("Security Features"):
                response = handle_predefined_query("Security measures")
                st.session_state.messages.append({"role": "user", "content": "What security measures do you have?"})
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.markdown("## V-Bot Assistant")
        st.markdown("Our AI assistant is here to help with your banking questions 24/7.")
        
        # Chat container
        with st.container(height=400, border=True):
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about V Bank..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Check if this is a predefined query
                predefined_response = None
                for key in BANK_INFO.keys():
                    if key.lower() in prompt.lower():
                        predefined_response = handle_predefined_query(key)
                        break
                
                if predefined_response:
                    full_response = predefined_response
                else:
                    # Generate streaming response
                    try:
                        response = ollama_client.generate(
                            model='llama3:instruct',
                            prompt=f"{st.session_state.bank_context}\n\nCustomer question: {prompt}",
                            stream=True,
                            options={'temperature': 0.3}
                        )
                        
                        for chunk in response:
                            if 'response' in chunk:
                                full_response += chunk['response']
                                message_placeholder.markdown(full_response + "▌")
                        full_response = full_response.strip()
                    except Exception as e:
                        full_response = f"Sorry, I'm having trouble connecting to our service. Error: {str(e)}"
                
                message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <p>© 2023 V Bank. All rights reserved.</p>
        <p style="font-size: small;">Member FDIC. Equal Housing Lender.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()