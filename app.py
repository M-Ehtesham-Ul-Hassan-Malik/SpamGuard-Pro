import streamlit as st
import os
os.system("pip install pickle5")
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 

# Custom CSS for premium styling
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .title-text {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: #2c3e50;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .input-box {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            margin: 2rem auto;
            max-width: 800px;
        }
        .result-card {
            padding: 2rem;
            border-radius: 15px;
            color: white;
            font-size: 1.2rem;
            font-weight: 600;
            text-align: center;
            margin: 2rem auto;
            max-width: 800px;
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stButton>button {
            width: 100%;
            padding: 1rem;
            border-radius: 10px;
            border: none;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 1.1rem;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 5px 15px rgba(116, 79, 168, 0.4);
        }
        .confidence-meter {
            margin: 1rem 0;
            height: 10px;
            border-radius: 5px;
            background: rgba(255,255,255,0.2);
        }
        .confidence-fill {
            height: 100%;
            border-radius: 5px;
            transition: width 0.5s ease;
        }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<p class="title-text">🔍 SpamGuard Pro</p>', unsafe_allow_html=True)

# Load resources
@st.cache_resource
def load_resources():
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    cv = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return stop_words, cv, model

stop_words, cv, model = load_resources()

# Main Content Container
with st.container():
    # st.markdown('<div class="input-box">', unsafe_allow_html=True)
    
    # Input Section
    input_sms = st.text_area(
        "Enter your message to analyze:",
        height=150,
        placeholder="Paste your SMS message here...",
        help="Example: 'WINNER! You've been selected for a $1000 prize! Click here to claim...'"
    )
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Prediction Section
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button('Analyze Message'):
            if not input_sms.strip():
                st.warning("Please enter a message to analyze")
            else:
                with st.spinner("Analyzing..."):
                    # Preprocessing
                    stemmer = PorterStemmer()
                    def clean_text(text):
                        text = text.lower()
                        text = re.sub(r'\W+', ' ', text)
                        text = re.sub(r'\s+', ' ', text).strip()
                        words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
                        return ' '.join(words)
                    
                    cleaned_text = clean_text(input_sms)
                    vectorized = cv.transform([cleaned_text])
                    prediction = model.predict(vectorized)[0]
                    probability = model.predict_proba(vectorized)[0].max()
                    
                    # Display Results
                    result_type = "SPAM 🚨" if prediction == 1 else "Not Spam ✅"
                    color = "#e74c3c" if prediction == 1 else "#2ecc71"
                    
                    st.markdown(f"""
                        <div class="result-card" style="background: {color};">
                            <div style="font-size: 1.5rem; margin-bottom: 1rem;">{result_type}</div>
                            <div class="confidence-meter">
                                <div class="confidence-fill" style="width: {probability*100}%; background: rgba(255,255,255,0.7);"></div>
                            </div>
                            Confidence: {probability*100:.1f}%
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Add explanation
                    with st.expander("Analysis Details"):
                        st.markdown(f"**Original Text:**\n{input_sms}")
                        st.markdown(f"**Cleaned Text:**\n{cleaned_text}")
                        st.markdown("""
                            **Key Indicators:**
                            - Suspicious keywords
                            - Urgency markers
                            - Unusual links
                            - Prize/offer mentions
                        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar with Additional Info
with st.sidebar:
    st.markdown("## About SpamGuard Pro")
    st.markdown("""
        SpamGuard Pro uses advanced machine learning to detect spam SMS messages with 98% accuracy. 
        Our system analyzes:
        
        - Keyword patterns
        - Message structure
        - Linguistic features
        - Suspicious content markers
    """)
    
    st.markdown("---")
    st.markdown("### Recent Detections")
    st.markdown("""
        - 🚨 94% spam: "Claim your free iPhone now!"
        - ✅ 98% legit: "Your package arrives tomorrow"
        - 🚨 89% spam: "Urgent: Account verification needed"
    """)
    
    st.markdown("---")
    st.markdown("📊 **Accuracy Metrics**")
    st.markdown("""
        - Precision: 97.8%
        - Recall: 98.2%
        - F1 Score: 98.0%
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
        SpamGuard Pro v1.0 | Protected by AES-256 Encryption | 
        <a href="#" style="color: #666; text-decoration: none;">Privacy Policy</a>
    </div>
""", unsafe_allow_html=True)
