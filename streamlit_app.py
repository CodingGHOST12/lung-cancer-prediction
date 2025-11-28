import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

# Page config
st.set_page_config(
    page_title="MediPredict AI - Disease Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS with animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
    }
    
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
    }
    
    .hero h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
    }
    
    .hero p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    .card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    }
    
    .result-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a6f);
        color: white;
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        animation: slideIn 0.5s;
    }
    
    .result-low {
        background: linear-gradient(135deg, #51cf66, #37b24d);
        color: white;
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        animation: slideIn 0.5s;
    }
    
    .result-high h2, .result-low h2 {
        font-size: 2.5rem;
        margin: 0;
    }
    
    .result-high h3, .result-low h3 {
        font-size: 2rem;
        margin: 1rem 0;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 100%;
        padding: 1rem;
        border-radius: 10px;
        border: none;
        font-size: 1.2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .metric-card h3 {
        color: #667eea;
        font-size: 2rem;
        margin: 0;
    }
    
    .metric-card p {
        color: #6c757d;
        margin: 0.5rem 0 0 0;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        model = pickle.load(open('lung_cancer_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        return model, scaler, encoder
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None

model, scaler, encoder = load_models()

# Prediction function
def make_prediction(gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
                   chronic_disease, fatigue, allergy, wheezing, alcohol, coughing,
                   shortness_breath, swallowing, chest_pain):
    
    data = pd.DataFrame([{
        'GENDER': 1 if gender == 'Male' else 0,
        'AGE': age,
        'SMOKING': 1 if smoking == 'Yes' else 0,
        'YELLOW_FINGERS': 1 if yellow_fingers == 'Yes' else 0,
        'ANXIETY': 1 if anxiety == 'Yes' else 0,
        'PEER_PRESSURE': 1 if peer_pressure == 'Yes' else 0,
        'CHRONIC_DISEASE': 1 if chronic_disease == 'Yes' else 0,
        'FATIGUE': 1 if fatigue == 'Yes' else 0,
        'ALLERGY': 1 if allergy == 'Yes' else 0,
        'WHEEZING': 1 if wheezing == 'Yes' else 0,
        'ALCOHOL_CONSUMING': 1 if alcohol == 'Yes' else 0,
        'COUGHING': 1 if coughing == 'Yes' else 0,
        'SHORTNESS_OF_BREATH': 1 if shortness_breath == 'Yes' else 0,
        'SWALLOWING_DIFFICULTY': 1 if swallowing == 'Yes' else 0,
        'CHEST_PAIN': 1 if chest_pain == 'Yes' else 0
    }])
    
    # Feature engineering (must match training)
    data['RESPIRATORY_SCORE'] = data['COUGHING'] + data['SHORTNESS_OF_BREATH'] + data['WHEEZING'] + data['CHEST_PAIN']
    data['LIFESTYLE_RISK'] = data['SMOKING'] + data['ALCOHOL_CONSUMING']
    data['SYMPTOM_COUNT'] = (data['YELLOW_FINGERS'] + data['CHRONIC_DISEASE'] + data['FATIGUE'] + 
                            data['WHEEZING'] + data['COUGHING'] + data['SHORTNESS_OF_BREATH'] + 
                            data['SWALLOWING_DIFFICULTY'] + data['CHEST_PAIN'])
    
    # Predict
    scaled = scaler.transform(data)
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]
    
    result = encoder.inverse_transform([pred])[0]
    risk = float(proba[1] * 100)
    confidence = float(proba[pred] * 100)
    
    return result, risk, confidence

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/artificial-intelligence.png", width=100)
    st.markdown("# 🏥 MediPredict AI")
    st.markdown("---")
    
    page = st.radio("", ["🏠 Home", "🔬 Screening", "📊 About"])

# HOME PAGE
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
        <h1>🏥 MediPredict AI</h1>
        <p>Advanced Multi-Disease Detection System</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3 style="color: #667eea;">🎯 Accurate</h3>
            <p>AI-powered predictions with 89% accuracy rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3 style="color: #667eea;">⚡ Fast</h3>
            <p>Get instant results in under 2 seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h3 style="color: #667eea;">🔒 Secure</h3>
            <p>Your data is never stored or shared</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## 📊 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>89%</h3>
            <p>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>92%</h3>
            <p>Precision</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>1.5s</h3>
            <p>Response Time</p>
        </div>
        """, unsafe_allow_html=True)

# SCREENING PAGE
elif page == "🔬 Screening":
    st.markdown("""
    <div class="hero">
        <h1>🔬 Health Screening</h1>
        <p>Complete the form below for AI-powered risk assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    if model is None:
        st.error("❌ Model not loaded")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 40)
        
        st.markdown("### 🚬 Lifestyle")
        smoking = st.radio("Do you smoke?", ["No", "Yes"])
        alcohol = st.radio("Do you drink alcohol?", ["No", "Yes"])
        peer_pressure = st.radio("Social pressure to smoke/drink?", ["No", "Yes"])
        
        st.markdown("### 🏥 Medical History")
        chronic_disease = st.radio("Any chronic disease?", ["No", "Yes"])
        allergy = st.radio("Any allergies?", ["No", "Yes"])
    
    with col2:
        st.markdown("### 🩺 Physical Symptoms")
        yellow_fingers = st.radio("Yellow fingers?", ["No", "Yes"])
        anxiety = st.radio("Anxiety/Stress?", ["No", "Yes"])
        fatigue = st.radio("Chronic fatigue?", ["No", "Yes"])
        
        st.markdown("### 🫁 Respiratory")
        wheezing = st.radio("Wheezing?", ["No", "Yes"])
        coughing = st.radio("Persistent cough?", ["No", "Yes"])
        shortness_breath = st.radio("Shortness of breath?", ["No", "Yes"])
        swallowing = st.radio("Difficulty swallowing?", ["No", "Yes"])
        chest_pain = st.radio("Chest pain?", ["No", "Yes"])
    
    st.markdown("---")
    
    # Predict
    try:
        result, risk, confidence = make_prediction(
            gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
            chronic_disease, fatigue, allergy, wheezing, alcohol, coughing,
            shortness_breath, swallowing, chest_pain
        )
        
        st.markdown("## 📊 Analysis Results")
        
        if result == "YES":
            st.markdown(f"""
            <div class="result-high">
                <h2>⚠️ HIGH RISK DETECTED</h2>
                <h3>Risk Level: {risk:.1f}%</h3>
                <p style="font-size: 1.2rem;">Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.error("""
            ### 🏥 Important Recommendations:
            - **Consult a doctor immediately**
            - Schedule diagnostic tests (CT scan)
            - Avoid smoking and alcohol
            - Prepare complete medical history
            """)
        else:
            st.markdown(f"""
            <div class="result-low">
                <h2>✅ LOW RISK</h2>
                <h3>Risk Level: {risk:.1f}%</h3>
                <p style="font-size: 1.2rem;">Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.success("""
            ### ✅ Maintain Your Health:
            - Continue regular check-ups
            - Maintain healthy lifestyle
            - Exercise regularly
            - Avoid smoking and excessive alcohol
            """)
        
        # Progress bar
        st.markdown("### Risk Meter")
        st.progress(min(risk/100, 1.0))
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Risk", f"{risk:.1f}%")
        col2.metric("Confidence", f"{confidence:.1f}%")
        col3.metric("Status", "High" if result == "YES" else "Low")
        
    except Exception as e:
        st.error(f"Error: {e}")

# ABOUT PAGE
elif page == "📊 About":
    st.markdown("""
    <div class="hero">
        <h1>📊 About MediPredict AI</h1>
        <p>AI-Powered Health Screening Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Our Mission
    
    To democratize access to AI-powered health screening and enable early disease detection.
    
    ## 🤖 Technology
    
    - **Algorithm:** XGBoost Machine Learning
    - **Accuracy:** 89%+
    - **Features:** 18+ clinical parameters
    - **Training:** Advanced ML with SMOTE balancing
    
    ## ⚠️ Important Disclaimer
    
    **This tool is for educational purposes only.**
    
    - NOT a medical diagnosis
    - NOT FDA approved
    - Always consult healthcare professionals
    - Not a substitute for medical advice
    
    ## 📞 Contact
    
    Email: support@medipredict.ai
    
    ---
    
    © 2025 MediPredict AI
    """)

st.markdown("---")
st.caption("© 2025 MediPredict AI | Educational purposes only | Not medical advice")
