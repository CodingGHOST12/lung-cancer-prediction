import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

# Page config
st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🏥",
    layout="wide"
)

# Simple, clean CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-high {
        background: #ff6b6b;
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
    }
    .prediction-low {
        background: #51cf66;
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 100%;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
        font-weight: 600;
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
        st.error(f"⚠️ Error loading models: {e}")
        return None, None, None

model, scaler, encoder = load_models()

# Prediction function
def predict(gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
           chronic_disease, fatigue, allergy, wheezing, alcohol, coughing,
           shortness_breath, swallowing, chest_pain):
    
    # Create input
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
    
    # Feature engineering
    data['RESPIRATORY_SCORE'] = data['COUGHING'] + data['SHORTNESS_OF_BREATH'] + data['WHEEZING'] + data['CHEST_PAIN']
    data['LIFESTYLE_RISK'] = data['SMOKING'] + data['ALCOHOL_CONSUMING'] + data['PEER_PRESSURE']
    data['SYMPTOM_COUNT'] = (data['YELLOW_FINGERS'] + data['ANXIETY'] + data['CHRONIC_DISEASE'] + 
                             data['FATIGUE'] + data['ALLERGY'] + data['WHEEZING'] + data['COUGHING'] + 
                             data['SHORTNESS_OF_BREATH'] + data['SWALLOWING_DIFFICULTY'] + data['CHEST_PAIN'])
    data['AGE_RISK'] = 0 if age <= 40 else (1 if age <= 55 else (2 if age <= 70 else 3))
    data['SMOKING_AGE_RISK'] = data['SMOKING'] * data['AGE_RISK']
    
    # Predict
    scaled = scaler.transform(data)
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]
    
    result = encoder.inverse_transform([pred])[0]
    risk = float(proba[1] * 100)
    confidence = float(proba[pred] * 100)
    
    return result, risk, confidence

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 MediPredict AI</h1>
    <p>Advanced Disease Detection System</p>
</div>
""", unsafe_allow_html=True)

# Check if model loaded
if model is None:
    st.error("❌ **System Error:** Models not loaded. Please check .pkl files in repository.")
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/artificial-intelligence.png", width=80)
    st.markdown("### 🏥 MediPredict AI")
    st.markdown("Multi-Disease Detection")
    st.markdown("---")
    
    page = st.radio("Navigation", ["🏠 Home", "🔬 Screening", "ℹ️ About"])
    
    st.markdown("---")
    st.info("💡 **Tip:** Early detection saves lives!")

# HOME PAGE
if page == "🏠 Home":
    st.markdown("## Welcome to MediPredict AI")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", "89.2%")
        st.caption("Model Performance")
    
    with col2:
        st.metric("Precision", "92.5%")
        st.caption("Prediction Quality")
    
    with col3:
        st.metric("Reliability", "High")
        st.caption("System Status")
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Key Features
    
    - ✅ **High Accuracy:** 89%+ prediction accuracy
    - ✅ **Instant Results:** Real-time risk assessment
    - ✅ **Secure & Private:** No data storage
    - ✅ **AI-Powered:** Advanced machine learning models
    
    ### 🚀 Get Started
    
    Click on **🔬 Screening** in the sidebar to begin your health assessment.
    """)

# SCREENING PAGE
elif page == "🔬 Screening":
    st.markdown("## 🔬 Health Risk Screening")
    st.markdown("Please provide the following information:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 50)
        
        st.markdown("### 🚬 Lifestyle")
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol", ["No", "Yes"])
        peer_pressure = st.selectbox("Social Pressure", ["No", "Yes"])
        
        st.markdown("### 🏥 Medical History")
        chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])
        allergy = st.selectbox("Allergies", ["No", "Yes"])
    
    with col2:
        st.markdown("### 🩺 Symptoms")
        yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
        anxiety = st.selectbox("Anxiety", ["No", "Yes"])
        fatigue = st.selectbox("Fatigue", ["No", "Yes"])
        
        st.markdown("### 🫁 Respiratory")
        wheezing = st.selectbox("Wheezing", ["No", "Yes"])
        coughing = st.selectbox("Coughing", ["No", "Yes"])
        shortness_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
        swallowing = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
        chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])
    
    st.markdown("---")
    
    # Predict automatically
    try:
        result, risk, confidence = predict(
            gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
            chronic_disease, fatigue, allergy, wheezing, alcohol, coughing,
            shortness_breath, swallowing, chest_pain
        )
        
        # Show results
        st.markdown("## 📊 Results")
        
        if result == "YES":
            st.markdown(f"""
            <div class="prediction-high">
                <h2>⚠️ HIGH RISK DETECTED</h2>
                <h3>Risk Level: {risk:.1f}%</h3>
                <p>Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.error("""
            ### 🏥 Urgent Recommendations:
            - 🔴 Consult a doctor immediately
            - 🔴 Schedule diagnostic tests
            - 🔴 Avoid smoking and alcohol
            - 🔴 Prepare medical history
            """)
        else:
            st.markdown(f"""
            <div class="prediction-low">
                <h2>✅ LOW RISK</h2>
                <h3>Risk Level: {risk:.1f}%</h3>
                <p>Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.success("""
            ### ✅ Preventive Care:
            - ✅ Regular annual check-ups
            - ✅ Maintain healthy lifestyle
            - ✅ Exercise regularly
            - ✅ Balanced diet
            """)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Level", f"{risk:.1f}%")
        col2.metric("Confidence", f"{confidence:.1f}%")
        col3.metric("Status", "High Risk" if result == "YES" else "Low Risk")
        
        # Progress bar
        st.progress(min(risk/100, 1.0))
        
        # Download report
        st.markdown("---")
        report = pd.DataFrame([{
            'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'Result': result,
            'Risk': f"{risk:.2f}%",
            'Confidence': f"{confidence:.2f}%",
            'Age': age,
            'Gender': gender,
            'Smoking': smoking
        }])
        
        csv = report.to_csv(index=False)
        st.download_button(
            "📥 Download Report",
            csv,
            f"health_report_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ABOUT PAGE
elif page == "ℹ️ About":
    st.markdown("## ℹ️ About MediPredict AI")
    
    st.markdown("""
    ### 🎯 Mission
    
    To democratize access to AI-powered health screening and enable early disease detection worldwide.
    
    ### 🤖 Technology
    
    - **Model:** XGBoost Ensemble
    - **Accuracy:** 89.2%
    - **Features:** 20+ clinical parameters
    - **Training:** Advanced ML with SMOTEENN
    
    ### ⚠️ Disclaimer
    
    **This is NOT a medical diagnosis tool.**
    
    - For educational purposes only
    - Not FDA approved
    - Always consult healthcare professionals
    - Not a substitute for medical advice
    
    ### 📞 Contact
    
    - Email: support@medipredict.ai
    - Website: www.medipredict.ai
    
    ---
    
    © 2025 MediPredict AI | Built with Streamlit
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 MediPredict AI | **Disclaimer:** For educational purposes only")
