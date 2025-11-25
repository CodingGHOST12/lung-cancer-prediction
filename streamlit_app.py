import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Lung Cancer Prediction",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
    }
    
    .prediction-box {
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
    }
    
    .high-risk {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        border: 3px solid #ff5252;
    }
    
    .low-risk {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
        border: 3px solid #2ecc71;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        border: none;
        width: 100%;
    }
    
    .info-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .risk-factor {
        background: #f8f9fa;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        model = pickle.load(open('lung_cancer_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

model, scaler, label_encoder = load_models()

# Initialize session state
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0

# Function to make prediction
def make_prediction(gender, age, smoking, yellow_fingers, anxiety, peer_pressure, 
                   chronic_disease, fatigue, allergy, wheezing, alcohol, coughing, 
                   shortness_breath, swallowing, chest_pain):
    
    # Prepare input
    input_dict = {
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
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # Feature engineering
    input_df['RESPIRATORY_SCORE'] = (
        input_df['COUGHING'] + input_df['SHORTNESS_OF_BREATH'] + 
        input_df['WHEEZING'] + input_df['CHEST_PAIN']
    )
    input_df['LIFESTYLE_RISK'] = (
        input_df['SMOKING'] + input_df['ALCOHOL_CONSUMING'] + 
        input_df['PEER_PRESSURE']
    )
    input_df['SYMPTOM_COUNT'] = (
        input_df['YELLOW_FINGERS'] + input_df['ANXIETY'] + 
        input_df['CHRONIC_DISEASE'] + input_df['FATIGUE'] + 
        input_df['ALLERGY'] + input_df['WHEEZING'] + 
        input_df['COUGHING'] + input_df['SHORTNESS_OF_BREATH'] + 
        input_df['SWALLOWING_DIFFICULTY'] + input_df['CHEST_PAIN']
    )
    
    # Age risk
    age_risk_val = 0 if age <= 40 else (1 if age <= 55 else (2 if age <= 70 else 3))
    input_df['AGE_RISK'] = age_risk_val
    input_df['SMOKING_AGE_RISK'] = input_df['SMOKING'] * input_df['AGE_RISK']
    
    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    result = label_encoder.inverse_transform([prediction])[0]
    confidence = float(probability[prediction] * 100)
    risk_level = float(probability[1] * 100)
    
    return result, confidence, risk_level, input_df

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/lungs.png", width=100)
    st.markdown("### 🫁 Lung Cancer AI")
    st.markdown("---")
    
    selected = st.selectbox(
        "Navigation",
        ["🏠 Home", "🔮 Prediction", "ℹ️ About"]
    )
    
    st.markdown("---")
    st.markdown("### 📈 Stats")
    st.metric("Total Predictions", st.session_state.total_predictions)
    st.metric("Model Accuracy", "87.5%")

# HOME PAGE
if selected == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>🫁 AI Lung Cancer Prediction</h1>
        <p>Advanced Machine Learning for Early Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h2 style="color: #667eea;">🎯 High Accuracy</h2>
            <p>Our AI model achieves <b>87.5%+ accuracy</b> using advanced XGBoost algorithm.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h2 style="color: #667eea;">⚡ Instant Results</h2>
            <p>Get immediate risk assessment. Analyzes <b>20+ clinical factors</b>.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <h2 style="color: #667eea;">🔒 Secure & Private</h2>
            <p>Your data is <b>never stored</b>. Complete privacy guaranteed.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "87.5%")
    col2.metric("Precision", "92%")
    col3.metric("Recall", "89%")
    col4.metric("F1-Score", "90.5%")

# PREDICTION PAGE
elif selected == "🔮 Prediction":
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Real-Time Risk Prediction</h1>
        <p>Enter patient information - Prediction updates automatically</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check model loaded
    if model is None:
        st.error("❌ Model not loaded! Check .pkl files.")
        st.stop()
    
    # Input form
    st.markdown("### 📋 Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 50)
        
        st.markdown("#### 🚬 Lifestyle")
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol", ["No", "Yes"])
        peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
        
        st.markdown("#### 🏥 Medical")
        chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])
        allergy = st.selectbox("Allergies", ["No", "Yes"])
    
    with col2:
        st.markdown("#### 🩺 Physical Symptoms")
        yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
        anxiety = st.selectbox("Anxiety", ["No", "Yes"])
        fatigue = st.selectbox("Fatigue", ["No", "Yes"])
        
        st.markdown("#### 🫁 Respiratory")
        wheezing = st.selectbox("Wheezing", ["No", "Yes"])
        coughing = st.selectbox("Cough", ["No", "Yes"])
        shortness_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
        swallowing = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
        chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])
    
    # Make prediction automatically
    try:
        result, confidence, risk_level, input_df = make_prediction(
            gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
            chronic_disease, fatigue, allergy, wheezing, alcohol, coughing,
            shortness_breath, swallowing, chest_pain
        )
        
        st.session_state.total_predictions += 1
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Live Prediction Results")
        
        # Result box
        risk_class = "high-risk" if result == "YES" else "low-risk"
        risk_emoji = "⚠️" if result == "YES" else "✅"
        risk_text = "HIGH RISK DETECTED" if result == "YES" else "LOW RISK DETECTED"
        
        st.markdown(f"""
        <div class="prediction-box {risk_class}">
            <h2>{risk_emoji} {risk_text}</h2>
            <p style="font-size: 1.5rem;">Confidence: {confidence:.1f}%</p>
            <p>Risk Probability: {risk_level:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        recommendation_type = "error" if result == "YES" else "success"
        recommendation_text = """
**URGENT - High Risk Detected:**
- 🔴 Consult oncologist immediately
- 🔴 Schedule diagnostic tests (CT scan)
- 🔴 Avoid smoking and secondhand smoke
- 🔴 Prepare medical history
- 🔴 Consider second opinion
""" if result == "YES" else """
**Maintain Healthy Lifestyle:**
- ✅ Regular annual check-ups
- ✅ Balanced diet with fruits/vegetables
- ✅ Exercise 30+ minutes daily
- ✅ Avoid smoking and limit alcohol
- ✅ Monitor any new symptoms
"""
        
        if result == "YES":
            st.error(recommendation_text)
        else:
            st.success(recommendation_text)
        
        # Metrics
        st.markdown("### 📊 Detailed Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Level", f"{risk_level:.1f}%")
        col2.metric("Confidence", f"{confidence:.1f}%")
        col3.metric("Status", "High Risk" if result == "YES" else "Low Risk")
        
        # Progress bar
        st.markdown("### 🎯 Risk Gauge")
        progress_val = float(risk_level / 100.0)
        progress_val = max(0.0, min(1.0, progress_val))
        st.progress(progress_val)
        
        # Risk factors
        st.markdown("### 📋 Risk Factor Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🚨 Present Risk Factors")
            risks = []
            if smoking == 'Yes':
                risks.append("🚬 Smoking")
            if age > 55:
                risks.append("👴 Age > 55")
            if chronic_disease == 'Yes':
                risks.append("🏥 Chronic Disease")
            if int(input_df['RESPIRATORY_SCORE'].values[0]) >= 2:
                risks.append("🫁 Multiple Respiratory Symptoms")
            if yellow_fingers == 'Yes':
                risks.append("✋ Yellow Fingers")
            if chest_pain == 'Yes':
                risks.append("💔 Chest Pain")
            
            if risks:
                for r in risks:
                    st.markdown(f"<div class='risk-factor'>{r}</div>", unsafe_allow_html=True)
            else:
                st.info("✅ No major risk factors")
        
        with col2:
            st.markdown("#### ✅ Protective Factors")
            protective = []
            if smoking == 'No':
                protective.append("✅ Non-smoker")
            if age < 40:
                protective.append("✅ Young age")
            if alcohol == 'No':
                protective.append("✅ No alcohol")
            if chronic_disease == 'No':
                protective.append("✅ No chronic disease")
            if int(input_df['RESPIRATORY_SCORE'].values[0]) == 0:
                protective.append("✅ No respiratory symptoms")
            
            if protective:
                for p in protective:
                    st.markdown(f"<div class='risk-factor'>{p}</div>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ Limited protective factors")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            st.markdown("### 📊 Feature Importance (Top 10)")
            importance_df = pd.DataFrame({
                'Feature': input_df.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            st.bar_chart(importance_df.set_index('Feature')['Importance'])
        
        # Download report
        st.markdown("---")
        st.markdown("### 📥 Download Report")
        
        report_data = {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Prediction': result,
            'Confidence': f"{confidence:.2f}%",
            'Risk_Level': f"{risk_level:.2f}%",
            'Age': age,
            'Gender': gender,
            'Smoking': smoking,
            'Respiratory_Score': int(input_df['RESPIRATORY_SCORE'].values[0]),
            'Symptom_Count': int(input_df['SYMPTOM_COUNT'].values[0])
        }
        
        report_df = pd.DataFrame([report_data])
        csv = report_df.to_csv(index=False)
        
        st.download_button(
            label="📄 Download CSV Report",
            data=csv,
            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

# ABOUT PAGE
elif selected == "ℹ️ About":
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ About</h1>
        <p>AI-powered lung cancer screening</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Mission
    
    Democratize access to advanced lung cancer screening through AI for early detection.
    
    ## 🤖 Technology
    
    - **Algorithm**: XGBoost
    - **Accuracy**: 87.5%+
    - **Features**: 20+ clinical factors
    - **Training**: SMOTEENN + hyperparameter tuning
    
    ## ⚠️ Disclaimer
    
    This is **NOT** a medical diagnosis tool!
    
    - ❌ Not a substitute for professional advice
    - ❌ Not FDA approved
    - ✅ For educational purposes only
    
    **Always consult healthcare professionals.**
    
    ## 📞 Contact
    
    - Email: support@lungcancerai.com
    - Website: www.lungcancerai.com
    
    ---
    © 2025 Lung Cancer AI
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 Lung Cancer AI | Built with Streamlit")
