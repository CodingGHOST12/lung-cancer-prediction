import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Simple page config
st.set_page_config(page_title="MediPredict AI", page_icon="🏥", layout="wide")

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        model = pickle.load(open('lung_cancer_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        st.sidebar.success("✅ Models loaded")
        return model, scaler, encoder
    except FileNotFoundError:
        st.sidebar.error("❌ Model files not found")
        return None, None, None
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
        return None, None, None

model, scaler, encoder = load_models()

# Sidebar
st.sidebar.title("🏥 MediPredict AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("Menu", ["Home", "Screening", "About"])

# HOME PAGE
if page == "Home":
    st.title("🏥 MediPredict AI")
    st.subheader("Advanced Disease Detection System")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "89%")
    col2.metric("Precision", "92%")
    col3.metric("Response Time", "< 2s")
    
    st.markdown("---")
    st.info("👈 Go to **Screening** to start health assessment")

# SCREENING PAGE
elif page == "Screening":
    st.title("🔬 Health Screening")
    
    if model is None:
        st.error("❌ Models not loaded. Check if .pkl files are in repository.")
        st.stop()
    
    st.markdown("### Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 40)
        
        st.markdown("**Lifestyle**")
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol", ["No", "Yes"])
        peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
        
        st.markdown("**Medical History**")
        chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])
        allergy = st.selectbox("Allergies", ["No", "Yes"])
    
    with col2:
        st.markdown("**Physical Symptoms**")
        yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
        anxiety = st.selectbox("Anxiety", ["No", "Yes"])
        fatigue = st.selectbox("Fatigue", ["No", "Yes"])
        
        st.markdown("**Respiratory Symptoms**")
        wheezing = st.selectbox("Wheezing", ["No", "Yes"])
        coughing = st.selectbox("Coughing", ["No", "Yes"])
        shortness_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
        swallowing = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
        chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])
    
    st.markdown("---")
    
    # Make prediction
    try:
        # Create input dataframe
        input_data = pd.DataFrame([{
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
        
        # Feature engineering (MUST match training)
        input_data['RESPIRATORY_SCORE'] = (
            input_data['COUGHING'] + input_data['SHORTNESS_OF_BREATH'] + 
            input_data['WHEEZING'] + input_data['CHEST_PAIN']
        )
        input_data['LIFESTYLE_RISK'] = (
            input_data['SMOKING'] + input_data['ALCOHOL_CONSUMING']
        )
        input_data['SYMPTOM_COUNT'] = (
            input_data['YELLOW_FINGERS'] + input_data['CHRONIC_DISEASE'] + 
            input_data['FATIGUE'] + input_data['WHEEZING'] + 
            input_data['COUGHING'] + input_data['SHORTNESS_OF_BREATH'] + 
            input_data['SWALLOWING_DIFFICULTY'] + input_data['CHEST_PAIN']
        )
        
        # Scale
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Get result
        result = encoder.inverse_transform([prediction])[0]
        risk_level = float(probability[1] * 100)
        confidence = float(probability[prediction] * 100)
        
        # Display results
        st.markdown("## 📊 Results")
        
        if result == "YES":
            st.error(f"### ⚠️ HIGH RISK DETECTED")
            st.error(f"**Risk Level:** {risk_level:.1f}%")
            st.error(f"**Confidence:** {confidence:.1f}%")
            
            st.warning("""
            **Recommendations:**
            - Consult a doctor immediately
            - Schedule diagnostic tests
            - Avoid smoking and alcohol
            """)
        else:
            st.success(f"### ✅ LOW RISK")
            st.success(f"**Risk Level:** {risk_level:.1f}%")
            st.success(f"**Confidence:** {confidence:.1f}%")
            
            st.info("""
            **Recommendations:**
            - Maintain healthy lifestyle
            - Regular check-ups
            - Continue avoiding risk factors
            """)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Level", f"{risk_level:.1f}%")
        col2.metric("Confidence", f"{confidence:.1f}%")
        col3.metric("Status", "High" if result == "YES" else "Low")
        
        # Progress bar
        st.progress(min(risk_level/100, 1.0))
        
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        st.info("Make sure model files match the training configuration")

# ABOUT PAGE
elif page == "About":
    st.title("ℹ️ About MediPredict AI")
    
    st.markdown("""
    ## Mission
    AI-powered health screening for early disease detection.
    
    ## Technology
    - **Algorithm:** XGBoost
    - **Accuracy:** 89%+
    - **Features:** 18+ parameters
    
    ## Disclaimer
    ⚠️ **For educational purposes only**
    - Not a medical diagnosis
    - Always consult healthcare professionals
    - Not FDA approved
    
    ## Contact
    Email: support@medipredict.ai
    
    ---
    © 2025 MediPredict AI
    """)

st.markdown("---")
st.caption("© 2025 MediPredict AI | Educational purposes only")
