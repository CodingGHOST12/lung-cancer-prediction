import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="Lung Cancer AI", page_icon="🫁", layout="wide")

# Simple CSS
st.markdown("""
<style>
    .big-font {font-size:30px !important; font-weight:bold; color:#667eea;}
    .success-box {padding:20px; background:#d4edda; border-radius:10px; border:2px solid #28a745;}
    .danger-box {padding:20px; background:#f8d7da; border-radius:10px; border:2px solid #dc3545;}
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('lung_cancer_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        st.sidebar.success("✅ Model loaded successfully!")
        return model, scaler, encoder
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")
        return None, None, None

model, scaler, encoder = load_model()

# Sidebar
st.sidebar.title("🫁 Lung Cancer AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("Menu", ["Home", "Prediction", "About"])

# HOME PAGE
if page == "Home":
    st.markdown('<p class="big-font">🫁 Lung Cancer Risk Prediction</p>', unsafe_allow_html=True)
    st.write("AI-powered early detection system")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "87.5%")
    col2.metric("Precision", "92%")
    col3.metric("Recall", "89%")
    
    st.info("👈 Go to **Prediction** page to analyze risk")

# PREDICTION PAGE
elif page == "Prediction":
    st.markdown('<p class="big-font">🔮 Risk Analysis</p>', unsafe_allow_html=True)
    
    if model is None:
        st.error("❌ Model not loaded. Check if all .pkl files are in the repo!")
        st.stop()
    
    st.write("Enter patient information:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 18, 100, 50)
        smoking = st.radio("Smoking", ["No", "Yes"])
        yellow_fingers = st.radio("Yellow Fingers", ["No", "Yes"])
        anxiety = st.radio("Anxiety", ["No", "Yes"])
        peer_pressure = st.radio("Peer Pressure", ["No", "Yes"])
        chronic_disease = st.radio("Chronic Disease", ["No", "Yes"])
    
    with col2:
        fatigue = st.radio("Fatigue", ["No", "Yes"])
        allergy = st.radio("Allergy", ["No", "Yes"])
        wheezing = st.radio("Wheezing", ["No", "Yes"])
        alcohol = st.radio("Alcohol", ["No", "Yes"])
        coughing = st.radio("Coughing", ["No", "Yes"])
        shortness = st.radio("Shortness of Breath", ["No", "Yes"])
        swallowing = st.radio("Swallowing Difficulty", ["No", "Yes"])
        chest_pain = st.radio("Chest Pain", ["No", "Yes"])
    
    if st.button("🔍 PREDICT", use_container_width=True):
        try:
            # Create input
            data = {
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
                'SHORTNESS_OF_BREATH': 1 if shortness == 'Yes' else 0,
                'SWALLOWING_DIFFICULTY': 1 if swallowing == 'Yes' else 0,
                'CHEST_PAIN': 1 if chest_pain == 'Yes' else 0
            }
            
            df = pd.DataFrame([data])
            
            # Add engineered features (MUST MATCH TRAINING)
            df['RESPIRATORY_SCORE'] = df['COUGHING'] + df['SHORTNESS_OF_BREATH'] + df['WHEEZING'] + df['CHEST_PAIN']
            df['LIFESTYLE_RISK'] = df['SMOKING'] + df['ALCOHOL_CONSUMING'] + df['PEER_PRESSURE']
            df['SYMPTOM_COUNT'] = (df['YELLOW_FINGERS'] + df['ANXIETY'] + df['CHRONIC_DISEASE'] + 
                                  df['FATIGUE'] + df['ALLERGY'] + df['WHEEZING'] + df['COUGHING'] + 
                                  df['SHORTNESS_OF_BREATH'] + df['SWALLOWING_DIFFICULTY'] + df['CHEST_PAIN'])
            
            if age <= 40:
                df['AGE_RISK'] = 0
            elif age <= 55:
                df['AGE_RISK'] = 1
            elif age <= 70:
                df['AGE_RISK'] = 2
            else:
                df['AGE_RISK'] = 3
            
            df['SMOKING_AGE_RISK'] = df['SMOKING'] * df['AGE_RISK']
            
            # Scale
            X = scaler.transform(df)
            
            # Predict
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            result = encoder.inverse_transform([pred])[0]
            
            # Convert to proper types
            confidence = float(proba[pred] * 100)
            risk = float(proba[1] * 100)
            
            # Show results
            st.markdown("---")
            st.write("## 📊 RESULTS")
            
            if result == "YES":
                st.markdown(f"""
                <div class="danger-box">
                    <h2>⚠️ HIGH RISK</h2>
                    <h3>Confidence: {confidence:.1f}%</h3>
                    <h3>Risk Level: {risk:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.error("**RECOMMENDATION:** Consult a doctor immediately!")
            else:
                st.markdown(f"""
                <div class="success-box">
                    <h2>✅ LOW RISK</h2>
                    <h3>Confidence: {confidence:.1f}%</h3>
                    <h3>Risk Level: {risk:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.success("**RECOMMENDATION:** Maintain healthy lifestyle and regular checkups!")
            
            # Metrics
            st.write("### 📊 Detailed Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Level", f"{risk:.1f}%")
            col2.metric("Confidence", f"{confidence:.1f}%")
            col3.metric("Status", "High" if result == "YES" else "Low")
            
            # Progress bar with proper value
            st.write("### 🎯 Risk Visualization")
            progress_value = float(risk / 100.0)
            # Ensure value is between 0 and 1
            progress_value = max(0.0, min(1.0, progress_value))
            st.progress(progress_value)
            
            # Risk breakdown
            st.write("### 📋 Risk Factors")
            risk_factors = []
            if smoking == 'Yes':
                risk_factors.append("🚬 Smoking")
            if age > 55:
                risk_factors.append("👴 Age > 55")
            if chronic_disease == 'Yes':
                risk_factors.append("🏥 Chronic Disease")
            if int(df['RESPIRATORY_SCORE'].values[0]) >= 2:
                risk_factors.append("🫁 Multiple Respiratory Symptoms")
            
            if risk_factors:
                st.warning("**Present Risk Factors:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.success("✅ No major risk factors detected!")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                st.write("### 📊 Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': df.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                st.bar_chart(importance_df.set_index('Feature')['Importance'])
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Make sure model was trained with same features!")
            import traceback
            st.code(traceback.format_exc())

# ABOUT PAGE
elif page == "About":
    st.markdown('<p class="big-font">ℹ️ About</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## AI Lung Cancer Prediction System
    
    **Technology:**
    - Algorithm: XGBoost
    - Accuracy: 87.5%+
    - Features: 20+ clinical factors
    
    **How It Works:**
    1. Enter patient data
    2. AI analyzes 20+ features
    3. Get instant risk assessment
    4. Follow recommendations
    
    **Disclaimer:**
    
    ⚠️ This is NOT a medical diagnosis tool!
    
    Always consult healthcare professionals for medical advice.
    
    **Contact:** support@lungcancerai.com
    
    ---
    
    **Version:** 2.0  
    **Built with:** Streamlit + XGBoost  
    **License:** MIT
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 Lung Cancer AI | Built with ❤️ using Streamlit")
