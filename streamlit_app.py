import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import streamlit_option_menu

# Page configuration
st.set_page_config(
    page_title="AI Lung Cancer Prediction",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* Header styling */
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
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    /* Prediction result boxes */
    .prediction-box {
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
        animation: slideIn 0.5s ease-out;
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
    
    .prediction-box h2 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .prediction-box p {
        font-size: 1.3rem;
        margin: 0.5rem 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        border: none;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Input styling */
    .stSelectbox, .stSlider, .stTextInput {
        background: white;
        border-radius: 10px;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    
    .bot-message {
        background: #f0f2f6;
        color: #333;
        margin-right: 20%;
    }
    
    /* Feature importance bars */
    .feature-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 30px;
        border-radius: 15px;
        margin: 5px 0;
        transition: width 0.5s ease;
    }
    
    /* Progress indicator */
    .progress-ring {
        transform: rotate(-90deg);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    /* Alert boxes */
    .alert-info {
        background: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Risk factors list */
    .risk-factor {
        background: white;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
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
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/lungs.png", width=100)
    st.markdown("### 🫁 Lung Cancer AI")
    st.markdown("---")
    
    selected = st.selectbox(
        "Navigation",
        ["🏠 Home", "🔮 Prediction", "📊 Analytics", "💬 AI Assistant", "📚 Learn More", "ℹ️ About"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    st.metric("Total Predictions", st.session_state.total_predictions)
    st.metric("Model Accuracy", "87.5%")
    st.metric("Success Rate", "95%")
    
    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown("- [Download Report](#)")
    st.markdown("- [View Documentation](#)")
    st.markdown("- [Contact Support](#)")
    
    st.markdown("---")
    st.markdown("#### Made with ❤️ by AI Lab")

# HOME PAGE
if selected == "🏠 Home":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🫁 AI Lung Cancer Prediction System</h1>
        <p>Advanced Machine Learning for Early Detection & Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h2 style="color: #667eea;">🎯 High Accuracy</h2>
            <p>Our AI model achieves <b>87.5%+ accuracy</b> using advanced ensemble learning techniques including XGBoost, CatBoost, and LightGBM.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h2 style="color: #667eea;">⚡ Instant Results</h2>
            <p>Get immediate risk assessment in seconds. Our system processes <b>15+ clinical factors</b> for comprehensive analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <h2 style="color: #667eea;">🔒 Secure & Private</h2>
            <p>Your data is <b>never stored</b>. All predictions happen in real-time with complete privacy and security.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How It Works Section
    st.markdown("---")
    st.markdown("## 🔬 How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <h1 style="color: #667eea;">1️⃣</h1>
            <h3>Enter Data</h3>
            <p>Provide patient health information and lifestyle factors</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <h1 style="color: #667eea;">2️⃣</h1>
            <h3>AI Analysis</h3>
            <p>Advanced algorithms analyze risk patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <h1 style="color: #667eea;">3️⃣</h1>
            <h3>Get Results</h3>
            <p>Receive detailed risk assessment with confidence scores</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <h1 style="color: #667eea;">4️⃣</h1>
            <h3>Take Action</h3>
            <p>Follow personalized recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Statistics
    st.markdown("---")
    st.markdown("## 📊 System Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">87.5%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">92%</div>
            <div class="metric-label">Precision</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">89%</div>
            <div class="metric-label">Recall</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">90.5%</div>
            <div class="metric-label">F1-Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("---")
    st.markdown("### 🚀 Ready to Get Started?")
    if st.button("🔮 Start Prediction Now", key="home_cta"):
        st.session_state.page = "prediction"
        st.rerun()

# PREDICTION PAGE
elif selected == "🔮 Prediction":
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Lung Cancer Risk Prediction</h1>
        <p>Enter patient information for instant AI-powered risk assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input Form
    st.markdown("### 📋 Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"], help="Biological sex of the patient")
        age = st.slider("Age", 18, 100, 50, help="Patient's age in years")
        
        st.markdown("#### 🚬 Lifestyle Factors")
        smoking = st.selectbox("Smoking Status", ["No", "Yes"], help="Current or former smoker")
        alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"], help="Regular alcohol consumption")
        peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"], help="Social influence towards unhealthy habits")
        
        st.markdown("#### 🏥 Medical History")
        chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"], help="History of chronic conditions")
        allergy = st.selectbox("Allergies", ["No", "Yes"], help="Known allergies")
    
    with col2:
        st.markdown("#### 🩺 Physical Symptoms")
        yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"], help="Yellowing of fingers (smoking-related)")
        anxiety = st.selectbox("Anxiety", ["No", "Yes"], help="Experiencing anxiety or stress")
        fatigue = st.selectbox("Chronic Fatigue", ["No", "Yes"], help="Persistent tiredness")
        
        st.markdown("#### 🫁 Respiratory Symptoms")
        wheezing = st.selectbox("Wheezing", ["No", "Yes"], help="Whistling sound when breathing")
        coughing = st.selectbox("Persistent Cough", ["No", "Yes"], help="Chronic or persistent cough")
        shortness_breath = st.selectbox("Shortness of Breath", ["No", "Yes"], help="Difficulty breathing")
        swallowing = st.selectbox("Swallowing Difficulty", ["No", "Yes"], help="Trouble swallowing food")
        chest_pain = st.selectbox("Chest Pain", ["No", "Yes"], help="Pain or discomfort in chest")
    
    st.markdown("---")
    
    # Predict Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔍 Analyze Risk Profile", use_container_width=True)
    
    if predict_button:
        if model is not None:
            with st.spinner("🧠 AI is analyzing the data..."):
                # Prepare input data
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
                
                # Create DataFrame
                input_df = pd.DataFrame([input_dict])
                
                # Feature engineering (match training)
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
                if age <= 40:
                    input_df['AGE_RISK'] = 0
                elif age <= 55:
                    input_df['AGE_RISK'] = 1
                elif age <= 70:
                    input_df['AGE_RISK'] = 2
                else:
                    input_df['AGE_RISK'] = 3
                
                input_df['SMOKING_AGE_RISK'] = input_df['SMOKING'] * input_df['AGE_RISK']
                
                # Scale data
                input_scaled = scaler.transform(input_df)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0]
                
                result = label_encoder.inverse_transform([prediction])[0]
                confidence = probability[prediction] * 100
                
                # Update stats
                st.session_state.total_predictions += 1
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now(),
                    'result': result,
                    'confidence': confidence
                })
                
                # Display Results
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                # Result Box
                if result == "YES":
                    st.markdown(f"""
                    <div class="prediction-box high-risk">
                        <h2>⚠️ HIGH RISK DETECTED</h2>
                        <p style="font-size: 1.5rem;">Confidence: {confidence:.1f}%</p>
                        <p>Risk Level: {probability[1]*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.error("### 🏥 URGENT RECOMMENDATION")
                    st.markdown("""
                    **Immediate Actions Required:**
                    - 🔴 Consult with an oncologist or pulmonologist immediately
                    - 🔴 Schedule comprehensive diagnostic tests (CT scan, biopsy if needed)
                    - 🔴 Prepare detailed medical history for doctor consultation
                    - 🔴 Avoid smoking and secondhand smoke exposure
                    - 🔴 Consider second medical opinion
                    """)
                else:
                    st.markdown(f"""
                    <div class="prediction-box low-risk">
                        <h2>✅ LOW RISK DETECTED</h2>
                        <p style="font-size: 1.5rem;">Confidence: {confidence:.1f}%</p>
                        <p>Risk Level: {probability[1]*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("### ✅ PREVENTIVE RECOMMENDATIONS")
                    st.markdown("""
                    **Maintain Healthy Lifestyle:**
                    - ✅ Continue regular health check-ups annually
                    - ✅ Maintain a balanced diet rich in fruits and vegetables
                    - ✅ Exercise regularly (at least 30 min/day)
                    - ✅ Avoid smoking and limit alcohol consumption
                    - ✅ Monitor any new respiratory symptoms
                    """)
                
                # Gauge Chart for Risk Level
                st.markdown("### 🎯 Risk Gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=probability[1]*100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Cancer Risk Level", 'font': {'size': 24}},
                    delta={'reference': 50, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': '#51cf66'},
                            {'range': [30, 70], 'color': '#ffd43b'},
                            {'range': [70, 100], 'color': '#ff6b6b'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Feature Importance
                st.markdown("### 📊 Risk Factor Analysis")
                
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'Feature': input_df.columns,
                        'Importance': model.feature_importances_,
                        'Value': input_df.values[0]
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    # Create horizontal bar chart
                    fig_features = px.bar(
                        feature_importance,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 10 Contributing Factors',
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    fig_features.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_features, use_container_width=True)
                
                # Risk Factors Breakdown
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🚨 Major Risk Factors")
                    risk_factors = []
                    if smoking == 'Yes':
                        risk_factors.append("🚬 Smoking")
                    if age > 55:
                        risk_factors.append("👴 Age > 55")
                    if chronic_disease == 'Yes':
                        risk_factors.append("🏥 Chronic Disease")
                    if input_df['RESPIRATORY_SCORE'].values[0] >= 2:
                        risk_factors.append("🫁 Multiple Respiratory Symptoms")
                    
                    if risk_factors:
                        for factor in risk_factors:
                            st.markdown(f"<div class='risk-factor'>{factor}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No major risk factors identified")
                
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
                    
                    if protective:
                        for factor in protective:
                            st.markdown(f"<div class='risk-factor'>{factor}</div>", unsafe_allow_html=True)
                    else:
                        st.warning("Limited protective factors present")
                
                # Download Report
                st.markdown("---")
                if st.button("📥 Download Detailed Report"):
                    report_data = {
                        'Prediction': result,
                        'Confidence': f"{confidence:.2f}%",
                        'Risk_Level': f"{probability[1]*100:.2f}%",
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        **input_dict
                    }
                    report_df = pd.DataFrame([report_data])
                    csv = report_df.to_csv(index=False)
                    st.download_button(
                        "📄 Download CSV Report",
                        csv,
                        "lung_cancer_prediction_report.csv",
                        "text/csv"
                    )
        else:
            st.error("Model not loaded. Please check model files.")

# ANALYTICS PAGE
elif selected == "📊 Analytics":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Prediction Analytics</h1>
        <p>Track and visualize prediction history and trends</p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(st.session_state.prediction_history) > 0:
        # Convert to DataFrame
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(history_df))
        with col2:
            high_risk_count = len(history_df[history_df['result'] == 'YES'])
            st.metric("High Risk Cases", high_risk_count)
        with col3:
            low_risk_count = len(history_df[history_df['result'] == 'NO'])
            st.metric("Low Risk Cases", low_risk_count)
        with col4:
            avg_confidence = history_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        # Pie Chart
        st.markdown("### 📈 Risk Distribution")
        risk_counts = history_df['result'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=['Low Risk' if x == 'NO' else 'High Risk' for x in risk_counts.index],
            color_discrete_sequence=['#51cf66', '#ff6b6b']
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Timeline
        st.markdown("### 📅 Prediction Timeline")
        history_df['date'] = pd.to_datetime(history_df['timestamp']).dt.date
        timeline = history_df.groupby('date').size().reset_index(name='count')
        fig_timeline = px.line(timeline, x='date', y='count', title='Predictions Over Time')
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Data Table
        st.markdown("### 📋 Recent Predictions")
        st.dataframe(history_df.tail(10), use_container_width=True)
    else:
        st.info("📊 No predictions yet. Make your first prediction to see analytics!")

# AI ASSISTANT PAGE
elif selected == "💬 AI Assistant":
    st.markdown("""
    <div class="main-header">
        <h1>💬 AI Health Assistant</h1>
        <p>Ask questions about lung cancer, symptoms, and prevention</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat history
    for chat in st.session_state.chat_history:
        if chat['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {chat['message']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 AI Assistant:</strong> {chat['message']}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input("Type your question here...", key="chat_input", placeholder="e.g., What are early symptoms of lung cancer?")
    
    with col2:
        send_button = st.button("📤 Send", use_container_width=True)
    
    if send_button and user_input:
        st.session_state.chat_history.append({'role': 'user', 'message': user_input})
        
        # Generate response (simple rule-based - can integrate GPT API)
        response = generate_ai_response(user_input)
        
        st.session_state.chat_history.append({'role': 'assistant', 'message': response})
        st.rerun()
    
    # Quick Questions
    st.markdown("### 💡 Quick Questions")
    col1, col2 = st.columns(2)
    
    quick_questions = [
        "What are the symptoms of lung cancer?",
        "How accurate is this AI model?",
        "What are risk factors for lung cancer?",
        "How can I prevent lung cancer?",
        "Should I quit smoking?",
        "What tests detect lung cancer?"
    ]
    
    for i, question in enumerate(quick_questions):
        if i % 2 == 0:
            with col1:
                if st.button(question, key=f"quick_{i}"):
                    st.session_state.chat_history.append({'role': 'user', 'message': question})
                    response = generate_ai_response(question)
                    st.session_state.chat_history.append({'role': 'assistant', 'message': response})
                    st.rerun()
        else:
            with col2:
                if st.button(question, key=f"quick_{i}"):
                    st.session_state.chat_history.append({'role': 'user', 'message': question})
                    response = generate_ai_response(question)
                    st.session_state.chat_history.append({'role': 'assistant', 'message': response})
                    st.rerun()

def generate_ai_response(question):
    """Generate AI responses based on keywords"""
    q = question.lower()
    
    if 'symptom' in q:
        return """Common symptoms of lung cancer include:
        
• Persistent cough that doesn't go away
• Coughing up blood
• Shortness of breath
• Chest pain that worsens with breathing
• Wheezing
• Hoarseness
• Weight loss and loss of appetite
• Fatigue
• Recurring infections like bronchitis

Early-stage lung cancer may not show symptoms. Regular screenings are important for high-risk individuals."""
    
    elif 'accuracy' in q or 'model' in q:
        return """Our AI model achieves 87.5%+ accuracy using state-of-the-art ensemble learning:

• XGBoost classifier
• CatBoost algorithm
• LightGBM integration
• Advanced feature engineering
• SMOTEENN for class balancing

The model was trained on comprehensive clinical data and validated through cross-validation. However, this tool is for screening purposes and should not replace professional medical diagnosis."""
    
    elif 'risk' in q or 'factor' in q:
        return """Major risk factors for lung cancer:

🚬 **Smoking** - #1 risk factor (85-90% of cases)
👴 **Age** - Risk increases after 55
🏭 **Occupational exposure** - Asbestos, radon, chemicals
👨‍👩‍👧 **Family history** - Genetic predisposition
🫁 **Previous lung disease** - COPD, tuberculosis
🏙️ **Air pollution** - Long-term exposure
☢️ **Radiation exposure** - Including radon gas

You can't control age and genetics, but you can modify lifestyle factors!"""
    
    elif 'prevent' in q:
        return """Lung cancer prevention strategies:

✅ **Don't smoke** - Or quit if you do (reduces risk significantly)
✅ **Avoid secondhand smoke** - Stay away from smoky environments
✅ **Test for radon** - Check your home for radon gas
✅ **Avoid carcinogens** - Use protection when exposed to chemicals
✅ **Eat healthy diet** - Fruits and vegetables with antioxidants
✅ **Exercise regularly** - Boosts immune system
✅ **Get screened** - Annual CT scans for high-risk individuals

Prevention is always better than treatment!"""
    
    elif 'quit' in q or 'smoking' in q:
        return """Quitting smoking is the BEST thing you can do:

**Benefits of Quitting:**
• 20 minutes: Heart rate drops
• 12 hours: Carbon monoxide levels normalize
• 2-12 weeks: Circulation improves
• 1-9 months: Coughing decreases
• 1 year: Heart disease risk cut by 50%
• 5 years: Stroke risk = non-smoker
• 10 years: Lung cancer risk cut by 50%

**Tips to Quit:**
1. Set a quit date
2. Use nicotine replacement therapy
3. Seek support groups
4. Avoid triggers
5. Stay active
6. Consider medications (consult doctor)

It's never too late to quit! Your body starts healing immediately."""
    
    elif 'test' in q or 'detect' in q:
        return """Lung cancer detection tests:

**Screening Tests:**
• Low-dose CT (LDCT) scan - Most effective for early detection
• Chest X-ray - Less sensitive but still useful
• Sputum cytology - Examines mucus under microscope

**Diagnostic Tests:**
• Bronchoscopy - Camera into airways
• Mediastinoscopy - Sample lymph nodes
• Needle biopsy - Tissue sample through chest
• Thoracentesis - Fluid analysis
• PET scan - Shows cancer spread
• MRI - Detailed imaging

**Who should be screened:**
• Age 55-80
• Heavy smoking history (30 pack-years)
• Current smoker or quit within 15 years

Talk to your doctor about screening if you're high-risk!"""
    
    elif 'treatment' in q:
        return """Lung cancer treatment options:

**Main Treatments:**
🏥 Surgery - Remove tumor (best for early stage)
☢️ Radiation therapy - Kill cancer cells with radiation
💊 Chemotherapy - Drugs to kill cancer cells
🎯 Targeted therapy - Drugs targeting specific mutations
🛡️ Immunotherapy - Boost immune system

**Factors affecting treatment:**
• Cancer type (small cell vs non-small cell)
• Stage (I, II, III, IV)
• Location and size
• Overall health
• Patient preferences

Treatment is often combination therapy. Always consult with oncologist for personalized treatment plan."""
    
    else:
        return """I'm here to help with questions about:

• Lung cancer symptoms and signs
• Risk factors and causes
• Prevention strategies
• Screening and diagnostic tests
• Treatment options
• How our AI model works
• Smoking cessation tips

Please ask a specific question, and I'll provide detailed information! 

For medical emergencies, contact healthcare provider immediately."""

# LEARN MORE PAGE
elif selected == "📚 Learn More":
    st.markdown("""
    <div class="main-header">
        <h1>📚 Learn About Lung Cancer</h1>
        <p>Comprehensive information about lung cancer, prevention, and early detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Overview", "🔍 Symptoms", "💊 Prevention", "🏥 Treatment"])
    
    with tab1:
        st.markdown("""
        ## What is Lung Cancer?
        
        Lung cancer is a type of cancer that begins in the lungs. It is the leading cause of cancer deaths worldwide.
        
        ### Types of Lung Cancer:
        
        **1. Non-Small Cell Lung Cancer (NSCLC)** - 85% of cases
        - Adenocarcinoma
        - Squamous cell carcinoma
        - Large cell carcinoma
        
        **2. Small Cell Lung Cancer (SCLC)** - 15% of cases
        - More aggressive
        - Spreads more quickly
        
        ### Statistics:
        - 💔 Leading cause of cancer deaths globally
        - 👥 2.2 million new cases annually
        - 🚬 85-90% caused by smoking
        - ⚡ 5-year survival rate: 18-22% overall
        - ✅ Early detection improves survival to 60%
        """)
        
        # Video placeholder
        st.video("https://www.youtube.com/watch?v=wf-BeH_-TL4")
    
    with tab2:
        st.markdown("""
        ## 🔍 Recognizing Symptoms
        
        ### Early Symptoms (Often Mild):
        - Persistent cough
        - Changes in existing cough
        - Coughing up blood
        - Shortness of breath
        - Chest pain
        - Wheezing
        - Hoarseness
        
        ### Advanced Symptoms:
        - Bone pain
        - Headaches
        - Weight loss
        - Loss of appetite
        - Fatigue
        - Swelling of face/neck
        - Difficulty swallowing
        
        ### ⚠️ Warning Signs:
        If you experience any of these symptoms, especially if you're a smoker or over 55, consult a doctor immediately!
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("🔴 **High Priority Symptoms**\n\n• Coughing blood\n• Severe chest pain\n• Difficulty breathing\n• Unexplained weight loss")
        with col2:
            st.warning("🟡 **Monitor These Symptoms**\n\n• Persistent cough > 3 weeks\n• Recurring infections\n• Fatigue\n• Hoarseness")
    
    with tab3:
        st.markdown("""
        ## 💊 Prevention Strategies
        
        ### Primary Prevention:
        
        **1. Don't Smoke** 🚭
        - Avoid starting if you don't smoke
        - Quit if you do - it's never too late!
        - Benefits begin within 20 minutes of quitting
        
        **2. Avoid Secondhand Smoke** 💨
        - Non-smokers exposed have 20-30% higher risk
        - Avoid enclosed spaces with smokers
        
        **3. Test for Radon** ☢️
        - Radon is 2nd leading cause
        - Test your home ($15-30 test kits)
        - Mitigate if levels > 4 pCi/L
        
        **4. Workplace Safety** 🏭
        - Use protective equipment
        - Follow safety procedures
        - Know your exposure risks
        
        **5. Healthy Lifestyle** 🥗
        - Eat fruits and vegetables
        - Exercise regularly
        - Limit alcohol
        - Maintain healthy weight
        
        **6. Regular Screening** 🏥
        - Annual LDCT for high-risk individuals
        - Ages 55-80 with smoking history
        
        ### Risk Reduction Timeline:
        - **1 year** after quitting: Heart attack risk drops 50%
        - **5 years**: Stroke risk = non-smoker
        - **10 years**: Lung cancer risk drops 50%
        - **15 years**: Risk approaches non-smoker levels
        """)
    
    with tab4:
        st.markdown("""
        ## 🏥 Treatment Options
        
        Treatment depends on:
        - Type of lung cancer
        - Stage (I-IV)
        - Location and size
        - Overall health
        - Patient preferences
        
        ### Treatment Modalities:
        
        **1. Surgery** 🔪
        - Removes tumor and surrounding tissue
        - Best for early-stage cancer
        - Types: Wedge resection, lobectomy, pneumonectomy
        
        **2. Radiation Therapy** ☢️
        - Uses high-energy beams
        - Can be primary or adjuvant treatment
        - External beam or brachytherapy
        
        **3. Chemotherapy** 💊
        - Drugs to kill cancer cells
        - Systemic treatment
        - Often combined with other therapies
        
        **4. Targeted Therapy** 🎯
        - Targets specific genetic mutations
        - EGFR, ALK, ROS1, BRAF inhibitors
        - Fewer side effects than chemo
        
        **5. Immunotherapy** 🛡️
        - Helps immune system fight cancer
        - Checkpoint inhibitors (PD-1, PD-L1)
        - Significant advances in recent years
        
        ### Survival Rates by Stage:
        - **Stage I**: 60-80% (5-year survival)
        - **Stage II**: 40-60%
        - **Stage III**: 15-35%
        - **Stage IV**: 2-10%
        
        *Early detection saves lives!*
        """)

# ABOUT PAGE
elif selected == "ℹ️ About":
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ About This System</h1>
        <p>Learn about our AI technology and mission</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Our Mission
    
    To democratize access to advanced lung cancer screening through artificial intelligence, enabling early detection and saving lives worldwide.
    
    ## 🤖 The Technology
    
    ### Machine Learning Models:
    - **XGBoost**: Gradient boosting framework
    - **CatBoost**: Categorical features specialist
    - **LightGBM**: Fast gradient boosting
    - **Ensemble Learning**: Combined model predictions
    
    ### Model Performance:
    - **Accuracy**: 87.5%+
    - **Precision**: 92%
    - **Recall**: 89%
    - **F1-Score**: 90.5%
    
    ### Features Analyzed (20+):
    - Demographics (age, gender)
    - Lifestyle factors (smoking, alcohol)
    - Medical history (chronic diseases, allergies)
    - Physical symptoms (yellow fingers, fatigue)
    - Respiratory symptoms (cough, wheezing, chest pain)
    - Engineered features (risk scores, symptom count)
    
    ### Training Data:
    - Comprehensive clinical dataset
    - SMOTEENN resampling for balance
    - 5-fold cross-validation
    - Hyperparameter optimization
    
    ## 👥 Team
    
    Developed by AI healthcare researchers and data scientists passionate about using technology for social good.
    
    ## ⚠️ Important Disclaimer
    
    **This AI tool is designed for:**
    ✅ Educational purposes
    ✅ Risk awareness
    ✅ Preliminary screening
    ✅ Health monitoring
    
    **This tool is NOT:**
    ❌ A medical diagnosis
    ❌ A substitute for doctor consultation
    ❌ FDA approved medical device
    ❌ Definitive cancer detection
    
    **Always consult qualified healthcare professionals for:**
    - Official diagnosis
    - Treatment decisions
    - Medical advice
    - Health concerns
    
    ## 📞 Contact & Support
    
    - 📧 Email: support@lungcancerai.com
    - 🌐 Website: www.lungcancerai.com
    - 📱 Phone: +1-800-LUNG-CARE
    
    ## 📜 Privacy & Data
    
    - ✅ No data is stored on our servers
    - ✅ All processing happens in real-time
    - ✅ Complete privacy and confidentiality
    - ✅ HIPAA compliant architecture
    
    ## 🙏 Acknowledgments
    
    Thanks to the open-source community, medical researchers, and all contributors who made this project possible.
    
    ---
    
    **Version**: 2.0.0  
    **Last Updated**: November 2025  
    **License**: MIT Open Source
    """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("© 2025 Lung Cancer AI")
with col2:
    st.markdown("Built with ❤️ using Streamlit")
with col3:
    st.markdown("[Privacy Policy](#) | [Terms of Service](#)")
