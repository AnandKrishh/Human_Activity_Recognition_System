import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
import joblib
from collections import deque
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import time

# Page config
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('logistic_regression_model.joblib')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# Feature extraction function
def extract_features(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize to a fixed size (smaller dimension to get 200 features)
    resized = cv2.resize(gray, (20, 10))  # 20x10 = 200 features
    
    # Flatten and normalize features
    features = resized.flatten()
    features = features / 255.0
    
    return features.reshape(1, -1)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = deque(maxlen=20)
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()

# Sidebar
with st.sidebar:
    st.title("Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        help="Minimum confidence score to display predictions"
    )
    
    fps_display = st.empty()
    prediction_rate = st.empty()
    
    show_metrics = st.button(
        "Show Performance Metrics",
        help="Display model performance metrics based on test data"
    )

# Main content
st.title("Human Activity Recognition System")

# Two columns layout
col1, col2 = st.columns([2, 1])

with col1:
    # Video feed placeholder
    video_placeholder = st.empty()
    
    # Camera input
    img_data = st.camera_input("Take a picture")

    if img_data is not None:
        # Convert to numpy array
        file_bytes = np.asarray(bytearray(img_data.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        # Update FPS calculation
        st.session_state.frame_count += 1
        elapsed_time = time.time() - st.session_state.start_time
        fps = st.session_state.frame_count / elapsed_time
        fps_display.metric("FPS", f"{fps:.2f}")
        
        # Extract features and make prediction
        model = load_model()
        if model is not None:
            features = extract_features(frame)
            prediction = model.predict(features)
            confidence = model.predict_proba(features).max()
            
            if confidence >= confidence_threshold:
                # Add to predictions history
                st.session_state.predictions.appendleft({
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'activity': prediction[0],
                    'confidence': f"{confidence:.2f}"
                })
                
                # Draw prediction on frame
                cv2.putText(
                    frame,
                    f"{prediction[0]} ({confidence:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            
            # Display annotated image
            video_placeholder.image(frame, channels="BGR", use_column_width=True)
            
            # Update prediction rate
            pred_rate = len(st.session_state.predictions) / elapsed_time
            prediction_rate.metric("Predictions/sec", f"{pred_rate:.2f}")

with col2:
    # Recent predictions table
    st.subheader("Recent Predictions")
    if st.session_state.predictions:
        df_predictions = pd.DataFrame(list(st.session_state.predictions))
        st.dataframe(
            df_predictions,
            hide_index=True,
            use_container_width=True
        )
    
    # Performance metrics
    if show_metrics:
        st.subheader("Model Performance")
        try:
            # Load test data
            test_data = pd.read_csv('test.csv')
            X_test = test_data.drop('Activity', axis=1)
            y_test = test_data['Activity']
            
            # Load model if not already loaded
            if 'model' not in locals():
                model = load_model()
            
            # Calculate metrics
            y_pred = model.predict(X_test)
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted')
            }
            
            # Display metrics
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.2%}")
            
            # Plot confusion matrix
            fig = px.imshow(
                confusion_matrix(y_test, y_pred, normalize='true'),
                labels=dict(x="Predicted", y="Actual"),
                title="Confusion Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading test data: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "Built with Streamlit • Human Activity Recognition System • "
    f"Session started: {datetime.fromtimestamp(st.session_state.start_time).strftime('%Y-%m-%d %H:%M:%S')}"
)
