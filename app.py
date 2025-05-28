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

# Load the label encoder
@st.cache_resource
def load_label_encoder():
    try:
        encoder = joblib.load('./label_encoder.joblib')
        return encoder
    except Exception as e:
        st.error(f"Failed to load label encoder: {str(e)}")
        return None

# Load the feature selector if available
@st.cache_resource
def load_feature_selector():
    try:
        selector = joblib.load('./feature_selector.joblib')
        return selector
    except Exception:
        return None

# Feature extraction function
def extract_features(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize to a fixed size
    resized = cv2.resize(gray, (64, 64))
    
    # Extract HOG features
    features = resized.flatten()
    
    # Normalize features
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
    # Camera input widget for browser-based photo capture
    camera_image = st.camera_input("Take a picture")
    image_placeholder = st.empty()

    if camera_image is not None:
        try:
            # Convert the uploaded image to a numpy array
            file_bytes = np.asarray(bytearray(camera_image.getvalue()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            model = load_model()
            encoder = load_label_encoder()
            selector = load_feature_selector()
            if model is not None and encoder is not None:
                features = extract_features(frame)
                if selector is not None:
                    features = selector.transform(features)
                else:
                    st.warning("Feature selector not found. Using raw features. Please ensure 'feature_selector.joblib' is present for correct predictions.")
                prediction = model.predict(features)
                confidence = model.predict_proba(features).max()
                activity_name = encoder.inverse_transform([prediction[0]])[0]
                # Draw prediction on frame
                cv2.putText(
                    frame,
                    f"{activity_name} ({confidence:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                # Show the frame
                image_placeholder.image(frame, channels="BGR", use_column_width=True)
                # Add to predictions history
                st.session_state.predictions.appendleft({
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'activity': activity_name,
                    'confidence': f"{confidence:.2f}"
                })
        except Exception as e:
            st.error(f"Error: {str(e)}")

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
