import os
import sys
import tempfile
import streamlit as st
import numpy as np
from ultralytics import YOLO
import torch
import cv2

# Environment information
st.info(f"Python version: {sys.version.split()[0]}")
st.info(f"OpenCV version: {cv2.__version__}")
st.info(f"PyTorch version: {torch.__version__}")
st.info(f"Ultralytics version: {YOLO.__version__}")

# Initialize YOLO model
@st.cache_resource
def load_model():
    model_path = "model/3000_best_60single.pt"
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {os.path.abspath(model_path)}")
        st.error("Current directory contents:")
        st.code(os.listdir())
        return None
        
    try:
        model = YOLO(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

model = load_model()
if not model:
    st.error("Failed to load model. Application cannot continue.")
    st.stop()

class_names = {0: 'defect', 1: 'bottle', 2: 'bottle_neck'}
MM_PER_PIXEL = 0.12  # Calibration factor

# Streamlit UI
st.title("🍾 Automated Bottle Inspection System")
st.markdown("### Upload a video file for inspection")

# Video processing parameters
conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
show_measurements = st.toggle("Show Measurements", True)

# File uploader
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

frame_placeholder = st.empty()
stop_button = st.button("Stop Processing")

# Font settings
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 1
LINE_HEIGHT = 25

# Process video if uploaded
if uploaded_file and not stop_button:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("Failed to open video file")
        st.stop()
    
    while cap.isOpened() and not stop_button:
        success, frame = cap.read()
        if not success:
            st.warning("End of video reached")
            break
        
        # YOLO prediction
        try:
            results = model.predict(
                frame, 
                conf=conf_threshold,
                verbose=False,
                imgsz=640
            )
            
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                
                for box, cls, conf in zip(boxes, classes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(cls)
                    label = class_names.get(class_id, 'unknown')
                    
                    # Set colors and labels
                    color = (0, 0, 255)  # Default: red for defects
                    text_lines = [f"{label} {conf:.2f}"]
                    
                    if label == "bottle":
                        color = (255, 0, 0)  # Blue
                    elif label == "bottle_neck":
                        color = (0, 255, 0)  # Green
                        width_mm = (x2 - x1) * MM_PER_PIXEL
                        height_mm = (y2 - y1) * MM_PER_PIXEL
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        
                        if show_measurements:
                            text_lines.extend([
                                f"Width: {width_mm:.1f}mm",
                                f"Height: {height_mm:.1f}mm"
                            ])
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    if text_lines:
                        y0 = max(y1 - len(text_lines) * LINE_HEIGHT, 10)
                        for i, line in enumerate(text_lines):
                            y_pos = y0 + (i * LINE_HEIGHT)
                            cv2.putText(
                                frame, line, 
                                (x1 + 5, y_pos), 
                                FONT, FONT_SCALE, 
                                (255, 255, 255), FONT_THICKNESS,
                                cv2.LINE_AA
                            )
                    
                    # Draw neck center
                    if label == "bottle_neck":
                        cv2.drawMarker(
                            frame, center, color,
                            markerType=cv2.MARKER_CROSS,
                            markerSize=20, thickness=2
                        )
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
        
        # Display frame
        frame_placeholder.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            use_column_width=True,
            channels="RGB"
        )
    
    # Release resources
    cap.release()
    if os.path.exists(tfile.name):
        os.unlink(tfile.name)

if not uploaded_file:
    st.info("Please upload a video file to start inspection")
elif stop_button:
    st.success("Processing stopped")
