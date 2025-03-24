import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Set page configuration
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    layout="wide"
)

# Sidebar settings
st.sidebar.title("YOLOv8 Settings")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt")
)

# Confidence threshold
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05
)

# IOU threshold
iou_threshold = 0.45

# Main content
st.title("YOLOv8 Object Detection")

# Load model
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Load the selected model
try:
    model = load_model(model_type)
    st.sidebar.success(f"Model {model_type} loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# Function to run detection on image
def detect_objects(image, conf, iou):
    results = model(image, conf=conf, iou=iou)
    # Return the first result since we're processing a single image
    return results[0]

# Function to display detection results
def display_results(results, image):
    # Get the plotted image with detections
    res_plotted = results.plot()
    # Convert from BGR to RGB for displaying in Streamlit
    res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    # Display the image with detections
    st.image(res_plotted_rgb, caption="Detection Results", use_column_width=True)
    
    # Display detection information
    if len(results.boxes) > 0:
        st.subheader("Detection Details:")
        
        # Create columns for the detected objects table
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Class")
        with col2:
            st.write("Confidence")
        
            
        # Display each detection
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()  # Convert to numpy array
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(class_name)
            with col2:
                st.write(f"{confidence:.2f}")
            
    else:
        st.info("No objects detected.")

# Video processor class for WebRTC
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, model, conf_threshold, iou_threshold):
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.detection_results = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Run YOLO detection
        results = self.model(img, conf=self.conf_threshold, iou=self.iou_threshold)[0]
        self.detection_results = results
        
        # Draw the detection results on the frame
        annotated_frame = results.plot()
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Input selection
input_type = st.radio("Select Input Type", ("Image", "Video", "Webcam"))

if input_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert the uploaded file to an image
        image = np.array(Image.open(uploaded_file))
        
        # Display the original image
        st.subheader("Original Image")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process the image when the user clicks the button
        if st.button("Detect Objects"):
            with st.spinner("Detecting objects..."):
                results = detect_objects(image, conf_threshold, iou_threshold)
                display_results(results, image)

elif input_type == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # Open the video file
        video = cv2.VideoCapture(tfile.name)
        
        # Display video information
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        st.write(f"FPS: {fps}")
        st.write(f"Frame Count: {frame_count}")
        st.write(f"Duration: {duration:.2f} seconds")
        
        # Process the video when the user clicks the button
        if st.button("Detect Objects"):
            stframe = st.empty()
            
            # Process every frame
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                
                # Run detection on the frame
                results = detect_objects(frame, conf_threshold, iou_threshold)
                
                # Get the frame with detections
                res_plotted = results.plot()
                
                # Convert from BGR to RGB
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                # Display the frame with detections
                stframe.image(res_plotted_rgb, caption="Detection Results", use_column_width=True)
            
            video.release()

elif input_type == "Webcam":
    st.write("Real-time Webcam Object Detection")
    
    # Create a placeholder for displaying detection results
    detection_placeholder = st.empty()
    info_placeholder = st.empty()
    
    # WebRTC configuration
    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Create the WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="yolov8-detection",
        video_processor_factory=lambda: YOLOVideoProcessor(
            model=model,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        ),
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Display information about detection
    if webrtc_ctx.video_processor:
        # Display detected objects
        if st.checkbox("Show Detection Details", value=True):
            # This will keep updating
            if webrtc_ctx.state.playing:
                info_text = "Stream is active. Detection results will appear here in real-time."
                info_placeholder.info(info_text)
            else:
                info_placeholder.info("Click 'Start' to begin webcam detection.")

# Add information about the project
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("This application uses YOLOv8 for real-time object detection in images, videos, and webcam streams.")
