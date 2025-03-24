# app_mobile.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import time

# Set page configuration - optimize for mobile
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    layout="centered"  # Use centered layout for mobile
)

# App title with mobile styling
st.markdown("""
    <style>
    .main-header {
        font-size: 1.8rem;
        margin-bottom: 0.8rem;
    }
    .stButton button {
        width: 100%;
        height: 3rem;
        font-size: 1rem;
    }
    </style>
    <h1 class="main-header">YOLOv8 Object Detection</h1>
    <h3>Mobile Version</h3>
    """, unsafe_allow_html=True)

# Expandable sidebar for settings
with st.expander("⚙️ Settings"):
    # Model selection - simplified for mobile
    model_type = st.selectbox(
        "Select Model Type",
        ("yolov8n.pt", "yolov8s.pt")  # Limit to smaller models for mobile
    )
    
    # Confidence threshold
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.35,  # Default higher for mobile
        step=0.05
    )
    
    # IOU threshold
    iou_threshold = 0.45  # Fixed value for mobile

# Load model - optimized for mobile
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Load the selected model
try:
    model = load_model(model_type)
    st.success(f"✅ Model {model_type} loaded")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Function to run detection on image
def detect_objects(image, conf, iou):
    results = model(image, conf=conf, iou=iou)
    return results[0]

# Function to display detection results - simplified for mobile
def display_results(results, image):
    # Get the plotted image with detections
    res_plotted = results.plot()
    # Convert from BGR to RGB for displaying in Streamlit
    res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    # Display the image with detections
    st.image(res_plotted_rgb, caption="Detection Results", use_column_width=True)
    
    # Display simplified detection information
    if len(results.boxes) > 0:
        st.write(f"**Detected {len(results.boxes)} objects:**")
        
        # Create simplified mobile-friendly output
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            confidence = float(box.conf[0])
            st.write(f"• {class_name}: {confidence:.2f}")
    else:
        st.info("No objects detected")

# Updated Video processor class for WebRTC - optimized for mobile
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, model, conf_threshold, iou_threshold):
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.frame_count = 0
        self.last_process_time = time.time()
        self.last_frame = None  # Store the last processed frame for debugging
    
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            
            current_time = time.time()
            # Process every 3rd frame on mobile for better performance
            if self.frame_count % 3 == 0 or current_time - self.last_process_time > 0.5:
                # Run YOLO detection
                results = self.model(img, conf=self.conf_threshold, iou=self.iou_threshold)[0]
                # Draw the detection results on the frame
                annotated_frame = results.plot()
                self.last_process_time = current_time
                self.last_frame = av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
            else:
                # Skip processing for better performance
                annotated_frame = img
                self.last_frame = av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
            
            return self.last_frame
        except Exception as e:
            # Log any errors during frame processing
            st.error(f"Error processing frame: {e}")
            return frame  # Return the original frame if processing fails

# Input selection - simplified mobile UI with tabs
tab1, tab2, tab3 = st.tabs(["📸 Camera", "🖼️ Image", "🎥 Video"])

with tab1:
    st.write("Use your camera to detect objects in real-time")
    
    # Use camera instead of WebRTC for better mobile compatibility
    camera_mode = st.radio("Choose camera mode:", ["Take Picture", "Live Stream"], horizontal=True)
    
    if camera_mode == "Take Picture":
        camera_input = st.camera_input("Capture image")
        if camera_input is not None:
            image = np.array(Image.open(camera_input))
            
            with st.spinner("Detecting objects..."):
                results = detect_objects(image, conf_threshold, iou_threshold)
                display_results(results, image)
    else:
        # Camera stream - use WebRTC with mobile optimizations
        st.write("📱 Point camera at objects")
        
        # Camera options for mobile
        camera_type = st.radio("Camera:", ["Back Camera", "Front Camera"], horizontal=True)
        facingMode = "environment" if camera_type == "Back Camera" else "user"
        
        # WebRTC configuration with mobile optimizations
        rtc_config = RTCConfiguration(
            {"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]}
        )
        
        # Video constraints for mobile
        video_constraints = {
            "video": {
                "facingMode": facingMode,
                "width": {"ideal": 320},
                "height": {"ideal": 240},
                "frameRate": {"max": 15}
            },
            "audio": False,
        }
        
        # Add a placeholder for the video feed
        video_placeholder = st.empty()
        
        # Create the WebRTC streamer with mobile optimizations
        webrtc_ctx = webrtc_streamer(
            key="yolov8-mobile-detection",
            video_processor_factory=lambda: YOLOVideoProcessor(
                model=model,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            ),
            rtc_configuration=rtc_config,
            media_stream_constraints=video_constraints,
            async_processing=True,
        )
        
        # Add debugging information
        if webrtc_ctx.state.playing:
            st.success("Camera active - detecting objects")
            # Check if the video processor is receiving frames
            if webrtc_ctx.video_processor:
                st.write("Video processor is active.")
            else:
                st.error("Video processor not initialized.")
        else:
            st.info("Click 'Start' to begin camera detection")
            st.warning("If the camera doesn't start, ensure camera permissions are granted in your browser.")
        
        # Display the video feed manually if needed
        if webrtc_ctx.video_processor:
            try:
                frame = webrtc_ctx.video_processor.last_frame
                if frame is not None:
                    # Convert the frame to an image and display it
                    frame_rgb = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, caption="Live Stream with Detections", use_column_width=True)
                else:
                    st.warning("No frames received yet. Ensure the camera is working and permissions are granted.")
            except AttributeError:
                st.error("Error accessing the last frame. The video processor might not be processing frames correctly.")

with tab2:
    st.write("Upload an image to detect objects")
    
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Process image
        image = np.array(Image.open(uploaded_file))
        
        # Make button prominent for mobile
        if st.button("Detect Objects in Image", use_container_width=True):
            with st.spinner("Processing..."):
                results = detect_objects(image, conf_threshold, iou_threshold)
                display_results(results, image)
                
                # Show original image after results for comparison
                with st.expander("Show Original Image"):
                    st.image(image, caption="Original Image", use_column_width=True)

with tab3:
    st.write("Upload a short video to detect objects")
    st.warning("For mobile devices, keep videos short (< 10s) for best performance")
    
    uploaded_file = st.file_uploader("Choose a video", type=["mp4", "mov"])
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # Mobile-optimized video processing
        video = cv2.VideoCapture(tfile.name)
        
        # Get some basic info
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        # Display warning if video is too long for mobile
        if duration > 15:
            st.warning(f"Video is {duration:.1f} seconds long. Processing may be slow on mobile.")
        
        # Mobile-friendly processing with skip frames
        skip_frames = 5  # Process every 5th frame for mobile
        
        if st.button("Process Video", use_container_width=True):
            stframe = st.empty()
            progress_bar = st.progress(0)
            
            # Process frames (optimized for mobile)
            frame_idx = 0
            
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                
                # Process every Nth frame for efficiency
                if frame_idx % skip_frames == 0:
                    # Update progress
                    progress = int(min(frame_idx / frame_count * 100, 100))
                    progress_bar.progress(progress)
                    
                    # Run detection
                    results = detect_objects(frame, conf_threshold, iou_threshold)
                    
                    # Show frame with detections
                    res_plotted = results.plot()
                    res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    stframe.image(res_plotted_rgb, caption=f"Frame {frame_idx}", use_column_width=True)
                
                frame_idx += 1
                
                # Break early if we've shown enough frames for mobile
                if frame_idx > 100:
                    st.info("Processing limited to first 100 frames for mobile performance")
                    break
            
            # Clean up
            video.release()
            progress_bar.progress(100)
            st.success("✅ Video processing complete!")

# Footer for mobile version
st.markdown("---")
st.info("📱 Optimized for mobile devices")
st.caption("For more features, try the desktop version")
