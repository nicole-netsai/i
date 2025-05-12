import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from ultralytics import YOLO
import plotly.express as px
import tempfile

# Page configuration (keep your existing setup)
st.set_page_config(
    page_title="UZ Smart Parking",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling (keep your existing styles)
st.markdown("""
<style>
    .header {
        color: #FFD700;
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #002366;
    }
    .stButton>button {
        background-color: #FFD700;
        color: #002366;
        font-weight: bold;
    }
    .stSelectbox, .stTextInput, .stNumberInput {
        background-color: #f0f2f6;
    }
    .stExpander {
        background-color: #f0f2f6;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state (keep your existing setup)
if "parking_lots" not in st.session_state:
    st.session_state.parking_lots = [
        {"id": 1, "name": "Main Car Park", "capacity": 50, "occupied": 0, "reserved": [], "parking_areas": []},
        {"id": 2, "name": "Library Parking", "capacity": 30, "occupied": 0, "reserved": [], "parking_areas": []},
        {"id": 3, "name": "Great Hall Parking", "capacity": 20, "occupied": 0, "reserved": [], "parking_areas": []}
    ]

# ... (keep your existing auth and credentials code)

# Load YOLO model with caching
@st.cache_resource(show_spinner="Loading vehicle detection model...")
def load_model():
    try:
        model = YOLO('yolov8s.pt')  # Using the same model as your detection code
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

@st.cache_data
def load_class_names():
    try:
        with open("coco.txt", "r") as f:
            class_list = f.read().split("\n")
        return class_list
    except Exception as e:
        st.error(f"Failed to load class names: {str(e)}")
        return []

def process_parking_video(video_path, model, class_list, parking_areas):
    cap = cv2.VideoCapture(video_path)
    area_counts = {i: 0 for i in range(len(parking_areas))}
    frames_processed = 0
    
    # Create placeholder for video display
    video_placeholder = st.empty()
    results_placeholder = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")
        
        # Reset counts for this frame
        current_frame_counts = {i: 0 for i in range(len(parking_areas))}
        
        for index, row in px.iterrows():
            x1, y1, x2, y2, _, d = map(int, row[:6])
            c = class_list[d]
            
            if 'car' in c or 'truck' in c or 'bus' in c:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Check each parking area
                for i, area in enumerate(parking_areas):
                    if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                        current_frame_counts[i] += 1
        
        # Update cumulative counts
        for i in current_frame_counts:
            if current_frame_counts[i] > 0:  # Only count if we see a vehicle in this frame
                area_counts[i] = max(area_counts[i], current_frame_counts[i])
        
        # Draw parking areas
        for i, area in enumerate(parking_areas):
            color = (0, 0, 255) if current_frame_counts[i] > 0 else (0, 255, 0)
            cv2.polylines(frame, [np.array(area, np.int32)], True, color, 2)
            cv2.putText(frame, str(i+1), tuple(area[2]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        
        frames_processed += 1
        video_placeholder.image(frame, channels="BGR", use_column_width=True)
        
        # Update real-time results
        with results_placeholder.container():
            st.subheader("Real-time Parking Detection")
            cols = st.columns(len(parking_areas))
            for i, area in enumerate(parking_areas):
                with cols[i]:
                    st.metric(f"Area {i+1} Vehicles", current_frame_counts[i])
    
    cap.release()
    return area_counts, frames_processed

# Enhanced Admin Dashboard with parking space detection
def admin_dashboard():
    st.title("🛠️ UZ Parking Administration")
    
    tab1, tab2, tab3 = st.tabs(["📹 Live Monitoring", "📋 Reservations", "📊 Analytics"])

    with tab1:
        st.header("CCTV Parking Monitoring")
        uploaded_video = st.file_uploader("Upload CCTV footage", type=["mp4", "mov", "avi"])
        selected_lot = st.selectbox("Select Parking Lot", 
                                  st.session_state.parking_lots, 
                                  format_func=lambda x: x["name"])
        
        # Parking area configuration
        st.subheader("Configure Parking Areas")
        num_areas = st.number_input("Number of Parking Areas", min_value=1, max_value=20, value=4)
        
        # For simplicity, we'll use some default areas - in a real app you'd want to let the admin draw these
        default_areas = [
            [(511,327),(557,388),(603,383),(549,324)],
            [(400,300),(450,350),(500,340),(450,290)],
            [(300,200),(350,250),(400,240),(350,190)],
            [(200,100),(250,150),(300,140),(250,90)]
        ]
        parking_areas = default_areas[:num_areas]
        
        model = load_model()
        class_list = load_class_names()
        
        if uploaded_video and model and class_list and st.button("Analyze Parking"):
            with st.spinner("Analyzing parking footage..."):
                # Save uploaded video to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(uploaded_video.getbuffer())
                    video_path = tmp_file.name
                
                # Process video
                area_counts, frames_processed = process_parking_video(video_path, model, class_list, parking_areas)
                
                # Calculate totals
                total_spaces = len(parking_areas)
                occupied_spaces = sum(1 for count in area_counts.values() if count > 0)
                vacant_spaces = total_spaces - occupied_spaces
                
                # Update parking lot status
                selected_lot["occupied"] = occupied_spaces
                selected_lot["capacity"] = total_spaces
                selected_lot["parking_areas"] = parking_areas
                
                # Display results
                st.success("Parking analysis complete!")
                st.subheader("Parking Lot Analysis Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Parking Areas", total_spaces)
                with col2:
                    st.metric("Occupied Spaces", occupied_spaces)
                with col3:
                    st.metric("Vacant Spaces", vacant_spaces)
                
                # Show detailed area status
                st.subheader("Parking Area Status")
                area_data = []
                for i, count in area_counts.items():
                    area_data.append({
                        "Area": f"Area {i+1}",
                        "Status": "Occupied" if count > 0 else "Vacant",
                        "Vehicle Count": count
                    })
                
                st.dataframe(pd.DataFrame(area_data), use_container_width=True)
                
                # Clean up
                os.unlink(video_path)
                st.rerun()

    # ... (keep your existing tab2 and tab3 code)

# ... (keep your remaining code including main() function)

if __name__ == "__main__":
    main()
