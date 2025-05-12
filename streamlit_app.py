import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import tempfile
from datetime import datetime
from ultralytics import YOLO
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="UZ Smart Parking",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
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
    .video-container {
        margin-top: 20px;
        border: 2px solid #FFD700;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "parking_lots" not in st.session_state:
    st.session_state.parking_lots = [
        {"id": 1, "name": "Main Car Park", "capacity": 50, "occupied": 0, "reserved": [], 
         "parking_areas": [[(511,327),(557,388),(603,383),(549,324)],
                          [(400,300),(450,350),(500,340),(450,290)]]},
        {"id": 2, "name": "Library Parking", "capacity": 30, "occupied": 0, "reserved": [],
         "parking_areas": [[(300,200),(350,250),(400,240),(350,190)]]},
        {"id": 3, "name": "Great Hall Parking", "capacity": 20, "occupied": 0, "reserved": [],
         "parking_areas": [[(200,100),(250,150),(300,140),(250,90)]]}
    ]
# User Dashboard
def user_dashboard():
    st.title("🏛️ UZ Smart Parking")
    st.subheader(f"Welcome {st.session_state.auth['role'].title()} from {st.session_state.auth['department']}")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Available Parking Lots")
        for lot in st.session_state.parking_lots:
            with st.expander(f"📍 {lot['name']} - {lot['capacity'] - lot['occupied'] - len(lot['reserved'])} spots available"):
                purpose = st.selectbox("Purpose of Visit", PURPOSE_OF_VISIT, key=f"purpose_{lot['id']}")
                duration = st.number_input("Duration (minutes)", 
                                         min_value=30, 
                                         max_value=240, 
                                         value=60,
                                         key=f"dur_{lot['id']}")
                
                if st.button(f"Reserve Spot at {lot['name']}", key=f"res_{lot['id']}"):
                    if reserve_spot(lot["id"], st.session_state.auth["username"], duration, purpose):
                        st.success(f"Reservation confirmed for {purpose}!")
                    else:
                        st.error("No available spots. Please try another lot.")
                    st.rerun()

    with col2:
        st.header("Quick Stats")
        total_capacity = sum(lot["capacity"] for lot in st.session_state.parking_lots)
        total_occupied = sum(lot["occupied"] + len(lot["reserved"]) for lot in st.session_state.parking_lots)
        
        st.metric("Total Spaces", total_capacity)
        st.metric("Available Spaces", total_capacity - total_occupied)
        
        # Pie chart of purposes
        purposes = [res["purpose"] for lot in st.session_state.parking_lots for res in lot["reserved"]]
        if purposes:
            purpose_df = pd.DataFrame({"Purpose": purposes})
            fig = px.pie(purpose_df, names="Purpose", title="Parking Purposes")
            st.plotly_chart(fig, use_container_width=True)

# Admin Dashboard
def admin_dashboard():
    st.title("🛠️ UZ Parking Administration")
    
    tab1, tab2, tab3 = st.tabs(["📹 Live Monitoring", "📋 Reservations", "📊 Analytics"])

    with tab1:
        st.header("CCTV Parking Monitoring")
        uploaded_video = st.file_uploader("Upload CCTV footage", type=["mp4", "mov"])
        selected_lot = st.selectbox("Select Parking Lot", 
                                  st.session_state.parking_lots, 
                                  format_func=lambda x: x["name"])
        
        model = load_model()
        
        if uploaded_video and model and st.button("Analyze Parking"):
            with st.spinner("Detecting vehicles..."):
                temp_video = "temp_video.mp4"
                with open(temp_video, "wb") as f:
                    f.write(uploaded_video.getbuffer())
                
                car_count = process_video(temp_video, model, selected_lot["id"])
                st.success(f"Detected {car_count} vehicles in {selected_lot['name']}")
                st.video(temp_video)
                os.remove(temp_video)
                st.rerun()

    with tab2:
        st.header("Current Reservations")
        reservation_data = []
        for lot in st.session_state.parking_lots:
            for res in lot["reserved"]:
                reservation_data.append({
                    "Lot": lot["name"],
                    "User": res["user_id"],
                    "Department": res["department"],
                    "Purpose": res["purpose"],
                    "Duration": f"{res['duration']} mins",
                    "Time": res["start_time"].strftime("%Y-%m-%d %H:%M")
                })
        
        if reservation_data:
            st.dataframe(pd.DataFrame(reservation_data), 
                        use_container_width=True,
                        hide_index=True)
        else:
            st.info("No current reservations")

    with tab3:
        st.header("Parking Analytics")
        
        # Capacity utilization
        utilization_data = []
        for lot in st.session_state.parking_lots:
            utilization = (lot["occupied"] + len(lot["reserved"])) / lot["capacity"] * 100
            utilization_data.append({
                "Lot": lot["name"],
                "Capacity": lot["capacity"],
                "Occupied": lot["occupied"] + len(lot["reserved"]),
                "Utilization": utilization
            })
        
        df = pd.DataFrame(utilization_data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Lot Utilization")
            fig = px.bar(df, x="Lot", y="Utilization", color="Lot",
                        title="Parking Lot Utilization (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Purpose Distribution")
            purposes = [res["purpose"] for lot in st.session_state.parking_lots for res in lot["reserved"]]
            if purposes:
                purpose_df = pd.DataFrame({"Purpose": purposes})
                fig = px.pie(purpose_df, names="Purpose", 
                            title="Purpose of Visits")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No reservation data available")
# ... (keep your existing auth and credentials code)

class ParkingDetector:
    def __init__(self):
        self.model = self.load_model()
        self.class_list = self.load_class_names()
        
    @st.cache_resource
    def load_model(_self):
        try:
            return YOLO('yolov8s.pt')
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            return None
    
    @st.cache_data
    def load_class_names(_self):
        try:
            with open("coco.txt", "r") as f:
                return f.read().split("\n")
        except Exception as e:
            st.error(f"Failed to load class names: {str(e)}")
            return []
    
    def process_frame(self, frame, parking_areas):
        frame = cv2.resize(frame, (1020, 500))
        results = self.model.predict(frame)
        px = pd.DataFrame(results[0].boxes.data).astype("float")
        
        area_counts = [0] * len(parking_areas)
        annotated_frame = frame.copy()
        
        for _, row in px.iterrows():
            x1, y1, x2, y2, _, d = map(int, row[:6])
            obj_class = self.class_list[d]
            
            if 'car' in obj_class or 'truck' in obj_class or 'bus' in obj_class:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                for i, area in enumerate(parking_areas):
                    if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        area_counts[i] += 1
        
        # Draw parking areas
        for i, area in enumerate(parking_areas):
            color = (0, 0, 255) if area_counts[i] > 0 else (0, 255, 0)
            cv2.polylines(annotated_frame, [np.array(area, np.int32)], True, color, 2)
            cv2.putText(annotated_frame, str(i+1), tuple(area[2]), 
                       cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame, area_counts

def process_parking_video(video_path, parking_lot):
    detector = ParkingDetector()
    parking_areas = parking_lot["parking_areas"]
    cap = cv2.VideoCapture(video_path)
    total_counts = [0] * len(parking_areas)
    
    video_placeholder = st.empty()
    results_placeholder = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated_frame, counts = detector.process_frame(frame, parking_areas)
        total_counts = [max(total_counts[i], counts[i]) for i in range(len(counts))]
        
        video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True, caption="Live Parking Detection")
        
        with results_placeholder.container():
            st.subheader("Real-time Detection Results")
            cols = st.columns(len(parking_areas))
            for i, count in enumerate(counts):
                with cols[i]:
                    st.metric(f"Area {i+1}", f"{count} vehicles", 
                             delta=None, delta_color="normal")
    
    cap.release()
    return total_counts

# ... (keep your existing user_dashboard and admin_dashboard functions, but modify the video processing part in admin_dashboard)

def admin_dashboard():
    st.title("🛠️ UZ Parking Administration")
    
    tab1, tab2, tab3 = st.tabs(["📹 Live Monitoring", "📋 Reservations", "📊 Analytics"])

    with tab1:
        st.header("CCTV Parking Monitoring")
        uploaded_video = st.file_uploader("Upload CCTV footage", type=["mp4", "mov", "avi"])
        selected_lot = st.selectbox("Select Parking Lot", 
                                  st.session_state.parking_lots, 
                                  format_func=lambda x: x["name"])
        
        if uploaded_video:
            st.markdown("<div class='video-container'>", unsafe_allow_html=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_video.getbuffer())
                video_path = tmp_file.name
            
            if st.button("Analyze Parking"):
                with st.spinner("Processing video..."):
                    area_counts = process_parking_video(video_path, selected_lot)
                    occupied = sum(1 for count in area_counts if count > 0)
                    selected_lot["occupied"] = occupied
                    
                    st.success("Analysis complete!")
                    st.subheader("Parking Space Utilization")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Spaces", len(area_counts))
                        st.metric("Occupied Spaces", occupied)
                        st.metric("Vacancy Rate", f"{(len(area_counts) - occupied)/len(area_counts)*100:.1f}%")
                    
                    with col2:
                        util_df = pd.DataFrame({
                            "Status": ["Occupied", "Vacant"],
                            "Spaces": [occupied, len(area_counts) - occupied]
                        })
                        fig = px.pie(util_df, values="Spaces", names="Status",
                                    color="Status", color_discrete_map={"Occupied":"red", "Vacant":"green"})
                        st.plotly_chart(fig, use_container_width=True)
            
            os.unlink(video_path)
            st.markdown("</div>", unsafe_allow_html=True)

    # ... (keep your existing tab2 and tab3 code)

# ... (keep your remaining code including main() function)

if __name__ == "__main__":
    main()
