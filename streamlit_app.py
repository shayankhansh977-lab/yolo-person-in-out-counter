
import os
import json
import threading

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import av
import streamlit as st

from pathlib import Path
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ---------------- SETTINGS ----------------

st.set_page_config(
    page_title="VisionTrack AI | Live Counter",
    page_icon="🎯",
    layout="wide"
)

MODEL_PATH = "yolo11n.pt"
COUNTS_FILE = Path(__file__).with_name("counts.json")

# Shared lock protects count updates and JSON writes
count_lock = threading.Lock()


# ---------------- SAVE / LOAD COUNTS ----------------

def load_counts():
    if COUNTS_FILE.exists():
        try:
            with open(COUNTS_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)

            return {
                "in_count": int(data.get("in_count", 0)),
                "out_count": int(data.get("out_count", 0))
            }
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    return {"in_count": 0, "out_count": 0}


def save_counts(in_count, out_count):
    data = {
        "in_count": int(in_count),
        "out_count": int(out_count)
    }

    temp_file = COUNTS_FILE.with_suffix(".tmp")

    try:
        with open(temp_file, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        os.replace(temp_file, COUNTS_FILE)
    except OSError as error:
        print("Counts save error:", error)


saved = load_counts()

if "final_in" not in st.session_state:
    st.session_state.final_in = saved["in_count"]

if "final_out" not in st.session_state:
    st.session_state.final_out = saved["out_count"]


# ---------------- PREMIUM UI ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #080d1b, #101a30, #080d1b);
    color: #f1f5f9;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
}
.hero {
    background: linear-gradient(120deg, #172b49, #101827);
    border: 1px solid #304663;
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
    font-size: 38px;
}
.hero p {
    color: #a9bdd8;
}
.metric-card {
    background: #131f32;
    border: 1px solid #2a3d58;
    border-radius: 16px;
    padding: 22px;
    text-align: center;
}
.metric-label {
    color: #a9bdd8;
    font-size: 15px;
}
.metric-value {
    font-size: 38px;
    font-weight: bold;
}
.green { color: #34d399; }
.red { color: #fb7185; }
.blue { color: #60a5fa; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🎯 VisionTrack AI</h1>
    <p>Real-Time Person Detection & IN/OUT Tracking</p>
    <p>YOLO11n · ByteTrack · OpenCV</p>
</div>
""", unsafe_allow_html=True)


# ---------------- LIVE PERSON COUNTER ----------------

class PersonCounter(VideoProcessorBase):

    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        current = load_counts()

        self.in_count = current["in_count"]
        self.out_count = current["out_count"]

        self.previous_positions = {}
        self.counted_ids = set()
        self.last_save = (
            self.in_count,
            self.out_count
        )

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        height, width = image.shape[:2]
        line_x = width // 2

        results = self.model.track(
            image,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.35,
            imgsz=640,
            verbose=False
        )

        result = results[0]
        annotated = result.plot()

        # Draw yellow counting line
        cv2.line(
            annotated,
            (line_x, 0),
            (line_x, height - 1),
            (0, 255, 255),
            3
        )

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()

            for box, person_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                cv2.circle(
                    annotated,
                    (center_x, center_y),
                    5,
                    (0, 0, 255),
                    -1
                )

                if person_id in self.previous_positions:
                    old_x = self.previous_positions[person_id]

                    # Count each tracked ID only once per processor
                    if person_id not in self.counted_ids:
                        if old_x < line_x <= center_x:
                            with count_lock:
                                self.in_count += 1
                                save_counts(
                                    self.in_count,
                                    self.out_count
                                )
                            self.counted_ids.add(person_id)

                        elif old_x > line_x >= center_x:
                            with count_lock:
                                self.out_count += 1
                                save_counts(
                                    self.in_count,
                                    self.out_count
                                )
                            self.counted_ids.add(person_id)

                self.previous_positions[person_id] = center_x

        # Overlay live counts
        cv2.rectangle(
            annotated,
            (10, 10),
            (260, 110),
            (15, 25, 40),
            -1
        )

        cv2.putText(
            annotated,
            f"IN: {self.in_count}",
            (25, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            f"OUT: {self.out_count}",
            (25, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 165, 255),
            2
        )

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("⚙️ Settings")
    st.write("**Model:** YOLO11n")
    st.write("**Tracker:** ByteTrack")
    st.write("**Camera:** Live webcam")

    st.success("Left → Right = IN")
    st.error("Right → Left = OUT")

    st.caption(
        "Counts are saved to counts.json after each crossing."
    )

    if st.button("Reset All Counts", type="primary"):
        with count_lock:
            save_counts(0, 0)

        st.session_state.final_in = 0
        st.session_state.final_out = 0
        st.warning(
            "Saved totals reset. Stop and restart the camera "
            "if it is currently running."
        )


# ---------------- LIVE CAMERA ----------------

st.subheader("📹 Live Webcam")

st.write(
    "Camera start karo. Person yellow line cross karega "
    "to IN/OUT count update hoga."
)

ctx = webrtc_streamer(
    key="person-live-counter",
    video_processor_factory=PersonCounter,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)


# ---------------- LIVE ANALYTICS ----------------

st.subheader("📊 Total Analytics")


@st.fragment(run_every="1s")
def update_dashboard():
    processor = ctx.video_processor

    # Read the persisted totals, including after camera stops
    with count_lock:
        saved_now = load_counts()

    # If the processor is running, show its latest totals
    if processor is not None:
        current_in = processor.in_count
        current_out = processor.out_count

        # A reset from the sidebar takes effect on the next
        # camera restart; avoid replacing a fresh reset with
        # an older processor's counts.
        st.session_state.final_in = max(
            saved_now["in_count"], current_in
        )
        st.session_state.final_out = max(
            saved_now["out_count"], current_out
        )
    else:
        st.session_state.final_in = saved_now["in_count"]
        st.session_state.final_out = saved_now["out_count"]

    in_count = st.session_state.final_in
    out_count = st.session_state.final_out
    total = in_count + out_count

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🟢 Total IN</div>
            <div class="metric-value green">{in_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🔴 Total OUT</div>
            <div class="metric-value red">{out_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">👥 Total Crossings</div>
            <div class="metric-value blue">{total}</div>
        </div>
        """, unsafe_allow_html=True)

    if ctx.state.playing:
        st.success("Camera status: LIVE")
    else:
        st.info("Camera status: OFF — saved totals are still shown.")


update_dashboard()

st.caption(
    "VisionTrack AI | Live person counting | "
    "Counts are stored locally in counts.json."
)
