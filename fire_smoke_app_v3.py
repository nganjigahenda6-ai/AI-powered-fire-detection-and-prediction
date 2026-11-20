"""
Fire & Smoke Detection - Streamlit App
Modes: Image, Video (watch live or fast analysis), Live Webcam.
Tuned for PCs without a GPU: the picture on screen never waits for the model.
Run with: streamlit run fire_smoke_app.py
"""

import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

APP_VERSION = "v4 - CPU-friendly (direct low-delay camera, real-speed video, fast analysis)"
DEFAULT_MODEL_PATH = r"C:\Users\User\Desktop\DATASETS\FINAL_MODEL\best.pt"
MAX_DISPLAY_WIDTH = 800
BOX_COLORS = {"fire": (0, 0, 255), "smoke": (200, 200, 200)}  # BGR

st.set_page_config(page_title="Fire & Smoke Detector", layout="wide")
st.title("🔥 Fire & Smoke Detection")
st.caption("YOLOv8 fire/smoke detector — image, video, or live webcam")


@st.cache_resource
def load_model(path: str) -> YOLO:
    return YOLO(path)


# ---------------- SIDEBAR ----------------
model_path = st.sidebar.text_input("Model path (.pt)", DEFAULT_MODEL_PATH, key="model_path")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05, key="conf")
mode = st.sidebar.radio("Mode", ["Image", "Video", "Live Webcam"], key="mode")

# Defaults (only used by Video / Live modes)
window_size, confirm_count = 10, 5
infer_size = 416
video_mode = "Watch live (real speed)"
detect_every = 3
sample_every = 5
cam_method = "Direct (lowest delay)"
cam_index = 0

if mode != "Image":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Confirmation")
    st.sidebar.caption(
        "A class is only CONFIRMED once it appears in enough of the recent "
        "analyzed frames. This filters out one-off flickers or glare."
    )
    window_size = st.sidebar.slider("Rolling window (analyzed frames)", 3, 30, 10, 1, key="window")
    confirm_count = st.sidebar.slider(
        "Frames required to confirm", 1, window_size, min(5, window_size), 1, key="confirm"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Speed")
    st.sidebar.caption(
        "Without a GPU the model is slower than the video, so the video is "
        "never held back by the model."
    )
    infer_size = st.sidebar.select_slider(
        "Detection size (smaller = faster)", options=[320, 416, 480, 640], value=416, key="infer_size"
    )
    if mode == "Live Webcam":
        cam_method = st.sidebar.radio(
            "Camera method", ["Direct (lowest delay)", "Browser (WebRTC)"], key="cam_method"
        )
        if cam_method.startswith("Direct"):
            cam_index = int(st.sidebar.number_input(
                "Camera number (0 = default camera)", min_value=0, max_value=5, value=0, step=1, key="cam_index"
            ))
    if mode == "Video":
        video_mode = st.sidebar.radio(
            "Video mode",
            ["Watch live (real speed)", "Fast analysis (report only)"],
            key="video_mode",
        )
        if video_mode.startswith("Watch live"):
            detect_every = st.sidebar.slider("Detect every N frames", 1, 10, 3, 1, key="detect_every")
        else:
            sample_every = st.sidebar.slider("Analyze every Nth frame", 1, 30, 5, 1, key="sample_every")

st.sidebar.markdown("---")
st.sidebar.caption(f"App version: {APP_VERSION}")

if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

model = load_model(model_path)
CLASS_NAMES = model.names  # {0: 'fire', 1: 'smoke'}
CLASS_LIST = list(CLASS_NAMES.values())


# ---------------- HELPERS ----------------
def summarize_detections(result) -> dict:
    boxes = result.boxes
    counts = {}
    if boxes is not None and len(boxes) > 0:
        for c in boxes.cls.cpu().numpy():
            name = CLASS_NAMES[int(c)]
            counts[name] = counts.get(name, 0) + 1
    return counts


def extract_boxes(result) -> list:
    """(x1, y1, x2, y2, class_name, confidence) for every box in a YOLO result,
    so the boxes can be redrawn on later frames without re-running the model."""
    out = []
    if result.boxes is not None:
        for b in result.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            out.append((x1, y1, x2, y2, CLASS_NAMES[int(b.cls[0])], float(b.conf[0])))
    return out


class DetectionSmoother:
    """Rolling window of which classes were present per analyzed frame. A class
    is CONFIRMED once it appears in enough of the recent frames."""

    def __init__(self, classes, window: int, required: int):
        self.history = {c: deque(maxlen=window) for c in classes}
        self.required = required

    def update(self, present_classes: set) -> dict:
        """Returns {class_name: (count_in_window, is_confirmed)}."""
        status = {}
        for name, hist in self.history.items():
            hist.append(1 if name in present_classes else 0)
            count = sum(hist)
            status[name] = (count, count >= self.required)
        return status


def draw_boxes(frame_bgr: np.ndarray, boxes: list) -> np.ndarray:
    for x1, y1, x2, y2, name, conf in boxes:
        color = BOX_COLORS.get(name, (0, 255, 0))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame_bgr, f"{name} {conf:.2f}", (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
    return frame_bgr


def draw_confirmation_banner(frame_bgr: np.ndarray, status: dict) -> np.ndarray:
    """Overlays a banner at the top of the frame for any confirmed class."""
    confirmed = [name for name, (count, ok) in status.items() if ok]
    if not confirmed:
        return frame_bgr
    label = "  +  ".join(f"{name.upper()} CONFIRMED" for name in confirmed)
    color = (0, 0, 255) if "fire" in confirmed else (0, 165, 255)  # red if fire, else orange
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (frame_bgr.shape[1], 45), color, -1)
    frame_bgr = cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0)
    cv2.putText(
        frame_bgr, label, (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return frame_bgr


def annotate_frame(frame_bgr: np.ndarray, boxes: list, status: dict) -> np.ndarray:
    """Returns a NEW image with boxes and the confirmation banner drawn on it."""
    out = draw_boxes(frame_bgr.copy(), boxes)
    return draw_confirmation_banner(out, status)


def fit_width(img: np.ndarray, max_width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    return cv2.resize(img, (max_width, int(h * max_width / w)))


_use_stretch = [True]


def show_image(target, img_rgb, **kwargs):
    """target.image(...) at full width. Uses the newer width="stretch" option and
    falls back to use_container_width on older Streamlit versions."""
    if _use_stretch[0]:
        try:
            target.image(img_rgb, width="stretch", **kwargs)
            return
        except Exception:
            _use_stretch[0] = False
    target.image(img_rgb, use_container_width=True, **kwargs)


def format_time(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def get_fps(cap) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps == fps and fps > 0 else 25.0


def analyze_video(path, yolo_model, conf, imgsz, sample_every, window, required,
                  max_snapshots=8, progress_cb=None) -> dict:
    """Runs the model on every Nth frame as fast as possible (no playback) and
    returns when fire / smoke were CONFIRMED, plus a few snapshots."""
    cap = cv2.VideoCapture(path)
    fps = get_fps(cap)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    smoother = DetectionSmoother(CLASS_LIST, window, required)
    intervals = {name: [] for name in CLASS_LIST}
    open_start = {name: None for name in CLASS_LIST}
    snapshots = []
    idx = 0
    analyzed = 0
    last_t = 0.0
    started = time.time()
    try:
        while cap.grab():  # grab() skips cheaply; retrieve() only on frames we analyze
            if idx % sample_every == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                result = yolo_model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
                boxes = extract_boxes(result)
                status = smoother.update({b[4] for b in boxes})
                t = idx / fps
                last_t = t
                for name, (count, confirmed) in status.items():
                    if confirmed and open_start[name] is None:
                        open_start[name] = t
                        if len(snapshots) < max_snapshots:
                            snapshots.append((name, t, annotate_frame(frame, boxes, status)))
                    elif not confirmed and open_start[name] is not None:
                        intervals[name].append((open_start[name], t))
                        open_start[name] = None
                analyzed += 1
                if progress_cb and analyzed % 3 == 0:
                    progress_cb(idx, total)
            idx += 1
    finally:
        cap.release()

    for name, start in open_start.items():  # still confirmed when the video ended
        if start is not None:
            intervals[name].append((start, max(last_t, start)))

    return {
        "intervals": intervals,
        "snapshots": snapshots,
        "analyzed": analyzed,
        "frames": idx,
        "duration": idx / fps,
        "elapsed": time.time() - started,
    }


def limit_torch_threads():
    """Leave a few CPU cores free for the camera, the browser and Streamlit, so the
    model cannot starve them (that starvation is what makes the picture lag)."""
    try:
        import torch
        torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4) - 2)))
    except Exception:
        pass


def open_camera(index: int, width: int = 640, height: int = 480):
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
    return None


class CameraStream:
    """Reads the camera in one thread (keeping ONLY the newest frame, so the picture
    is never delayed) and runs the model in a second thread on that newest frame."""

    def __init__(self, index, yolo_model, conf, imgsz, window, required):
        self.index, self.model, self.conf, self.imgsz = index, yolo_model, conf, imgsz
        self.smoother = DetectionSmoother(CLASS_LIST, window, required)
        self.cap = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.latest = None
        self.frame_id = 0
        self.last_frame_time = time.time()
        self.boxes = []
        self.status = {name: (0, False) for name in CLASS_LIST}
        self.det_ms = 0.0
        self.det_count = 0
        self.started = time.time()
        self.threads = []

    def open(self) -> bool:
        self.cap = open_camera(self.index)
        return self.cap is not None

    def start(self):
        self.started = self.last_frame_time = time.time()
        self.threads = [
            threading.Thread(target=self._read, daemon=True),
            threading.Thread(target=self._detect, daemon=True),
        ]
        for t in self.threads:
            t.start()

    def _read(self):
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest = frame
                self.frame_id += 1
                self.last_frame_time = time.time()

    def _detect(self):
        limit_torch_threads()
        done_id = -1
        while not self.stop_event.is_set():
            with self.lock:
                frame, fid = self.latest, self.frame_id
            if frame is None or fid == done_id:
                time.sleep(0.005)
                continue
            done_id = fid
            t0 = time.time()
            try:
                result = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
                boxes = extract_boxes(result)
                status = self.smoother.update({b[4] for b in boxes})
            except Exception:
                continue
            with self.lock:
                self.boxes, self.status = boxes, status
                self.det_ms = (time.time() - t0) * 1000
                self.det_count += 1

    def snapshot(self):
        with self.lock:
            frame = None if self.latest is None else self.latest.copy()
            return frame, self.frame_id, list(self.boxes), dict(self.status), self.det_ms, self.det_count

    def stop(self):
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=1.5)
        if self.cap is not None:
            self.cap.release()


def run_direct_camera():
    st.info(
        "Direct camera mode: the app opens your camera itself, so the picture has almost "
        "no delay. Tick the box to start the camera, untick it to stop. Changing any "
        "setting restarts the camera for a moment."
    )
    if not st.checkbox("Start camera", key="cam_on"):
        return

    stream = CameraStream(cam_index, model, conf_threshold, infer_size, window_size, confirm_count)
    if not stream.open():
        st.error(
            "Could not open the camera. Close other apps that use it (Zoom, Teams, other "
            "browser tabs), check Windows Settings > Privacy & security > Camera, or try "
            "another camera number or the Browser (WebRTC) method."
        )
        return

    placeholder = st.empty()
    info = st.empty()
    stream.start()
    shown_id, shown_dets, last_info = -1, -1, 0.0
    try:
        while True:
            frame, fid, boxes, status, det_ms, det_count = stream.snapshot()
            now = time.time()
            if frame is None:
                if now - stream.started > 5:
                    st.error("The camera opened but is not sending pictures.")
                    break
                time.sleep(0.02)
                continue
            if now - stream.last_frame_time > 3:
                st.warning("The camera stopped sending pictures.")
                break
            if fid == shown_id and det_count == shown_dets:
                time.sleep(0.005)  # nothing new to show yet
                continue
            shown_id, shown_dets = fid, det_count

            out = fit_width(annotate_frame(frame, boxes, status), MAX_DISPLAY_WIDTH)
            show_image(placeholder, cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB")
            if now - last_info > 1:
                info.caption(
                    f"Last analysis took {det_ms:.0f} ms | "
                    f"{det_count / max(now - stream.started, 0.001):.1f} analyses per second"
                )
                last_info = now
            time.sleep(1 / 30)
    finally:
        stream.stop()


def run_watch_live(path: str):
    """Plays the video at its real speed: frames the model cannot keep up with
    are skipped, and the last known boxes are redrawn on the frames in between."""
    cap = cv2.VideoCapture(path)
    stframe = st.empty()
    status_box = st.empty()
    stats_box = st.empty()
    stop_button = st.button("Stop processing")

    smoother = DetectionSmoother(CLASS_LIST, window_size, confirm_count)
    video_fps = get_fps(cap)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_idx = 0     # position in the video
    shown = 0         # frames actually displayed
    fire_total = 0
    smoke_total = 0
    last_boxes = []
    last_status = {name: (0, False) for name in CLASS_LIST}
    start_time = time.time()

    try:
        while cap.isOpened():
            # Skip the frames we are already late for, so playback matches real time
            target_idx = int((time.time() - start_time) * video_fps)
            while frame_idx < target_idx:
                if not cap.grab():
                    break
                frame_idx += 1

            ret, frame = cap.read()
            if not ret or stop_button:
                break
            frame_idx += 1
            shown += 1

            if shown == 1 or shown % detect_every == 0:
                result = model.predict(frame, conf=conf_threshold, imgsz=infer_size, verbose=False)[0]
                last_boxes = extract_boxes(result)
                counts = summarize_detections(result)
                last_status = smoother.update(set(counts.keys()))
                fire_total += counts.get("fire", 0)
                smoke_total += counts.get("smoke", 0)

            annotated = fit_width(annotate_frame(frame, last_boxes, last_status), MAX_DISPLAY_WIDTH)
            show_image(stframe, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

            status_box.write(
                "Confirmation window: "
                + " | ".join(f"{n}: {c}/{window_size}" for n, (c, ok) in last_status.items())
            )
            progress = f"{frame_idx}/{total_frames}" if total_frames else str(frame_idx)
            stats_box.write(
                f"Video frame {progress} | boxes counted on analyzed frames - "
                f"fire: {fire_total}, smoke: {smoke_total}"
            )
    finally:
        cap.release()
    st.success("Video finished.")


def run_fast_analysis(path: str):
    progress = st.progress(0.0, text="Analyzing video...")

    def on_progress(idx, total):
        if total:
            progress.progress(min(idx / total, 1.0), text=f"Analyzing... frame {idx}/{total}")

    report = analyze_video(
        path, model, conf_threshold, infer_size, sample_every,
        window_size, confirm_count, progress_cb=on_progress,
    )
    progress.progress(1.0, text="Done")

    st.subheader("Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Video length", format_time(report["duration"]))
    c2.metric("Frames analyzed", report["analyzed"])
    c3.metric("Analysis time", f"{report['elapsed']:.0f} s")

    any_event = False
    for name in CLASS_LIST:
        spans = report["intervals"][name]
        if spans:
            any_event = True
            st.write(
                f"**{name.upper()} confirmed:** "
                + ", ".join(f"{format_time(a)} - {format_time(b)}" for a, b in spans)
            )
    if not any_event:
        st.info("No fire or smoke was confirmed in this video.")

    if report["snapshots"]:
        st.subheader("Snapshots (first moment each was confirmed)")
        cols = st.columns(min(len(report["snapshots"]), 4))
        for i, (name, t, img) in enumerate(report["snapshots"]):
            show_image(
                cols[i % len(cols)],
                cv2.cvtColor(fit_width(img, 480), cv2.COLOR_BGR2RGB),
                caption=f"{name.upper()} confirmed at {format_time(t)}",
            )


# ---------------- IMAGE MODE ----------------
if mode == "Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not read this file as an image. Please upload a JPG, PNG or BMP.")
        else:
            results = model.predict(img_bgr, conf=conf_threshold, verbose=False)
            annotated_bgr = results[0].plot()

            show_image(st, cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), caption="Detections")

            counts = summarize_detections(results[0])
            if counts:
                st.write("Detections:", counts)
            else:
                st.write("No fire or smoke detected.")

# ---------------- VIDEO MODE ----------------
elif mode == "Video":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tfile.write(uploaded.read())
        tfile.close()
        try:
            if video_mode.startswith("Watch live"):
                run_watch_live(tfile.name)
            else:
                run_fast_analysis(tfile.name)
        finally:
            try:
                os.unlink(tfile.name)
            except OSError:
                pass

# ---------------- LIVE WEBCAM MODE ----------------
elif mode == "Live Webcam" and cam_method.startswith("Direct"):
    run_direct_camera()

elif mode == "Live Webcam":
    st.info(
        "Browser camera (WebRTC). Click START below and allow camera access. "
        "If the picture is delayed, switch to the Direct camera method in the sidebar."
    )

    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    except ImportError:
        st.error(
            "Live webcam mode needs two extra packages. In your terminal run:\n\n"
            "pip install streamlit-webrtc av\n\n"
            "then restart the app."
        )
        st.stop()

    class YOLOProcessor(VideoProcessorBase):
        """recv() draws the latest known detections on every camera frame and
        returns immediately. The model runs in a background thread on the newest
        frame; frames it cannot keep up with are simply dropped."""

        def __init__(self):
            self.conf = conf_threshold
            self.imgsz = infer_size
            self.smoother = DetectionSmoother(CLASS_LIST, window_size, confirm_count)
            self._lock = threading.Lock()
            self._pending = None
            self._boxes = []
            self._status = {name: (0, False) for name in CLASS_LIST}
            self._last_recv = time.time()
            self._worker = None
            self._start_worker()

        def _start_worker(self):
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()

        def _run(self):
            limit_torch_threads()
            # Exits by itself once the camera stream has stopped sending frames
            while time.time() - self._last_recv < 15:
                with self._lock:
                    frame, self._pending = self._pending, None
                if frame is None:
                    time.sleep(0.01)
                    continue
                try:
                    result = model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
                    boxes = extract_boxes(result)
                    status = self.smoother.update({b[4] for b in boxes})
                except Exception:
                    continue
                with self._lock:
                    self._boxes, self._status = boxes, status

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self._last_recv = time.time()
            if not self._worker.is_alive():
                self._start_worker()
            with self._lock:
                self._pending = img.copy()  # newest frame for the model
                boxes, status = list(self._boxes), dict(self._status)
            out = annotate_frame(img, boxes, status)
            return av.VideoFrame.from_ndarray(out, format="bgr24")

    RTC_CONFIG = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key="fire-smoke-live",
        video_processor_factory=YOLOProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 15}},
            "audio": False,
        },
        async_processing=True,
    )
    if ctx and ctx.video_processor:  # confidence / size sliders apply live
        ctx.video_processor.conf = conf_threshold
        ctx.video_processor.imgsz = infer_size
