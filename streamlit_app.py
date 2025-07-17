import streamlit as st
import os
import subprocess
import shutil
import atexit
from ruamel.yaml import YAML
from datetime import datetime
import traceback # Važno za detaljnije ispravljanje grešaka
from PIL import Image

from models.ModelManager import ModelManager

# 📦 Global variables for session
if "temp_file_path" not in st.session_state:
    st.session_state.temp_file_path = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
if "save_video" not in st.session_state:
    st.session_state.save_video = False
if "file_type" not in st.session_state:
    st.session_state.file_type = None
# Inicijalizacija modela u session state, podrazumevana vrednost je DeepSort
if 'tracking_model' not in st.session_state:
    st.session_state.tracking_model = "DeepSort"

# Inicijalizacija ModelManager-a u session_state (lazy load)
if 'model_manager' not in st.session_state:
    st.session_state['model_manager'] = ModelManager()




# 📁 Paths
TMP_FOLDER = "tmp"
OUTPUT_FOLDER = "output_videos"
os.makedirs(TMP_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 🧹 Cleanup function
@atexit.register
def delete_current_temp_file():
    if st.session_state.get("temp_file_path") and os.path.exists(st.session_state.temp_file_path):
        try:
            os.remove(st.session_state.temp_file_path)
            st.session_state.temp_file_path = None
        except Exception as e:
            # U pozadinskom procesu ne prikazujemo st.error
            print(f"Error deleting temp file: {e}")

# Increase upload limit
config_dir = os.path.expanduser("~/.streamlit")
config_file = os.path.join(config_dir, "config.toml")
if not os.path.exists(config_file):
    os.makedirs(config_dir, exist_ok=True)
    with open(config_file, "w") as f:
        f.write("[server]\n")
        f.write("maxUploadSize = 10000\n")

st.set_page_config(page_title="Analiza Videa", layout="wide")
st.title("👁️‍🗨️ Video Analysis for People Counting")
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 * 1024

# ==============================================================================
# 1. PRVO IDE LOGIKA ZA UPLOAD I OBRADU FAJLA
# ==============================================================================

uploaded_file = st.file_uploader(
    "📂 Upload image or video (max 10GB)",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

file_display_area = st.container()

if uploaded_file:
    if st.session_state.current_file_name != uploaded_file.name:
        delete_current_temp_file()
        try:
            for filename in os.listdir(TMP_FOLDER):
                file_path = os.path.join(TMP_FOLDER, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    st.error(f'Error deleting {file_path}: {e}')
        except Exception as e:
            st.error(f'Error cleaning tmp folder: {e}')
        
        if uploaded_file.size > MAX_FILE_SIZE_BYTES:
            st.error("❌ File exceeds maximum size of 10GB.")
            st.stop()

        try:
            temp_file_path = os.path.join(TMP_FOLDER, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.temp_file_path = temp_file_path
            st.session_state.current_file_name = uploaded_file.name
            
            if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
                st.error("❌ Failed to save uploaded file properly")
                st.stop()

        except Exception as e:
            st.error(f"Error saving file: {e}")
            st.stop()

    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
    if file_ext in ['.jpg', '.jpeg', '.png']:
        st.session_state.file_type = 'image'
    elif file_ext in ['.mp4', '.avi', '.mov']:
        st.session_state.file_type = 'video'
    else:
        st.session_state.file_type = None

else:
    if st.session_state.current_file_name:
        delete_current_temp_file()
        st.session_state.current_file_name = None
        st.session_state.file_type = None
        st.rerun()

# ==============================================================================
# 2. PRIKAZ GLAVNOG DELA (SA DROPDOWN-OM) PRE SIDEBAR-A
# Ovo radimo da bi sidebar mogao da reaguje na promenu modela
# ==============================================================================

with file_display_area:
    if st.session_state.file_type == 'image':
        st.image(st.session_state.temp_file_path, caption=f"Uploaded image: {st.session_state.current_file_name}", use_container_width=True)
    elif st.session_state.file_type == 'video':
        # Combobox se iscrtava ovde, a njegova vrednost se čuva u st.session_state.tracking_model
        st.selectbox(
            "Izaberite model za praćenje (tracker):",
            ("DeepSort", "CDETR"),
            key="tracking_model" # Korišćenje ključa automatski čuva stanje
        )
        st.video(st.session_state.temp_file_path)
        st.caption(f"Uploaded video: {st.session_state.current_file_name}")

# ==============================================================================
# 3. ZATIM SE ISCRTAVA SIDEBAR NA OSNOVU AŽURNOG STANJA
# ==============================================================================

with st.sidebar:
    st.header("🎛️ Detection Parameters")
    
    # ===> POCETAK IZMENE: Ažurirana logika za onemogućavanje kontrola <===
    is_video = st.session_state.file_type == 'video'
    is_deepsort = st.session_state.get('tracking_model') == 'DeepSort'
    
    # Kontrole su onemogućene ako NIJE video ILI ako NIJE izabran DeepSort
    disabled_status = not is_video or not is_deepsort
    
    # Prikazivanje odgovarajućeg upozorenja
    if not is_video:
        st.warning("⚠️ Controls are disabled. Upload a video to adjust parameters.")
    elif not is_deepsort:
        st.warning("⚠️ Controls are available only for the DeepSort model.")
    # ===> KRAJ IZMENE <===
    
    top, bottom, left, right = 0, 0, 0, 0
    diag_shift1, diag_shift2 = 0, 0
    radius = 50

    # Sve kontrole sada zavise od jedne promenljive: disabled_status
    perspective = st.radio("Camera Perspective", ['front', 'side', 'worm', 'top'], 
                             disabled=disabled_status,
                             help="Select the camera viewpoint for proper line detection")
    
    if perspective != 'top':  
        st.subheader("Line Adjustment")
        if perspective in ['front', 'worm']:
            top = st.slider("Top line", -300, 300, 0, help="Adjust the upper boundary line", disabled=disabled_status)
            bottom = st.slider("Bottom line", -300, 300, 0, help="Adjust the lower boundary line", disabled=disabled_status)
        elif perspective == 'side':
            left = st.slider("Left line", -300, 300, 0, help="Adjust the left boundary line", disabled=disabled_status)
            right = st.slider("Right line", -300, 300, 0, help="Adjust the right boundary line", disabled=disabled_status)

    st.subheader("Additional Options")
    all_lines = st.checkbox("Show all lines", help="Display all detection lines regardless of perspective", disabled=disabled_status)
    diag = st.checkbox("Use diagonals", help="Enable diagonal line detection", disabled=disabled_status)
    show_boxes = st.checkbox("Show bounding boxes", help="Display detection bounding boxes", disabled=disabled_status)
    st.session_state.save_video = st.checkbox("Save processed video", disabled=disabled_status)

    if diag:
        st.subheader("Diagonal Adjustment")
        diag_shift1 = st.slider("Diagonal 1 & 3", -300, 300, 0, help="Adjust first and third diagonal lines", disabled=disabled_status)
        diag_shift2 = st.slider("Diagonal 2 & 4", -300, 300, 0, help="Adjust second and fourth diagonal lines", disabled=disabled_status)

    st.subheader("Circle Settings")
    circle = st.slider("Number of circles", 0, 10, 0, help="Set number of concentric circles for detection", disabled=disabled_status)
    if circle > 0:
        radius = st.slider("Circle spacing", 10, 300, 50, help="Distance between circles", disabled=disabled_status)

    st.subheader("Advanced Options")
    plot = st.checkbox("Show real-time statistics dashboard", disabled=disabled_status)
    headless = st.checkbox("Run without GUI (headless mode)", disabled=disabled_status)
    reset_stats = st.checkbox("Reset statistics", disabled=disabled_status)

    st.subheader("🕐 Video Range (optional)")
    total_frames = None
    if st.session_state.file_type == 'video':
        import cv2

        cap = cv2.VideoCapture(st.session_state.temp_file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration_sec = total_frames / fps
        cap.release()

        range_sec = st.slider(
            "Select processing range (in seconds):",
            0.0, duration_sec, (0.0, duration_sec),
            step=1.0, format="%0.0f sec"
        )

        # Konvertujemo u frame-ove
        start_frame = int(range_sec[0] * fps)
        end_frame = int(range_sec[1] * fps)

        # Čuvamo u session_state da kasnije koristimo pri obradi
        st.session_state['start_frame'] = start_frame
        st.session_state['end_frame'] = end_frame

    else:
        # Ako nije video - očistimo vrednosti (opciono)
        st.session_state['start_frame'] = None
        st.session_state['end_frame'] = None

# ==============================================================================
# 4. NA KRAJU, DUGMIĆI U GLAVNOM DELU
# ==============================================================================

with file_display_area:
    if st.session_state.file_type == 'image':
        # Izbor modela za obradu slike (novi dropdown)
        st.selectbox(
            "Izaberite model za analizu slike:",
            ("CSRNet", "CDETR"),
            key="image_model"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧠 Process Image"):
                with st.spinner("Processing image..."):
                    image = Image.open(st.session_state.temp_file_path)
                    manager = st.session_state['model_manager']

                    selected_model = st.session_state.get("image_model", "CSRNet")
                    result = manager.predict_image(selected_model, image)

                    count = result['count']
                    overlay_img = result.get('overlay_img') or result.get('frame_with_points')

                # Interpretacija gužve
                if count < 50:
                    crowd_level_label = "Nizak"
                    advisory_message = "Gužva je mala — bezbedno za prolazak."
                elif count < 200:
                    crowd_level_label = "Srednji"
                    advisory_message = "Umerena gužva — budite oprezni."
                else:
                    crowd_level_label = "Visok"
                    advisory_message = "Velika gužva — preporučuje se izbegavanje mesta."

                max_capacity = 1000
                occupancy_percent = min(100, (count / max_capacity) * 100)

                st.markdown(f"## Procena broja ljudi: **{int(count)}** osoba")
                st.markdown(f"### Nivo gužve: **{crowd_level_label}**")
                st.markdown(f"**{advisory_message}**")
                st.markdown(f"Popunjenost: **{occupancy_percent:.1f}%**")

                st.subheader("Mapa gustine (Heatmap)")
                st.image(overlay_img, use_container_width=True)



    elif st.session_state.file_type == 'video':
        if st.button("🔍 Start Analysis", type="primary"):
            if st.session_state.tracking_model == "DeepSort":
                st.info("Starting analysis with DeepSort...")
                
                cmd = [
                    "python3", "../main.py",
                    f"--perspective={perspective}", f"--left={left}", f"--right={right}",
                    f"--top={top}", f"--bottom={bottom}", f"--circle={circle}"
                ]
                
                if diag: cmd.extend([f"--diag_shift1={diag_shift1}", f"--diag_shift2={diag_shift2}", "--diag"])
                if all_lines: cmd.append("--all")
                if show_boxes: cmd.append("--show_boxes")
                if plot: cmd.append("--plot")
                if headless: cmd.append("--headless")
                if reset_stats: cmd.append("--reset_stats")
                if circle > 0: cmd.append(f"--radius={radius}")
                
                output_path = None
                if st.session_state.save_video:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_ext = os.path.splitext(st.session_state.current_file_name)[-1]
                    output_filename = f"{timestamp}{file_ext}"
                    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                    cmd.extend(["--output", output_path])

                yaml_path = "/config/app_config.yaml"
                yaml = YAML()
                try:
                    with open(yaml_path, "r") as f: data = yaml.load(f)
                    if "video" in data:
                        data["video"]["input_path"] = st.session_state.temp_file_path
                        with open(yaml_path, "w") as f: yaml.dump(data, f)
                    else:
                        st.error("❌ 'video:' section not found in app_config.yaml"); st.stop()
                except Exception as e:
                    st.error(f"❌ Config update error: {e}"); st.stop()

                try:
                    with st.spinner("Processing video with DeepSort..."):
                        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        stdout, stderr = process.communicate()

                    if process.returncode == 0:
                        st.success("✅ Analysis completed successfully!")
                        if stdout: st.text_area("Processing Output", stdout, height=200)
                        
                        if st.session_state.save_video and output_path and os.path.exists(output_path):
                            try:
                                with open(output_path, "rb") as f:
                                    st.download_button(
                                        label="💾 Download processed video",
                                        data=f,
                                        file_name=os.path.basename(output_path),
                                        mime="video/mp4"
                                    )
                            except Exception as e:
                                st.error(f"Error creating download button: {e}")
                    else:
                        st.error(f"❌ Analysis failed with error:")
                        st.text_area("Error Log", stderr, height=200)
                        
                except Exception as e:
                    st.error(f"❌ Error during processing: {e}")
                    st.text(traceback.format_exc())
            
            elif st.session_state.tracking_model == "CDETR":
                st.info("Pokrećem CDETR analizu...")
                try:
                    manager = st.session_state['model_manager']

                    progress_bar = st.progress(0)

                    def update_progress(percent):
                        progress_bar.progress(percent, text=f"Processing: {int(percent * 100)}%")

                    with st.spinner("Procesiranje videa sa CDETR..."):
                        start_frame = st.session_state.get('start_frame', 0)
                        end_frame = st.session_state.get('end_frame', None)  # None je fallback ako nema

                        output_path = manager.predict_video(
                            "CDETR",
                            st.session_state.temp_file_path,
                            start_frame=start_frame,
                            end_frame=end_frame,
                            progress_callback=update_progress
                        )
                        progress_bar.empty()

                    st.success("✅ CDETR analiza završena!")

                    # Prikaz dugmeta za download izlaznog videa
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="💾 Download CDETR video",
                            data=f,
                            file_name=os.path.basename(output_path),
                            mime="video/avi"
                        )

                except Exception as e:
                    st.error(f"❌ Greška tokom DETR obrade: {str(e)}")
                    st.text(traceback.format_exc())

    # Prikazivanje poruke ako nema fajla
    if not st.session_state.file_type:
        st.info("📂 Upload an image or video to begin.")
        st.markdown("""
        **Usage Guide:**
        1. Upload an image or video file  
        2. If video, select a tracking model.  
        3. If DeepSort is selected, configure parameters in the sidebar.
        4. Click the analysis button.
        
        **Perspective Guide:**
        - Front: For frontal views (shows top/bottom lines)
        - Side: For side views (shows left/right lines)
        - Worm: For low-angle views
        - Top: For overhead views (shows all lines)
        """)