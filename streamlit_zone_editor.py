import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.set_page_config(layout="wide")
st.title("🎯 Podešavanje Zona: Linije, Dijagonale, Krugovi")

video_file = st.file_uploader("🎥 Izaberi video", type=["mp4", "avi"])

if video_file is not None:
    temp_file = "temp_video.mp4"
    with open(temp_file, "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(temp_file)
    success, frame = cap.read()
    cap.release()

    if not success:
        st.error("❌ Ne mogu učitati frejm.")
    else:
        h, w = frame.shape[:2]
        annotated = frame.copy()

        # === SIDEBAR ===
        st.sidebar.subheader("📏 Aktivne zone")

        use_horizontal = st.sidebar.checkbox("➡ Horizontalne linije", value=True)
        use_vertical = st.sidebar.checkbox("⬆ Uspravne linije", value=True)
        use_diagonal = st.sidebar.checkbox("⤴ Dijagonalne linije", value=True)
        use_circles = st.sidebar.checkbox("⭕Krugovi", value=False)

        # === SLIDERS: Horizontalne ===
        if use_horizontal:
            line_up = st.sidebar.slider("Line UP (plava)", 0, h, int(h * 0.3))
            line_down = st.sidebar.slider("Line DOWN (crvena)", 0, h, int(h * 0.7))
            cv2.line(annotated, (0, line_up), (w, line_up), (255, 0, 0), 3)
            cv2.line(annotated, (0, line_down), (w, line_down), (0, 0, 255), 3)

        # === SLIDERS: Vertikalne ===
        if use_vertical:
            line_left = st.sidebar.slider("Line LEFT (ljubičasta)", 0, w, int(w * 0.3))
            line_right = st.sidebar.slider("Line RIGHT (žuta)", 0, w, int(w * 0.7))
            cv2.line(annotated, (line_left, 0), (line_left, h), (128, 0, 128), 3)
            cv2.line(annotated, (line_right, 0), (line_right, h), (0, 255, 255), 3)

        # === DIJAGONALE ===
        if use_diagonal:
            diag1 = st.sidebar.checkbox("Dijagonala 1 (↘)", value=True)
            diag2 = st.sidebar.checkbox("Dijagonala 2 (↙)", value=True)
            diag3 = st.sidebar.checkbox("Dijagonala 3 (─↘─)", value=False)
            diag4 = st.sidebar.checkbox("Dijagonala 4 (│↘│)", value=False)

            if diag1:
                cv2.line(annotated, (0, 0), (w, h), (0, 200, 0), 2)
            if diag2:
                cv2.line(annotated, (0, h), (w, 0), (0, 165, 255), 2)
            if diag3:
                cv2.line(annotated, (0, h // 2), (w, h // 2), (0, 255, 0), 2)
            if diag4:
                cv2.line(annotated, (w // 2, 0), (w // 2, h), (0, 255, 0), 2)

        # === KRUGOVI ===
        if use_circles:
            cx = st.sidebar.slider("Centar X", 0, w, w // 2)
            cy = st.sidebar.slider("Centar Y", 0, h, h // 2)
            r1 = st.sidebar.slider("Radijus 1", 10, int(min(w, h) // 2), 80)
            r2 = st.sidebar.slider("Radijus 2", 10, int(min(w, h) // 2), 150)

            cv2.circle(annotated, (cx, cy), r1, (255, 100, 0), 2)
            cv2.circle(annotated, (cx, cy), r2, (255, 150, 0), 2)

        # === Prikaz slike sa zonama (kao image)
        st.image(annotated, caption="🖼️ Zona Pregled", channels="BGR")

        # === CRTANJE PRAVIH LINIJA ===
        st.subheader("📏 Crtanje pravih linija (klik-drag)")

        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",  # nema popunjavanja
            stroke_width=3,
            stroke_color="#00FF00",
            background_image=img_pil,
            update_streamlit=True,
            height=h,
            width=w,
            drawing_mode="line",  # ⚠️ OVO JE KLJUČNO!
            key="canvas_line"
        )

        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            st.markdown("📐 **Iscrtane linije (apsolutne koordinate):**")
            for i, obj in enumerate(objects):
                if obj["type"] == "line":
                    left = obj.get("left", 0)
                    top = obj.get("top", 0)
                    x1 = int(left + obj["x1"])
                    y1 = int(top + obj["y1"])
                    x2 = int(left + obj["x2"])
                    y2 = int(top + obj["y2"])
                    st.write(f"🔹 Linija {i + 1}: ({x1}, {y1}) → ({x2}, {y2})")

        # === Konfiguracija ispis ===
        with st.expander("🧾 Zona Konfiguracija (za kod)"):
            st.code(f"""
# Horizontal:
line_up = {locals().get('line_up', 'N/A')}
line_down = {locals().get('line_down', 'N/A')}

# Vertikalno:
line_left = {locals().get('line_left', 'N/A')}
line_right = {locals().get('line_right', 'N/A')}

# Krugovi:
circle_center = ({locals().get('cx', 'N/A')}, {locals().get('cy', 'N/A')})
circle_radii = [{locals().get('r1', 'N/A')}, {locals().get('r2', 'N/A')}]

# Freeform drawing (JSON):
canvas_paths = {canvas_result.json_data if canvas_result.json_data else 'None'}
""", language="python")
