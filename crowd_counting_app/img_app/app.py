import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import streamlit as st
import matplotlib.pyplot as plt
import io
import logging
from datetime import datetime

# Import CSRNet model
from network.CSRNet import CSRNet

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crowd_counting.log"),
        logging.StreamHandler()
    ]
)

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Korišćenje uređaja: {device}')

# Define preprocessing steps
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# GLOBAL model (lazy load)
model = None


# Lazy load function
def load_model():
    global model
    if model is None:
        logging.info('Učitavanje modela (lazy)...')
        model_local = CSRNet()
        checkpoint = torch.load('weights/weights.pth', map_location=device)
        model_local.load_state_dict(checkpoint)
        model_local.to(device)
        model_local.eval()
        model = model_local
        logging.info('Model uspešno učitan.')
    return model


# Prediction function
def predict_crowd(image):
    model_local = load_model()
    image = image.convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model_local(input_tensor)

    count = output.sum().item()
    logging.info(f'Predikcija broja ljudi: {count:.2f} osoba.')

    # Crowd level
    if count < 50:
        color = 'green'
        crowd_level_label = "Nizak"
        advisory_message = "Gužva je mala — bezbedno za prolazak."
    elif count < 200:
        color = 'orange'
        crowd_level_label = "Srednji"
        advisory_message = "Umerena gužva — budite oprezni."
    else:
        color = 'red'
        crowd_level_label = "Visok"
        advisory_message = "Velika gužva — preporučuje se izbegavanje mesta."

    max_capacity = 1000
    occupancy_percent = min(100, (count / max_capacity) * 100)

    # Density map
    density_map = output.squeeze().cpu().numpy()
    density_map_smoothed = gaussian_filter(density_map, sigma=3)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(density_map_smoothed, cmap='jet', interpolation='bilinear')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return count, crowd_level_label, advisory_message, occupancy_percent, buf


# Streamlit UI
st.title("Brojanje ljudi u gužvi (CSRNet)")
st.write("Otpremite sliku za procenu broja ljudi.")

uploaded_file = st.file_uploader("Izaberite sliku", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Vaša učitana slika", use_container_width=True)

    with st.spinner('Obrada slike...'):
        count, crowd_level_label, advisory_message, occupancy_percent, density_img_buf = predict_crowd(image)

    st.markdown(f"## Procena broja ljudi: **{count:.2f}** osoba")
    st.markdown(f"### Nivo gužve: **{crowd_level_label}**")
    st.markdown(f"**{advisory_message}**")
    st.markdown(f"Popunjenost: **{occupancy_percent:.1f}%**")
    st.image(density_img_buf, caption="Mapa gustine", use_container_width=True)
