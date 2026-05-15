import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
import json

st.set_page_config(page_title="Classificador Gats vs Gossos", layout="centered")
st.title("Classificador de Gossos i Gats")
st.markdown("Puja una imatge i la IA intentarà dir si veu un gos o un gat.")

uploaded_file = st.file_uploader("Puja una imatge en format JPG o PNG", type=["jpg", "jpeg", "png"])

if not os.path.exists("model_gats_gossos.json") or not os.path.exists("model_gats_gossos.weights.h5"):
    st.error("El model no s'ha trobat.")
else:
    import keras
    with open("model_gats_gossos.json", "r") as f:
        model = keras.models.model_from_json(f.read())
    model.load_weights("model_gats_gossos.weights.h5")

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Imatge pujada", use_container_width=True)

            img_array = np.array(image.resize((100, 100))) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prob = float(model.predict(img_array)[0][0])

            if prob > 0.5:
                st.success(f"La IA creu que és un gos amb un {prob*100:.2f}% de confiança.")
            else:
                st.success(f"La IA creu que és un gat amb un {(1-prob)*100:.2f}% de confiança.")

            st.info("Recorda: el model és senzill i pot equivocar-se.")
        except UnidentifiedImageError:
            st.error("No s'ha pogut llegir la imatge.")