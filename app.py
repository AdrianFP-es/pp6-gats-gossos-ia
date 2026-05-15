import streamlit as st
from tensorflow.keras.models import model_from_json
from PIL import Image, UnidentifiedImageError
import numpy as np
import os

st.set_page_config(
    page_title="Classificador Gats vs Gossos",
    layout="centered"
)

st.title("Classificador de Gossos i Gats")
st.markdown("Puja una imatge i la IA intentarà dir si veu un gos o un gat.")

uploaded_file = st.file_uploader(
    "Puja una imatge en format JPG o PNG",
    type=["jpg", "jpeg", "png"]
)

if not os.path.exists("model_gats_gossos.json") or not os.path.exists("model_gats_gossos.weights.h5"):
    st.error("El model no s'ha trobat. Revisa que els fitxers del model estiguin pujats correctament al repositori.")
else:
    with open("model_gats_gossos.json", "r") as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json)
    model.load_weights("model_gats_gossos.weights.h5")

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Imatge pujada", use_container_width=True)

            image_resized = image.resize((100, 100))
            img_array = np.array(image_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            prob = float(prediction[0][0])

            if prob > 0.5:
                confidence = prob * 100
                st.success(f"La IA creu que és un gos amb un {confidence:.2f}% de confiança.")
            else:
                confidence = (1 - prob) * 100
                st.success(f"La IA creu que és un gat amb un {confidence:.2f}% de confiança.")

            st.info("Recorda: el model és senzill i pot equivocar-se. L'objectiu és entendre el procés d'entrenament i predicció.")

        except UnidentifiedImageError:
            st.error("No s'ha pogut llegir la imatge. Puja un fitxer JPG o PNG vàlid.")
