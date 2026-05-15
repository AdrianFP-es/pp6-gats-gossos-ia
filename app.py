import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import json
import h5py
import os

st.set_page_config(page_title="Classificador Gats vs Gossos", layout="centered")
st.title("Classificador de Gossos i Gats")
st.markdown("Puja una imatge i la IA intentarà dir si veu un gos o un gat.")

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def conv2d(x, filters, biases):
    h, w, c = x.shape
    fh, fw, _, nf = filters.shape
    out = np.zeros((h - fh + 1, w - fw + 1, nf))
    for f in range(nf):
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j, f] = np.sum(x[i:i+fh, j:j+fw, :] * filters[:,:,:,f]) + biases[f]
    return relu(out)

def maxpool(x):
    h, w, c = x.shape
    out = np.zeros((h//2, w//2, c))
    for i in range(h//2):
        for j in range(w//2):
            out[i, j, :] = np.max(x[i*2:i*2+2, j*2:j*2+2, :], axis=(0,1))
    return out

def load_weights():
    weights = {}
    with h5py.File("model_gats_gossos.weights.h5", "r") as f:
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                weights[name] = np.array(obj)
        f.visititems(collect)
    return weights

def predict(img_array, weights):
    keys = sorted(weights.keys())
    w_list = [weights[k] for k in keys]
    
    x = img_array[0]
    x = conv2d(x, w_list[0], w_list[1])
    x = maxpool(x)
    x = conv2d(x, w_list[2], w_list[3])
    x = maxpool(x)
    x = x.flatten()
    x = relu(x @ w_list[4] + w_list[5])
    x = sigmoid(x @ w_list[6] + w_list[7])
    return float(x[0])

uploaded_file = st.file_uploader("Puja una imatge en format JPG o PNG", type=["jpg", "jpeg", "png"])

if not os.path.exists("model_gats_gossos.weights.h5"):
    st.error("El model no s'ha trobat.")
elif uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imatge pujada", use_container_width=True)
        img_array = np.expand_dims(np.array(image.resize((100, 100))) / 255.0, axis=0)
        
        with st.spinner("Analitzant..."):
            weights = load_weights()
            prob = predict(img_array, weights)
        
        if prob > 0.5:
            st.success(f"La IA creu que és un **gos** amb un {prob*100:.2f}% de confiança.")
        else:
            st.success(f"La IA creu que és un **gat** amb un {(1-prob)*100:.2f}% de confiança.")
        
        st.info("Recorda: el model és senzill i pot equivocar-se.")
    except UnidentifiedImageError:
        st.error("No s'ha pogut llegir la imatge.")