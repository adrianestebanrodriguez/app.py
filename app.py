import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
import streamlit as st

# --- Configuración ---
DATA_DIR = Path("data")
PHOTOS_DIR = DATA_DIR / "photos"
MODEL_PATH = Path("vision_model_params.npz")
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

# --- Cargar modelo ---
MODEL = None
if MODEL_PATH.exists():
    npz = np.load(MODEL_PATH, allow_pickle=True)
    W = npz["W"]
    b = npz["b"]
    CLASSES = npz["classes"].tolist()
    MODEL = dict(W=W, b=b, classes=CLASSES)
    st.sidebar.success(f"Modelo cargado con clases: {CLASSES}")
else:
    st.sidebar.warning("⚠️ No se encontró modelo. Se usará heurística de respaldo.")

# --- Función de features ---
def extract_features(img: Image.Image):
    img = img.convert("RGB").resize((128,128))
    arr = np.array(img).astype(np.float32)/255.0
    r_mean, g_mean, b_mean = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
    r_std, g_std, b_std = arr[:,:,0].std(), arr[:,:,1].std(), arr[:,:,2].std()
    gray = np.dot(arr, [0.2989, 0.5870, 0.1140])
    gray_mean, gray_std = gray.mean(), gray.std()
    dark_pct, bright_pct = (gray < 0.2).mean(), (gray > 0.8).mean()
    return np.array([r_mean,g_mean,b_mean,r_std,g_std,b_std,gray_mean,gray_std,dark_pct,bright_pct], dtype=np.float32)

# --- Función de inferencia ---
def infer_photo(photo_id, img: Image.Image):
    inferred_at = datetime.utcnow().isoformat()
    details = {}
    score = None
    try:
        feats = extract_features(img)
        logits = feats.dot(MODEL['W'].T) + MODEL['b']
        exp = np.exp(logits - logits.max())
        probs = (exp / exp.sum()).tolist()
        pred_idx = int(np.argmax(probs))
        pred_class = MODEL['classes'][pred_idx]
        score = float(round(float(probs[pred_idx]),3))
        details = {"method":"vision-model", "probs":probs, "classes":MODEL["classes"]}
    except Exception as e:
        # Fallback heurístico
        base = 0.2
        if any(k in photo_id.lower() for k in ["insect","sucio","basura","roedor"]):
            base += 0.5
        score = min(1.0, round(base + np.random.random()*0.4, 3))
        pred_class = "desconocido"
        details = {"method":"fallback", "error":str(e)}

    return {
        "photo_id": photo_id,
        "pred_class": pred_class,
        "score": score,
        "details": details,
        "inferred_at": inferred_at
    }

# --- Interfaz Streamlit ---
st.title("🍽️ Riesgo de Intoxicación Alimentaria - Medellín")
st.markdown("Sube una foto de un restaurante para evaluar el **riesgo de higiene**.")

uploaded_file = st.file_uploader("📷 Subir foto", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    photo_id = uploaded_file.name.split(".")[0]
    img = Image.open(uploaded_file)
    
    # Guardar foto
    save_path = PHOTOS_DIR / uploaded_file.name
    img.save(save_path)

    # Mostrar
    st.image(img, caption="Foto subida", use_column_width=True)

    # Inferencia
    result = infer_photo(photo_id, img)

    st.subheader("📊 Resultado de la evaluación")
    st.write(f"**Clase predicha:** {result['pred_class']}")
    st.write(f"**Riesgo estimado:** {result['score']*100:.1f}%")

    st.json(result["details"])
