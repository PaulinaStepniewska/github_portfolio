import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from openai import OpenAI
import plotly.express as px
import pandas as pd
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from io import BytesIO

# Konfiguracja strony
st.set_page_config(page_title="EmoTrack 🤖", layout="wide")

INPUT_SIZE = (75, 75)

# Model
@st.cache_resource(show_spinner=False)
def load_model():
    model = tf.keras.models.load_model("emotion_model_final.h5", compile=False)
    #model = tf.keras.models.load_model("emotion_cnn_model1.h5", compile=False)
    dummy_input = tf.zeros((1, *INPUT_SIZE, 3))
    model(dummy_input)
    return model

model = load_model()

class_labels = {
    0: 'złość',
    1: 'obrzydzenie',
    2: 'strach',
    3: 'szczęście',
    4: 'neutralność',
    5: 'smutek',
    6: 'zaskoczenie'
}


# Wykrywanie twarzy
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess_image(pil_img):
    img = pil_img.resize(INPUT_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    return arr

# Funkcja do predykcji dla LIME (wejście uint8)
def model_predict(images):
    processed = []
    for img in images:
        img_pil = Image.fromarray(img)
        img_resized = img_pil.resize(INPUT_SIZE)
        arr = np.array(img_resized).astype("float32") / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        processed.append(arr)
    processed = np.array(processed)
    preds = model.predict(processed, verbose=0)
    return preds

# OpenAI
openai_api_key = st.secrets["openai"]["api_key"]
client = OpenAI(api_key=openai_api_key)

def get_chatgpt_recommendation(emotion):
    try:
        prompt = f"""Wykryto emocję: {emotion}.
        Napisz krótką, empatyczną rekomendację (maksymalnie 2-3 zdania), która pomoże komuś w tym stanie emocjonalnym.
        Napisz po polsku, empatycznie, bez imion i oceniania."""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Jesteś wspierającym i empatycznym psychologiem."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Błąd zapytania do ChatGPT:\n\n{e}"

def tell_joke():
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Jesteś zabawnym komikiem."},
            {"role": "user", "content": "Opowiedz bardzo krótki, zabawny żart po polsku, który może poprawić nastrój."}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content

# Nagłówek
st.markdown("""
<div style='text-align: center;'>
    <h1 style='font-size: 64px;'>EmoTrack 🤖✨</h1>
    <p style='font-size: 18px;'>Wgraj zdjęcie twarzy – model wykryje emocję i pokaże personalizowane rekomendacje.</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Upload
st.markdown("### 📤 Wgraj zdjęcie")
uploaded_file = st.file_uploader("Obraz (JPG/PNG)", type=["jpg", "png"])

if uploaded_file:
    original_pil = Image.open(uploaded_file).convert("RGB")
    original_cv = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
    faces = face_cascade.detectMultiScale(original_cv, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.warning("Nie wykryto twarzy na zdjęciu.")
    else:
        x, y, w, h = faces[0]
        face_img = original_cv[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_arr = preprocess_image(face_pil)

        preds = model.predict(np.expand_dims(face_arr, 0), verbose=0)[0]
        pred_class = int(np.argmax(preds))
        confidence = float(np.max(preds))
        emotion_name = class_labels[pred_class]
        recommendation = get_chatgpt_recommendation(emotion_name)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 📷 Wgrane zdjęcie")
            st.image(original_pil, caption="Wgrane zdjęcie", width=250)

        with col2:
            st.markdown("### 🧑‍🦲🔍 Wykryta twarz")
            st.image(face_pil.resize((face_pil.width // 2, face_pil.height // 2)), width=80)

            st.markdown("### 🧠 Wynik klasyfikacji")
            st.markdown(f"**Emocja:** {emotion_name}")
            st.markdown(f"**Pewność:** {confidence*100:.2f}%")

            st.markdown("### 💡 Rekomendacja")
            st.info(recommendation)

            if emotion_name == "smutek":
                st.markdown("### 🧩 Żart na poprawę humoru")
                try:
                    st.success(tell_joke())
                except Exception as e:
                    st.error(f"Coś poszło nie tak: {e}")

        st.markdown("### 🔎 Inne emocje czające się w tle")
        emotions = list(class_labels.values())
        df = pd.DataFrame({
            'emocje': emotions,
            'prawdopodobieństwa': preds,
        })
        df['text'] = (df['prawdopodobieństwa'] * 100).map('{:.1f}%'.format)

        fig = px.bar(
            df, x='emocje', y='prawdopodobieństwa',
            color=df.index == pred_class,
            color_discrete_map={True: 'green', False: 'lightgray'},
            labels={'prawdopodobieństwa': 'prawdopodobieństwo'},
            text='text'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            xaxis_tickangle=-30,
            yaxis=dict(range=[0, 1], showticklabels=False),
            plot_bgcolor='white',
            showlegend=False,
            margin=dict(t=10, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🔍 Wyjaśnienie predykcji (LIME)"):
            try:
                explainer = lime_image.LimeImageExplainer()
                face_uint8 = (face_arr * 255).astype(np.uint8)

                explanation = explainer.explain_instance(
                    face_uint8,
                    classifier_fn=model_predict,
                    top_labels=1,
                    hide_color=0,
                    num_samples=1000
                )

                top_label = explanation.top_labels[0]

                temp, mask = explanation.get_image_and_mask(
                    top_label,
                    positive_only=False,
                    num_features=10,
                    hide_rest=False
                )
                
                plt.rcParams['figure.dpi'] = 100
                fig, ax = plt.subplots(figsize=(4,4), dpi=100)
                ax.imshow(mark_boundaries(temp / 255, mask))
                ax.axis('off')
                buf = BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                buf.seek(0)
                st.image(buf, caption=f"LIME - wpływ cech na emocję: {class_labels[top_label]}", width=250)



                st.markdown(f"""
                <div style='text-align: center; font-size: 14px;'>
                Powyższa heatmapa przedstawia lokalne ważności obszarów obrazu wygenerowane przez 
                LIME, wskazując, które fragmenty twarzy miały największy wpływ na zaklasyfikowanie emocji jako "{class_labels[top_label]}".

                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div style='text-align: center; font-size: 14px;'>
                <b>Legenda:</b><br>
                🟢 – regiony, które miały największy wpływ na wykrycie tej emocji<br>
                🔴 – mniej istotne lub wpływające w inną stronę
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Błąd podczas generowania wyjaśnienia LIME: {e}")


st.markdown("---")
st.caption("© 2025 EmoTrack App by Neuronauci. All rights reserved.")
