import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image  # type: ignore
from ultralytics import YOLO
import shutil
import base64

# ====================== CONFIG ======================
st.set_page_config(
    page_title="Solar Panel Defect Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== BACKGROUND ======================
def set_background(image_path: str):
    """Set background image for Streamlit app."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .main .block-container {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 10px;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set your custom background image here
set_background(r"C:\Users\91902\Desktop\solarpanel_defect_detection\Faulty_solar_panel\Clean\Clean (00).jpg")

# ====================== APP TITLE ======================
st.title("Solar Panel Defect Detection")
st.write(
    """
Upload solar panel images to detect defects and get actionable maintenance recommendations.
You can run **Classification**, **YOLO Object Detection**, or both.
"""
)

# ====================== PATHS ======================
TF_MODEL_PATH = r"C:\Users\91902\Desktop\solarpanel_defect_detection\fine_tuned_solar_model.keras"
YOLO_MODEL_PATH = r"C:\Users\91902\Desktop\solarpanel_defect_detection\runs\detect\solarpanel_defect_detection\weights\best.pt"
CLASS_INDICES_PATH = r"C:\Users\91902\Desktop\solarpanel_defect_detection\class_indices.npy"

IMG_HEIGHT, IMG_WIDTH = 224, 224

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_tf_model():
    try:
        model = tf.keras.models.load_model(TF_MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ TensorFlow model load failed: {e}")
        return None

@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO(YOLO_MODEL_PATH)
        return model
    except Exception as e:
        st.warning(f"⚠ YOLO model load failed: {e}")
        return None

tf_model = load_tf_model()
yolo_model = load_yolo_model()

# ====================== LOAD CLASS LABELS ======================
class_indices = np.load(CLASS_INDICES_PATH, allow_pickle=True).item()
inv_class_indices = {v: k for k, v in class_indices.items()}
class_labels = [inv_class_indices[i] for i in range(len(inv_class_indices))]

# ====================== MAINTENANCE RECOMMENDATIONS ======================
recommendations = {
    "Clean": "Panel is clean. No action required.",
    "Dusty": "Schedule cleaning soon to improve efficiency.",
    "Bird-drop": "Clean the panel to remove droppings; monitor for frequent bird activity.",
    "Electrical-damage": "Inspect wiring and electrical components immediately; repair required.",
    "Physical-Damage": "Check for cracks or broken parts; consider panel replacement or repair.",
    "Snow-Covered": "Remove snow to restore sunlight exposure; monitor during winter season."
}

# ====================== SIDEBAR ======================
st.sidebar.header("Options")
st.sidebar.markdown("Choose which model(s) to run:")
run_classification = st.sidebar.checkbox("Run Classification (TensorFlow)", value=True)
run_yolo = st.sidebar.checkbox("Run YOLO Object Detection", value=True)

st.sidebar.markdown(
    """
**Instructions**  
1. Upload JPG, JPEG, or PNG images of solar panels.  
2. Select which model(s) to run.  
3. View the results and download CSV.
"""
)

# ====================== IMAGE UPLOAD ======================
uploaded_files = st.file_uploader(
    "Choose one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    results = []
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        row_result = {"Image": uploaded_file.name}

        # --- TensorFlow Classification ---
        if run_classification and tf_model:
            img_resized = img.resize((IMG_HEIGHT, IMG_WIDTH))
            x = image.img_to_array(img_resized) / 255.0
            x = np.expand_dims(x, axis=0)
            preds = tf_model.predict(x)
            preds = np.squeeze(preds)
            pred_idx = np.argmax(preds)
            pred_class = class_labels[pred_idx]
            confidence = np.max(preds) * 100

            st.markdown(f"### Classification: **{pred_class}** ({confidence:.2f}%)")
            action = recommendations.get(pred_class, "No recommendation available.")
            st.markdown(f"**Recommendation:** {action}")

            row_result.update({
                "Prediction": pred_class,
                "Confidence (%)": round(confidence, 2),
                "Recommended Action": action,
                **{class_labels[i]: round(preds[i]*100, 2) for i in range(len(class_labels))}
            })

        # --- YOLO Object Detection ---
        if run_yolo and yolo_model:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            img.save(temp_path)
            yolo_results = yolo_model.predict(source=temp_path, conf=0.25)
            yolo_img = yolo_results[0].plot()
            st.image(yolo_img, caption="YOLO Object Detection", use_container_width=True)

            detected_objects = []
            for r in yolo_results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0]) * 100
                    detected_objects.append(f"{r.names[cls_id]} ({conf:.1f}%)")
            row_result["YOLO Detected Objects"] = ", ".join(detected_objects) if detected_objects else "None"

        results.append(row_result)

    # --- Display results table ---
    df_results = pd.DataFrame(results)
    st.markdown("### Prediction Results")
    st.dataframe(df_results)

    # --- CSV Download ---
    csv = df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="solar_panel_predictions.csv",
        mime="text/csv"
    )

    # Clean temp directory
    shutil.rmtree(temp_dir)

else:
    st.info("Please upload at least one image to get predictions.")