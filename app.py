import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

IMG_SIZE = (224, 224)

st.set_page_config(page_title="Waste Segregation AI", page_icon="♻️")

st.markdown("""
<style>
.stApp { background-color: #FAF7F2; }
html, body, [class*="css"], .stMarkdown, p, label, span { color: #4A3B2A; }
h1, h2, h3 { color: #5C4630 !important; font-weight: 700; }
[data-testid="stHeader"] { background: #FAF7F2; }
[data-testid="stFileUploaderDropzone"], [data-testid="stCameraInput"] {
    background: #F1E7DA; border: 1.5px dashed #D9C3A5; border-radius: 16px;
}
button, [data-testid="stBaseButton-secondary"] {
    background: #E9D8C2 !important; color: #4A3B2A !important;
    border: 1px solid #D9C3A5 !important; border-radius: 12px !important;
}
button:hover { background: #DFC9AE !important; }
img { border-radius: 16px; }
.card { background: #F5ECE0; border: 1px solid #E6D6C1; border-radius: 16px;
        padding: 14px 18px; margin: 10px 0; }
.card .label { font-size: 12px; letter-spacing: 1px; text-transform: uppercase; color: #9A7B5A; }
.card .value { font-size: 17px; margin-top: 4px; line-height: 1.5; }
.card.big .value { font-size: 26px; font-weight: 700; color: #5C4630; }
.card .conf { font-size: 15px; font-weight: 500; color: #9A7B5A; margin-left: 6px; }
.card.warn { background: #F8EDDC; border-color: #E3C9A0; }
.hero { background: #F1E7DA; border-radius: 20px; padding: 18px 20px; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

RECYCLING_GUIDE = {
    "battery":     {"bin": "Hazardous / E-waste (red)", "how": "Never put in normal trash. Drop at a battery collection point or e-waste centre.", "tip": "Tape the terminals of lithium batteries before drop-off."},
    "biological":  {"bin": "Wet waste (green)", "how": "Compost it at home or send to a municipal composting unit.", "tip": "Keep it free of plastic and packaging."},
    "brown-glass": {"bin": "Glass (white/blue)", "how": "Rinse, remove the cap, and send to a glass recycler. Glass can be recycled endlessly.", "tip": "Do not mix with ceramics or window glass."},
    "green-glass": {"bin": "Glass (white/blue)", "how": "Rinse, remove the cap, and send to a glass recycler.", "tip": "Separating by colour improves recycling quality."},
    "white-glass": {"bin": "Glass (white/blue)", "how": "Rinse, remove the cap, and send to a glass recycler.", "tip": "Broken glass: wrap in paper and label it."},
    "glass":       {"bin": "Glass (white/blue)", "how": "Rinse, remove the cap, and send to a glass recycler.", "tip": "Do not mix with ceramics or window glass."},
    "cardboard":   {"bin": "Dry waste (blue)", "how": "Flatten the boxes, keep them dry, and send to paper recycling.", "tip": "Greasy or wet cardboard (pizza box) cannot be recycled."},
    "paper":       {"bin": "Dry waste (blue)", "how": "Keep it dry and clean, and send to paper recycling.", "tip": "Shred sensitive documents before recycling."},
    "plastic":     {"bin": "Dry waste (blue)", "how": "Rinse, remove the cap, crush it, and send to a plastic recycler. Check the resin code (1 to 7).", "tip": "Single-use plastic bags and wrappers often need special collection."},
    "metal":       {"bin": "Dry waste (blue)", "how": "Rinse cans and send to a metal scrap or recycling unit.", "tip": "Aluminium and steel can be recycled repeatedly."},
    "clothes":     {"bin": "Textile donation", "how": "Donate wearable clothes. Send worn-out fabric to textile recyclers or upcycle it.", "tip": "Wash before donating."},
    "shoes":       {"bin": "Donation / textile recycling", "how": "Donate usable pairs. Worn-out shoes go to shoe-recycling programmes.", "tip": "Tie pairs together."},
    "trash":       {"bin": "Residual waste (black)", "how": "Non-recyclable items go to landfill or waste-to-energy.", "tip": "Reduce and reuse first to cut this pile."},
}


@st.cache_resource
def load_model():
    model = keras.models.load_model("waste_classifier.keras", compile=False)
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names


model, class_names = load_model()

st.markdown("""
<div class="hero">
<h1 style="margin:0">♻️ Waste Segregation AI</h1>
<p style="margin:6px 0 0 0">Take or upload a photo of a waste item. The AI tells you what it is, which bin to use, and how to recycle it.</p>
</div>
""", unsafe_allow_html=True)

source = st.radio("Choose input", ["Upload a photo", "Use camera"], horizontal=True)
file = st.file_uploader("Photo", type=["jpg", "jpeg", "png"]) if source == "Upload a photo" else st.camera_input("Take a photo")

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

    arr = np.expand_dims(np.array(img.resize(IMG_SIZE), dtype="float32"), 0)   # raw 0-255, model scales it
    probs = model.predict(arr, verbose=0)[0]
    top3 = np.argsort(probs)[::-1][:3]
    label, conf = class_names[top3[0]], float(probs[top3[0]])

    info = RECYCLING_GUIDE.get(label, {"bin": "Check local rules", "how": "No guidance stored.", "tip": ""})

    def card(title, text, extra=""):
        st.markdown(f'<div class="card {extra}"><div class="label">{title}</div><div class="value">{text}</div></div>', unsafe_allow_html=True)

    name = label.replace("-", " ").title()
    card("Waste type", f'{name}<span class="conf">{conf * 100:.1f}% sure</span>', "big")
    if conf < 0.60:
        card("Not sure", "Try another angle or better lighting.", "warn")
    card("Which bin", info["bin"])
    card("How to recycle", info["how"])
    if info["tip"]:
        card("Tip", info["tip"])

    with st.expander("Top 3 predictions"):
        for i in top3:
            st.write(f"{class_names[i]}: {probs[i] * 100:.1f}%")

st.caption("AI Capstone Project | MobileNetV2 transfer learning | Trained on the Kaggle Garbage Classification dataset")
