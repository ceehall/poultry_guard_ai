import streamlit as st
from PIL import Image
import io
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import time

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PoultryGuard AI",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ──────────────────────────────────────── */
.stApp {
    background-color: #f0f7f4;
}

/* ── Force all default Streamlit text to be dark ─────────── */
.stApp, .stApp p, .stApp span, .stApp div,
.stApp label, .stApp li, .stApp h1, .stApp h2,
.stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: #1a2e25;
}

/* ── Top header bar ──────────────────────────────────────── */
.header-bar {
    background-color: #1a2e25;
    padding: 16px 24px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
}
.header-title {
    color: #ffffff !important;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.header-sub {
    color: #90c8a8 !important;
    font-size: 13px;
    margin-top: 3px;
}
.header-right {
    color: #90c8a8 !important;
    font-size: 13px;
}

/* ── KPI stat cards ──────────────────────────────────────── */
.stat-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px 20px 16px 20px;
    border-top: 4px solid #1a5c38;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    margin-bottom: 16px;
    min-height: 110px;
}
.stat-label {
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #5a7a68 !important;
    margin-bottom: 8px;
}
.stat-value {
    font-size: 30px !important;
    font-weight: 700 !important;
    color: #1a2e25 !important;
    line-height: 1.1;
}
.stat-change-pos {
    color: #1e7e44 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    margin-top: 6px;
}
.stat-change-neg {
    color: #c0392b !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    margin-top: 6px;
}

/* ── Section pill headers ────────────────────────────────── */
.section-header {
    background: #1a5c38;
    color: #ffffff !important;
    padding: 7px 16px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 14px;
    display: inline-block;
}

/* ── Result banners ──────────────────────────────────────── */
.result-healthy {
    background: #d4edda;
    border-left: 5px solid #1e7e44;
    padding: 14px 18px;
    border-radius: 8px;
    color: #145a2e !important;
    font-weight: 600;
    font-size: 14px;
    margin-top: 12px;
}
.result-disease {
    background: #fde8e8;
    border-left: 5px solid #c0392b;
    padding: 14px 18px;
    border-radius: 8px;
    color: #7b1a1a !important;
    font-weight: 600;
    font-size: 14px;
    margin-top: 12px;
}
.result-warning {
    background: #fff8e1;
    border-left: 5px solid #e67e22;
    padding: 14px 18px;
    border-radius: 8px;
    color: #7d4e00 !important;
    font-weight: 600;
    font-size: 14px;
    margin-top: 12px;
}

/* ── Empty state placeholder ─────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 50px 20px;
    color: #6b8f78 !important;
    background: #ffffff;
    border-radius: 14px;
    border: 2px dashed #b0d4c0;
    margin-top: 8px;
}
.empty-state-icon {
    font-size: 56px;
    margin-bottom: 12px;
}
.empty-state-title {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #1a5c38 !important;
    margin-bottom: 6px;
}
.empty-state-sub {
    font-size: 14px !important;
    color: #6b8f78 !important;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #1a2e25 !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSelectbox label {
    color: #d8ede2 !important;
}
.stButton > button {
    background-color: #1a5c38 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 10px 20px !important;
    font-size: 15px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  PRODUCTION CONFIGURATION & TRANSFORMS
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = ["Coccidiosis", "Healthy", "Newcastle Disease", "Salmonella"]

# ⬇️ ACTION REQUIRED: Update this based on the output from gatekeeper.py ⬇️
GATEKEEPER_CLASSES = {
    0: "Valid_Feces",  # Replace with the exact name of your valid dataset folder
    1: "Invalid_Image" # Replace with the exact name of your invalid dataset folder
}

# Tell the app which string exactly represents a bad image that should be blocked
INVALID_CLASS_NAME = "Invalid_Image" 


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────
#  MODEL ARCHITECTURE & IMAGE UTILITIES
# ─────────────────────────────────────────────────────────────
class PoultryDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(PoultryDiseaseClassifier, self).__init__()
        try:
            self.backbone = models.mobilenet_v3_large(weights=None)
        except TypeError:
            self.backbone = models.mobilenet_v3_large(pretrained=False)
            
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

def letterbox_resize(pil_img, target_size=(224, 224)):
    pil_img.thumbnail(target_size, Image.Resampling.LANCZOS)
    background = Image.new('RGB', target_size, (0, 0, 0))
    background.paste(pil_img, ((target_size[0] - pil_img.size[0]) // 2,
                               (target_size[1] - pil_img.size[1]) // 2))
    return background

def verify_structural_suitability(pil_image):
    img = np.array(pil_image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False
        
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    total_area = img.shape[0] * img.shape[1]
    area_ratio = contour_area / total_area
    
    if area_ratio > 0.85 or area_ratio < 0.01:
        return False
    return True

def isolate_region_of_interest_from_pil(pil_image, output_size=(224, 224)):
    img = np.array(pil_image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        pad = 10
        crop = img[max(0, y-pad):min(img.shape[0], y+h+pad), 
                   max(0, x-pad):min(img.shape[1], x+w+pad)]
        if crop.size > 0:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            return letterbox_resize(Image.fromarray(crop_rgb), output_size)
            
    return letterbox_resize(pil_image, output_size)

def calculate_flock_risk_score(cv_probability, disease_class, bird_age_weeks, humidity_pct, flock_density_high):
    if disease_class.lower() == "healthy":
        return float(cv_probability * 0.1)
        
    multiplier = 0.0
    cls_lower = disease_class.lower()
    if ("cocci" in cls_lower) and humidity_pct > 75:
        multiplier += 0.15
    if ("newcastle" in cls_lower or "ncd" in cls_lower) and flock_density_high:
        multiplier += 0.15
    if bird_age_weeks < 3:
        multiplier += 0.10
        
    final_score = (cv_probability * 0.65) + (multiplier * 0.35)
    return min(float(final_score), 1.0)


@st.cache_resource
def load_production_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Gatekeeper Validation Model
    gatekeeper = models.mobilenet_v3_small(weights=None)
    in_features = gatekeeper.classifier[3].in_features
    gatekeeper.classifier[3] = nn.Linear(in_features, 2)
    
    if Path("gatekeeper_guard.pth").exists():
        gatekeeper.load_state_dict(torch.load("gatekeeper_guard.pth", map_location=device))
    gatekeeper.eval()
    
    # Disease Classification Model
    classifier = PoultryDiseaseClassifier(num_classes=len(CLASS_NAMES))
    if Path("best_vetai_vision_model.pth").exists():
        classifier.load_state_dict(torch.load("best_vetai_vision_model.pth", map_location=device))
    classifier.eval()
    
    return gatekeeper, classifier, device

gatekeeper_model, main_classifier, compute_device = load_production_models()

def predict_pipeline(pil_image):
    # ── STAGE 1: Gatekeeper Guardrail Model Checking ──
    input_tensor = TRANSFORM(pil_image).unsqueeze(0).to(compute_device)
    with torch.no_grad():
        gate_output = gatekeeper_model(input_tensor)
        gate_class_index = gate_output.argmax(dim=1).item()
    
    # Resolve the predicted string class from our dictionary mapping
    predicted_label = GATEKEEPER_CLASSES.get(gate_class_index, "Unknown")
    
    if predicted_label == INVALID_CLASS_NAME and Path("gatekeeper_guard.pth").exists():
         return f"❌ Failed Gatekeeper Validation (Detected: {predicted_label})", 0.0, pil_image

    # ── STAGE 2: Geometric Structural Check ──
    if not verify_structural_suitability(pil_image):
        return "❌ Failed Structural Quality Check", 0.0, pil_image

    # ── STAGE 3: Extract Region of Interest ──
    processed_roi_img = isolate_region_of_interest_from_pil(pil_image)

    # ── STAGE 4: Final Disease Inference ──
    if not Path("best_vetai_vision_model.pth").exists():
        return "⚠️ Model weights missing", 0.0, processed_roi_img

    roi_tensor = TRANSFORM(processed_roi_img).unsqueeze(0).to(compute_device)
    with torch.no_grad():
        outputs = main_classifier(roi_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        top_index = probabilities.argmax().item()
        
    return CLASS_NAMES[top_index], float(probabilities[top_index]) * 100, processed_roi_img


# ─────────────────────────────────────────────────────────────
#  DISEASE INFORMATION DATABASE
# ─────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Healthy": {
        "severity": "None",
        "description": "The bird appears healthy. Feces look normal.",
        "action": "No treatment needed. Continue routine monitoring.",
        "color": "healthy",
    },
    "Coccidiosis": {
        "severity": "High",
        "description": "Coccidiosis is a parasitic disease affecting the intestinal tract.",
        "action": "Isolate affected birds. Administer Amprolium. Consult a vet.",
        "color": "disease",
    },
    "Newcastle Disease": {
        "severity": "Critical",
        "description": "A highly contagious viral disease. Can devastate a flock rapidly.",
        "action": "Quarantine immediately. Contact a licensed veterinarian NOW.",
        "color": "disease",
    },
    "Salmonella": {
        "severity": "High",
        "description": "Bacterial infection causing white/greenish droppings.",
        "action": "Start antibiotic therapy (vet prescription). Improve biosecurity.",
        "color": "warning",
    },
}

# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🐔 PoultryGuard AI")
    st.markdown("---")

    st.markdown("### Navigation")
    page = st.radio(
        label="Go to",
        options=["🏠 Dashboard", "🔬 Diagnose Bird", "📋 Disease Guide", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Farm Settings")
    farm_name  = st.text_input("Farm Name",  value="My Poultry Farm")
    flock_size = st.number_input("Total Flock Size", min_value=1, value=500, step=50)
    bird_type  = st.selectbox("Bird Type", ["Broiler", "Layer", "Turkey", "Duck"])

    st.markdown("---")
    st.markdown("### Environmental Telemetry")
    bird_age = st.number_input("Bird Age (Weeks)", min_value=1, max_value=150, value=4, step=1)
    humidity = st.slider("Coop Humidity Level (%)", min_value=0, max_value=100, value=65)
    high_density = st.checkbox("High Flock Density Pen?", value=False)

    st.markdown("---")
    st.caption("PoultryGuard AI v1.0  |  Powered by VetAI Vision")


# ─────────────────────────────────────────────────────────────
#  PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown(f"""
    <div class="header-bar">
        <div>
            <div class="header-title">🐔 PoultryGuard AI — Dashboard</div>
            <div class="header-sub">HOME › DASHBOARD &nbsp;|&nbsp; {farm_name}</div>
        </div>
        <div class="header-right">{bird_type} &nbsp;·&nbsp; Flock: {flock_size:,} birds</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Flock</div>
            <div class="stat-value">{flock_size}</div>
            <div class="stat-change-pos">▲ Active Environment</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Scans Today</div>
            <div class="stat-value">12</div>
            <div class="stat-change-pos">▲ +4 since yesterday</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Disease Alerts</div>
            <div class="stat-value">2</div>
            <div class="stat-change-neg">▼ Requires attention</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Healthy Rate</div>
            <div class="stat-value">94%</div>
            <div class="stat-change-pos">▲ +1.2% since last month</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown('<div class="section-header">Quick Status</div>', unsafe_allow_html=True)
        st.info("🩺 **Last Diagnosis:** Coccidiosis detected (3 hrs ago)")
        st.success("✅ **Biosecurity Status:** Good")
        st.warning("⚠️ **Pending Scans:** 5 birds flagged for review")
        st.error("🚨 **Alert:** Check Pen 3 — abnormal droppings reported")

    with col_right:
        st.markdown('<div class="section-header">How To Use PoultryGuard AI</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#ffffff; border-radius:12px; padding:20px 24px;">
            <div class="how-to-step">
                <div class="how-to-num">1</div>
                <div><strong>Go to 🔬 Diagnose Bird</strong> in the left sidebar.</div>
            </div>
            <div class="how-to-step">
                <div class="how-to-num">2</div>
                <div><strong>Upload a clear photo</strong> of the bird's droppings — or use your camera to snap one directly.</div>
            </div>
            <div class="how-to-step">
                <div class="how-to-num">3</div>
                <div><strong>Click "Run Diagnosis"</strong> and wait a few seconds for the AI to analyse the image.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  PAGE: DIAGNOSE BIRD
# ─────────────────────────────────────────────────────────────
elif page == "🔬 Diagnose Bird":
    st.markdown("""
    <div class="header-bar">
        <div>
            <div class="header-title">🔬 Bird Feces Diagnosis</div>
            <div class="header-sub">HOME › DIAGNOSE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    input_method = st.radio("Provide the image:", ["📁 Upload from device", "📷 Use camera"], horizontal=True)
    image = None

    if input_method == "📁 Upload from device":
        uploaded_file = st.file_uploader("Select a photo", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
    else:
        camera_photo = st.camera_input("Take a photo")
        if camera_photo is not None:
            image = Image.open(camera_photo).convert("RGB")

    if image is not None:
        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.markdown('<div class="section-header">Image Preview</div>', unsafe_allow_html=True)
            st.image(image, caption="Original image", use_container_width=True)

        with col_result:
            st.markdown('<div class="section-header">Diagnosis Result</div>', unsafe_allow_html=True)

            if st.button("🧠 Run Diagnosis", type="primary", use_container_width=True):
                with st.spinner("Processing multi-stage guardrail validation & inference pipeline..."):
                    predicted_class, confidence, processed_roi = predict_pipeline(image)

                st.markdown("---")

                if "Failed" in predicted_class:
                    st.error(f"**Image rejected by Verification Architecture:**\n\n{predicted_class}")
                    st.warning("Please ensure you upload a clear photo of valid poultry fecal matter.")
                
                elif predicted_class == "⚠️ Model weights missing":
                    st.warning("**Model weights file not found.** Ensure `.pth` files are in the directory.")
                else:
                    info = DISEASE_INFO.get(predicted_class)
                    st.image(processed_roi, caption="Isolated Region of Interest (ROI) Analysis Target", width=180)

                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric(label="Predicted Condition", value=predicted_class)
                    with metric_col2:
                        st.metric(label="Inference Confidence", value=f"{confidence:.1f}%")
                    st.progress(int(confidence))

                    raw_prob = confidence / 100.0
                    risk_score = calculate_flock_risk_score(raw_prob, predicted_class, bird_age, humidity, high_density)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.metric(label="Calculated Flock Risk Index", value=f"{risk_score:.2f} / 1.00")
                    st.progress(int(risk_score * 100))

                    banner_class = f"result-{info['color']}"
                    severity_icon = {"None": "✅", "High": "⚠️", "Critical": "🚨"}.get(info["severity"], "ℹ️")
                    st.markdown(f"""
                    <div class="{banner_class}">
                        {severity_icon} <strong>Severity Status: {info['severity']}</strong><br><br>
                        {info['description']}
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>**Recommended Action Plan:**", unsafe_allow_html=True)
                    st.info(info["action"])
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📸</div>
            <div class="empty-state-title">No image yet</div>
            <div class="empty-state-sub">Upload a photo or use your camera above to get started.</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  PAGE: DISEASE GUIDE
# ─────────────────────────────────────────────────────────────
elif page == "📋 Disease Guide":
    st.markdown("""
    <div class="header-bar">
        <div>
            <div class="header-title">📋 Disease Reference Guide</div>
            <div class="header-sub">HOME › DISEASE GUIDE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    for disease, info in DISEASE_INFO.items():
        if disease == "Healthy": continue
        with st.expander(f"🦠 {disease}  —  Severity: {info['severity']}", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Description:** \n{info['description']}")
            with col_b:
                st.markdown(f"**Recommended Action:** \n{info['action']}")

# ─────────────────────────────────────────────────────────────
#  PAGE: ABOUT
# ─────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown("""
    <div class="header-bar">
        <div>
            <div class="header-title">ℹ️ About PoultryGuard AI</div>
            <div class="header-sub">HOME › ABOUT</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    ### What is PoultryGuard AI?
    **PoultryGuard AI** is an AI-powered disease detection tool designed for poultry farmers. By analysing a simple photo of bird droppings, the system can identify common diseases early.
    """)