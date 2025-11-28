import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import timm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sqlite3
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
import io
import datetime
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RetiNet AI Suite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (MEDICAL THEME) ---
st.markdown("""
    <style>
    :root {
        --primary-color: #2c3e50;
        --secondary-color: #3498db;
        --accent-color: #e74c3c;
        --background-color: #f8f9fa;
        --text-color: #2c3e50;
    }
    
    .main {
        background-color: var(--background-color);
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        font-family: 'Segoe UI', sans-serif;
        color: var(--primary-color);
    }
    
    /* Navigation */
    .nav-link {
        font-size: 1.1rem;
        margin: 10px 0;
    }
    
    /* Cards */
    .medical-card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 5px solid var(--secondary-color);
    }
    
    /* Status Badges */
    .badge {
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .badge-success { background-color: #d4edda; color: #155724; }
    .badge-warning { background-color: #fff3cd; color: #856404; }
    .badge-danger { background-color: #f8d7da; color: #721c24; }
    
    /* Wizard Steps */
    .step-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 30px;
    }
    .step {
        text-align: center;
        flex: 1;
        font-weight: bold;
        color: #bdc3c7;
    }
    .step.active {
        color: var(--secondary-color);
        border-bottom: 3px solid var(--secondary-color);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: var(--secondary-color);
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL DEFINITION ---
class CoordinateAttention(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_h * a_w
        return out

class SwinTransformerCA(nn.Module):
    def __init__(self, num_classes=5, pretrained=False):
        super(SwinTransformerCA, self).__init__()
        self.backbone = timm.create_model('swinv2_tiny_window8_256.ms_in1k', pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        self.ca = CoordinateAttention(num_features)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = x.permute(0, 3, 1, 2)
        x = self.ca(x)
        x = self.avg_pool(x).flatten(1)
        x = self.head(x)
        return x

# --- DATABASE ---
def init_db():
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS records
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_id TEXT, patient_name TEXT, scan_date TEXT, 
                  eye_side TEXT, diagnosis TEXT, confidence REAL, severity INTEGER)''')
    conn.commit()
    conn.close()

init_db()

def save_record(p_id, p_name, date, side, diag, conf, sev):
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    c.execute("INSERT INTO records (patient_id, patient_name, scan_date, eye_side, diagnosis, confidence, severity) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (p_id, p_name, str(date), side, diag, conf, sev))
    conn.commit()
    conn.close()

# --- PDF GENERATOR ---
def generate_pdf(p_id, p_name, date, side, diag, conf, sev, heatmap_img=None):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Header
    c.setFillColor(colors.HexColor('#2c3e50'))
    c.rect(0, height - 100, width, 100, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 60, "RetiNet AI Diagnostic Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 85, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Patient Info
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 140, "Patient Details")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 165, f"ID: {p_id}")
    c.drawString(300, height - 165, f"Name: {p_name}")
    c.drawString(50, height - 185, f"Date: {date}")
    c.drawString(300, height - 185, f"Eye: {side}")
    
    # Diagnosis Box
    c.setFillColor(colors.HexColor('#ecf0f1'))
    c.rect(50, height - 280, width - 100, 70, fill=1, stroke=0)
    
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(70, height - 240, f"Diagnosis: {diag}")
    c.setFont("Helvetica", 12)
    c.drawString(70, height - 260, f"Severity Level: {sev} | Confidence: {conf:.1f}%")
    
    # Heatmap Image
    if heatmap_img:
        c.drawString(50, height - 320, "AI Attention Map (Grad-CAM):")
        # Convert PIL to ReportLab Image
        img_buffer = io.BytesIO()
        heatmap_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img = ImageReader(img_buffer)
        c.drawImage(img, 50, height - 600, width=250, height=250, preserveAspectRatio=True)
    
    # Disclaimer
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.gray)
    c.drawString(50, 50, "DISCLAIMER: This report is generated by an AI system (CAD). It is not a definitive medical diagnosis.")
    c.drawString(50, 40, "Please consult a qualified ophthalmologist for verification and treatment.")
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# --- UTILS ---
def check_image_quality(image):
    # Convert to CV2
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Sharpness (Laplacian Variance)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Brightness
    brightness = np.mean(gray)
    
    return sharpness, brightness

def auto_enhance(image, current_sharpness, current_brightness):
    """
    Automatically adjusts brightness and sharpness if they are below thresholds.
    """
    enhanced_image = image.copy()
    enhancements = []
    
    # Target thresholds
    TARGET_BRIGHTNESS = 110.0
    
    # 1. Brightness Correction
    if current_brightness < 80:
        # Calculate factor to reach target (capped at 2.0x to avoid washout)
        factor = TARGET_BRIGHTNESS / max(current_brightness, 10)
        factor = min(factor, 2.0) 
        enhancer = ImageEnhance.Brightness(enhanced_image)
        enhanced_image = enhancer.enhance(factor)
        enhancements.append(f"Brightness (+{int((factor-1)*100)}%)")
        
    # 2. Sharpness Correction
    if current_sharpness < 60:
        enhancer = ImageEnhance.Sharpness(enhanced_image)
        enhanced_image = enhancer.enhance(2.0) # Boost sharpness by 2x
        enhancements.append("Sharpness Boost (2.0x)")
        
    return enhanced_image, enhancements

def validate_fundus(image):
    """
    Heuristic check to see if image looks like a retina.
    1. Color Check: Retina should be Red/Orange dominant.
    2. Texture Check: Should have some complexity (vessels), not flat or geometric.
    """
    img_array = np.array(image)
    
    # 1. Color Dominance Check
    r_mean = np.mean(img_array[:,:,0])
    g_mean = np.mean(img_array[:,:,1])
    b_mean = np.mean(img_array[:,:,2])
    
    # Retinas are overwhelmingly red. R should be significantly higher than B.
    if r_mean < g_mean or r_mean < b_mean * 1.2:
        return False, "Image color profile does not match a retina (Not Red-Dominant)."
        
    # 2. Edge/Vessel Check
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    
    # Retinas have moderate edge density (vessels). 
    # Too low = Blank wall/sky. Too high = Text/Urban scene.
    # RELAXED THRESHOLDS: 0.001 to 0.6 to allow for severe DR (more edges) and smoother images.
    if edge_density < 0.001:
        return False, "Image is too featureless (No vessels detected)."
    if edge_density > 0.6:
        return False, "Image is too complex/noisy (Likely not a medical scan)."
        
    return True, "Valid Fundus Image"

def generate_gradcam(model, input_tensor, original_image):
    target_layers = [model.ca] # Target Coordinate Attention
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])[0] # Target class 0 for now, or predicted
    
    # Resize original for overlay
    img_resized = np.array(original_image.resize((256, 256))) / 255.0
    visualization = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)
    return Image.fromarray(visualization)

# --- APP LOGIC ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model():
    model = SwinTransformerCA(num_classes=5)
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        model.to(device)
        model.eval()
        return model, True
    except FileNotFoundError:
        return model, False

model, weights_loaded = load_model()

# --- NAVIGATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
st.sidebar.title("RetiNet")
page = st.sidebar.radio("Navigation", ["Home", "New Analysis", "Patient Education", "Model Info"])

if page == "Home":
    st.title("Welcome to RetiNet")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); padding: 50px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h1 style="color: white; margin-bottom: 15px; font-size: 3rem;">RetiNet</h1>
        <p style="font-size: 1.5rem; opacity: 0.9; font-weight: 300;">Advanced Diabetic Retinopathy Screening System</p>
        <div style="margin-top: 30px; font-size: 4rem;">👁️</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🚀 **Fast Analysis**\n\nGet results in seconds using our optimized Swin Transformer.")
    with col2:
        st.success("🎯 **High Accuracy**\n\nResearch-grade performance with Coordinate Attention technology.")
    with col3:
        st.warning("🛡️ **Privacy First**\n\nLocal processing ensures patient data never leaves this machine.")

elif page == "New Analysis":
    st.title("Diagnostic Analysis")
    
    # Wizard
    step = st.session_state.get('step', 1)
    st.markdown(f"""
    <div class="step-container">
        <div class="step {'active' if step >= 1 else ''}">1. Patient Data</div>
        <div class="step {'active' if step >= 2 else ''}">2. Upload Scan</div>
        <div class="step {'active' if step >= 3 else ''}">3. Quality Check</div>
        <div class="step {'active' if step >= 4 else ''}">4. Results</div>
    </div>
    """, unsafe_allow_html=True)
    
    if step == 1:
        st.markdown("### Step 1: Patient Information")
        with st.form("patient_form"):
            p_id = st.text_input("Patient ID", "PID-2025-001")
            p_name = st.text_input("Patient Name")
            date = st.date_input("Scan Date")
            side = st.radio("Eye Scanned", ["Left (OS)", "Right (OD)"])
            
            if st.form_submit_button("Next Step ➡️"):
                if p_name:
                    st.session_state['p_info'] = {'id': p_id, 'name': p_name, 'date': date, 'side': side}
                    st.session_state['step'] = 2
                    st.rerun()
                else:
                    st.error("Please enter Patient Name.")

    elif step == 2:
        st.markdown("### Step 2: Upload Retinal Scan")
        uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Preview", width=300)
            st.session_state['image'] = image
            if st.button("Next: Quality Check ➡️"):
                st.session_state['step'] = 3
                st.rerun()
        
        if st.button("⬅️ Back"):
            st.session_state['step'] = 1
            st.rerun()

    elif step == 3:
        st.markdown("### Step 3: Image Quality Assessment")
        image = st.session_state['image']
        
        # Calculate Metrics First
        sharpness, brightness = check_image_quality(image)
        
        # Calculate Edge Density for display
        img_array = np.array(image)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 50, 150)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        
        # Display Metrics (Debug View)
        st.markdown("#### 📊 Diagnostic Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpness", f"{sharpness:.1f}", help="Higher is better (>100)")
        c2.metric("Brightness", f"{brightness:.1f}", help="Target: 80-150")
        c3.metric("Complexity", f"{edge_density:.3f}", help="Target: 0.001 - 0.6")
        
        # Run Validation (Soft Check)
        is_valid, msg = validate_fundus(image)
        
        if not is_valid:
            st.warning(f"⚠️ **Quality Warning**: {msg}")
            st.info("The system detected potential issues, but you can proceed if you are sure this is a valid scan.")
        
        # Auto-Enhancement Logic
        if sharpness < 60 or brightness < 80:
            st.warning("⚠️ Suboptimal Quality Detected. Applying Auto-Correction...")
            enhanced_image, changes = auto_enhance(image, sharpness, brightness)
            if changes:
                st.success(f"🛠️ Applied Corrections: {', '.join(changes)}")
                st.image([image, enhanced_image], caption=["Original", "Enhanced"], width=300)
                st.session_state['image'] = enhanced_image
        else:
            if is_valid:
                st.success("✅ Image Quality Looks Good.")
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("⬅️ Upload New Image"):
                st.session_state['step'] = 2
                st.rerun()
        with col_btn2:
            # ALWAYS allow proceeding, just warn
            if st.button("Run Analysis 🚀", type="primary"):
                st.session_state['step'] = 4
                st.rerun()
            
    elif step == 4:
        st.markdown("### Step 4: Analysis Results")
        
        if not weights_loaded:
            st.error("Model weights not found.")
        else:
            with st.spinner("Analyzing..."):
                image = st.session_state['image']
                # Preprocess
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                # Inference
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)
                
                CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
                pred_class = CLASSES[pred.item()]
                conf_score = conf.item() * 100
                severity = pred.item()
                
                # Grad-CAM
                heatmap = generate_gradcam(model, input_tensor, image)
                
                # Display
                col_res, col_xai = st.columns(2)
                
                with col_res:
                    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
                    if pred_class == 'No DR':
                        st.markdown(f"<h2 style='color:green'>Negative (No DR)</h2>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h2 style='color:red'>Positive: {pred_class}</h2>", unsafe_allow_html=True)
                    
                    st.progress(int(conf_score))
                    st.caption(f"Confidence: {conf_score:.1f}%")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Recommendations
                    st.markdown("#### Clinical Plan")
                    if severity == 0: st.success("Routine rescreening in 12 months.")
                    elif severity <= 2: st.warning("Refer to ophthalmologist (Non-Urgent).")
                    else: st.error("URGENT referral required.")
                
                with col_xai:
                    st.markdown("#### Explainable AI (Grad-CAM)")
                    view_mode = st.radio("View Mode", ["Original", "Heatmap Overlay"])
                    if view_mode == "Original":
                        st.image(image, use_container_width=True)
                    else:
                        st.image(heatmap, caption="Model Attention Map", use_container_width=True)
                
                # Save & Report
                p_info = st.session_state['p_info']
                save_record(p_info['id'], p_info['name'], p_info['date'], p_info['side'], pred_class, conf_score, severity)
                
                pdf = generate_pdf(p_info['id'], p_info['name'], p_info['date'], p_info['side'], pred_class, conf_score, severity, heatmap)
                st.download_button("📄 Download Full Report (PDF)", pdf, f"Report_{p_info['id']}.pdf", "application/pdf")
                
        if st.button("Start New Analysis"):
            st.session_state['step'] = 1
            st.rerun()

elif page == "Patient Education":
    st.title("Patient Education Center")
    st.markdown("### Understanding Diabetic Retinopathy")
    st.markdown("""
    **Diabetic Retinopathy (DR)** is a complication of diabetes that affects the eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina).
    
    #### Stages of DR:
    1.  **Mild Nonproliferative Retinopathy**: Microaneurysms occur. Small areas of balloon-like swelling in the retina's tiny blood vessels.
    2.  **Moderate Nonproliferative Retinopathy**: As the disease progresses, some blood vessels that nourish the retina are blocked.
    3.  **Severe Nonproliferative Retinopathy**: Many more blood vessels are blocked, depriving several areas of the retina with their blood supply.
    4.  **Proliferative Retinopathy**: At this advanced stage, the signals sent by the retina for nourishment trigger the growth of new blood vessels.
    """)
    st.info("Early detection is key. 90% of vision loss can be prevented with early treatment.")

elif page == "Model Info":
    st.title("Technical Specifications")
    st.markdown("### Model Architecture")
    st.code("""
    Model: Swin Transformer V2 (Tiny)
    Neck: Custom Coordinate Attention Module
    Input: 256x256 RGB Images
    Training Data: Kaggle 2015 (Pretrain) + APTOS 2019 (Finetune)
    """, language="text")
    
    st.markdown("### Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "77.0%", "Baseline")
    col2.metric("Sensitivity", "82.0%", "Severe Cases")
    col3.metric("Inference Time", "120ms", "GPU")

# Footer
st.markdown("---")
st.caption("© 2025 RetiNet | All Rights Reserved | Created by Rafi Uddin")
