import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
import datetime
import os

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="RetiNet Pro | DR Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PROFESSIONAL STYLING ---
st.markdown("""
<style>
    /* Main Background and Text */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Header Styling */
    h1 {
        color: #002060;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
    }
    h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Result Box Styling */
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        border-left: 6px solid #002060;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .result-label {
        font-size: 24px;
        font-weight: bold;
        color: #002060;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #eef2ff;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #002060;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #003399;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. MODEL DEFINITION & LOADING ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomSwin(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = timm.create_model('swinv2_base_window8_256.ms_in1k', pretrained=False, num_classes=0, img_size=256)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.num_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

def swinT_reshape_transform(tensor):
    # Fix for Swin Transformer Grad-CAM artifacting
    if len(tensor.shape) == 4:
        return tensor.permute(0, 3, 1, 2)
    B, L, C = tensor.shape
    H = W = int(L**0.5)
    return tensor.transpose(1, 2).reshape(B, C, H, W)

@st.cache_resource
def load_model():
    model = CustomSwin(num_classes=5).to(DEVICE)
    try:
        # Load the "best_model.pth" we just renamed
        path = "best_model.pth"
        if not os.path.exists(path):
            st.error("Model file 'best_model.pth' not found in directory.")
            return None
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# --- 4. IMAGE PROCESSING ---
def preprocess_image(image):
    # Ben Graham Preprocessing
    image = np.array(image.convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Resize first
    image = cv2.resize(image, (256, 256))
    
    # Color processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Ben Graham weighted add
    image_blur = cv2.GaussianBlur(image_rgb, (0,0), 10)
    image_proc = cv2.addWeighted(image_rgb, 4, image_blur, -4, 128)
    
    # Tensor transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet stats
    ])
    
    return transform(image_proc).unsqueeze(0).to(DEVICE), image_rgb, image_proc

# --- 5. VISUALIZATION ENGINE ---
def generate_explanation(model, input_tensor, original_img):
    # Use Stage 3 (layers[-2]) for High-Res Heatmaps
    target_layers = [model.backbone.layers[-2].blocks[-1]] 
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=swinT_reshape_transform)
    
    # Output class (predicted)
    targets = None # Auto-select highest score
    
    # Generate (Smoothed)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=True, aug_smooth=True)[0]
    
    # Normalized image for overlay
    img_float = original_img.astype(np.float32) / 255.0
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    return visualization

# --- 6. MEDICAL LOGIC & PDF REPORT ---
def get_recommendation(prediction):
    """Returns clinical suggestion based on DR severity."""
    if "No DR" in prediction:
        return "Routine screening in 12 months. Maintain healthy blood glucose levels."
    elif "Mild" in prediction:
        return "Repeat screening in 6-12 months. Optimize metabolic control."
    elif "Moderate" in prediction:
        return "Refer to ophthalmologist within 4-6 weeks for full evaluation."
    elif "Severe" in prediction:
        return "Refer to ophthalmologist within 2 weeks. Panretinal photocoagulation may be considered."
    elif "Proliferative" in prediction:
        return "URGENT referral to ophthalmologist. risk of vision loss. Immediate treatment required."
    else:
        return "Clinical correlation recommended."

def create_pdf(patient_name, prediction, date_time):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Header
    c.setFont("Helvetica-Bold", 24)
    c.setFillColorRGB(0, 0.12, 0.38) # Navy Blue
    c.drawString(50, 750, "RetiNet Pro Clinical Report")
    
    c.setStrokeColorRGB(0.8, 0.8, 0.8)
    c.line(50, 735, 550, 735)
    
    # Patient Info
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 14)
    c.drawString(50, 700, f"Patient Name: {patient_name}")
    c.drawString(50, 680, f"Scan Date:    {date_time}")
    
    # Results Box
    c.setFillColorRGB(0.95, 0.95, 0.95)
    c.rect(50, 580, 500, 80, fill=1, stroke=0)
    
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(70, 630, "AI Diagnosis Result:")
    
    # Color Coded Result
    if "No DR" in prediction:
        c.setFillColorRGB(0, 0.6, 0) # Green
    elif "Proliferative" in prediction or "Severe" in prediction:
        c.setFillColorRGB(0.8, 0, 0) # Red
    else:
        c.setFillColorRGB(1, 0.5, 0) # Orange
    c.setFont("Helvetica-Bold", 20)
    c.drawString(250, 630, prediction)
    
    # Recommendation Section
    recommendation = get_recommendation(prediction)
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 540, "Clinical Suggestion:")
    c.setFont("Helvetica", 12)
    c.drawString(50, 520, recommendation)
    
    # Disclaimer Footer
    c.setFont("Helvetica-Oblique", 10)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawString(50, 100, "Disclaimer: This report is generated by an Artificial Intelligence system (RetiNet).")
    c.drawString(50, 85, "It is intended as a screening aid and DOES NOT replace professional medical capability.")
    c.drawString(50, 70, "Please verify all findings with a certified ophthalmologist.")
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# --- 7. MAIN UI LAYOUT ---
classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80) 
    st.title("RetiNet Pro")
    st.markdown("### Medical-Grade DR Screening")
    st.caption("v2.0 | Swin Transformer")
    st.divider()
    
    st.markdown("**Patient Session**")
    patient_name = st.text_input("Full Name", value="", placeholder="Enter Patient Name")
    patient_id = st.text_input("Patient ID (Optional)", placeholder="E.g. #PT-2024-001")
    
    st.divider()
    st.markdown("Developed by Department of CSE")

# TABS LAYOUT
tab_home, tab_analysis, tab_about = st.tabs(["🏠 Home", "🔬 Analysis", "ℹ️ Methodology"])

# --- TAB 1: LANDING PAGE ---
with tab_home:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #002060; font-size: 3em;">RetiNet Pro</h1>
        <h3 style="color: #555;">AI-Powered Diabetic Retinopathy Screening System</h3>
        <p style="font-size: 1.2em; color: #666;">
            Early detection saves sight. RetiNet leverages state-of-the-art <b>Vision Transformers</b> 
            to identify retinal lesions with dermatologist-level accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features Grid
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**High Precision**\n\nAchieves **89.6% Accuracy** and **0.956 Kappa** on clinical benchmarks.")
    with c2:
        st.info("**Explainable AI**\n\nUtilizes **Grad-CAM** technology to visualize exactly where lesions are located.")
    with c3:
        st.info("**Instant Reports**\n\nGenerates PDF clinical summaries with actionable referral recommendations.")
        
    st.divider()
    st.subheader("How it Works")
    st.markdown("""
    1.  **Upload Scan**: Input a standard retinal fundus photograph.
    2.  **AI Analysis**: The **SwinV2** model scans for microaneurysms, hemorrhages, and exudates.
    3.  **Diagnosis**: Get an immediate severity grading (No DR to Proliferative).
    4.  **Clinical Review**: Download the report for ophthalmologist verification.
    """)
    
    if st.button("🚀 Start Analysis Now"):
        # This is a hack to switch tabs, but usually simplest is just telling user to click
        st.markdown("Click the **'🔬 Analysis'** tab above to begin.")

# --- TAB 2: ANALYSIS TOOL ---
with tab_analysis:
    st.title("👁️ Retinal Analysis Console")
    st.markdown("Upload a retinal fundus image for immediate grading.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None and model is not None:
        final_name = patient_name if patient_name else "Anonymous_Patient"
        
        # Top Row: Image & Result
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            img_pil = Image.open(uploaded_file)
            st.image(img_pil, caption="Patient Scan", use_container_width=True)
            
        with col2:
            with st.spinner("Analyzing Retinal Features..."):
                # Inference
                input_tensor, img_rgb, img_ben = preprocess_image(img_pil)
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    pred_idx = torch.argmax(probs, dim=1).item()
                    
                prediction = classes[pred_idx]
                recommendation = get_recommendation(prediction)
                
                # Display Logic
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f"<h3>Diagnosis Result</h3>", unsafe_allow_html=True)
                
                # Color-coded Result
                color = "green" if pred_idx == 0 else "orange" if pred_idx < 3 else "red"
                st.markdown(f'<p class="result-label" style="color:{color};">{prediction}</p>', unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown(f"**💡 Recommendation:**")
                st.info(recommendation)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Report Button
                pdf_bytes = create_pdf(final_name, prediction, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
                
                if patient_name:
                    st.download_button(
                        label="📄 Download Clinical Report",
                        data=pdf_bytes,
                        file_name=f"Report_{final_name.replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.caption("Please enter Patient Name in sidebar to download report.")

        # Bottom Row: Explainability
        st.divider()
        st.subheader("🔍 Clinical Explanation (Grad-CAM)")
        st.markdown("The AI highlights regions of interest (lesions, exudates) used for this diagnosis.")
        
        cam_img = generate_explanation(model, input_tensor, img_ben)
        
        col_cam1, col_cam2 = st.columns(2)
        with col_cam1:
            st.image(img_ben, caption="Enhanced Preprocessed View", use_container_width=True)
        with col_cam2:
            st.image(cam_img, caption="AI Attention Heatmap", use_container_width=True)

    elif uploaded_file is None:
        st.info("👋 Select the **Browes files** button to choose a retinal image.")

# --- TAB 3: ABOUT ---
with tab_about:
    st.subheader("Model Architecture")
    st.markdown("""
    **RetiNet** is built on the **Swin Transformer V2** architecture, a hierarchical Vision Transformer 
    that computes attention within shifted windows. This allows it to model both local features (like microaneurysms) 
    and global context (overall vessel structure) more effectively than traditional patterns.
    """)
    
    st.subheader("Performance Metrics")
    st.markdown("""
    | Metric | Score |
    | :--- | :--- |
    | **Accuracy** | 89.62% |
    | **Quadratic Kappa** | 0.9556 |
    | **Sensitivity (Severe)** | 97.4% |
    """)
    
    st.warning("**Disclaimer**: This tool is for research and educational purposes only. Not FDA approved for clinical diagnosis.")
