#  RetiNet AI Suite
### Advanced Diabetic Retinopathy Detection System

![RetiNet Banner](https://img.shields.io/badge/RetiNet-AI%20Diagnostic%20Suite-blue?style=for-the-badge&logo=python)
![Python](https://img.shields.io/badge/Python-3.11-yellow?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square)

**RetiNet** is a state-of-the-art medical imaging application designed to detect and grade **Diabetic Retinopathy (DR)** from retinal fundus images. Powered by a custom **Swin Transformer V2** with **Coordinate Attention**, it delivers research-grade accuracy with a clinical-friendly interface.

---

##  Key Features

*   ** Clinical-Grade UI**: Professional, clean interface designed for medical practitioners.
*   ** Advanced AI Model**: Uses a custom **Swin Transformer V2 (Tiny)** architecture enhanced with **Coordinate Attention** for precise feature extraction.
*   ** Explainable AI (XAI)**: Integrated **Grad-CAM** visualization shows exactly *where* the model is looking (lesions, hemorrhages).
*   ** Auto-Enhancement**: Automatically detects and fixes low-quality images (too dark/blurry) before analysis.
*   ** PDF Reporting**: Generates downloadable, professional diagnostic reports with patient details and AI findings.
*   ** Privacy-First**: All processing happens locally (or in-container); no patient data is sent to the cloud.

---

##  Installation

### Option A: Docker (Recommended)
The easiest way to run RetiNet is using Docker. This ensures all dependencies (including system libraries for OpenCV) are correct.

1.  **Build the Image**:
    `ash
    docker build -t retinet-app .
    `

2.  **Run the Container**:
    `ash
    docker run -p 8501:8501 retinet-app
    `

3.  **Access the App**:
    Open your browser and go to http://localhost:8501.

### Option B: Local Installation
If you prefer running it directly on your machine:

1.  **Clone the Repository**:
    `ash
    git clone https://github.com/Rafi-Uddin15/Retinet.git
    cd Retinet
    `

2.  **Install Dependencies**:
    `ash
    pip install -r requirements.txt
    `

3.  **Run the App**:
    `ash
    streamlit run app.py
    `

---

##  Usage Guide

1.  **Patient Data**: Enter the Patient ID, Name, and Scan Date.
2.  **Upload Scan**: Upload a retinal fundus image (.jpg, .png).
3.  **Quality Check**:
    *   The system automatically checks for image quality (sharpness, brightness, validity).
    *   If the image is poor, **Auto-Enhance** will offer to fix it.
    *   *Note: You can bypass warnings if necessary.*
4.  **Analysis**: Click "Run Analysis" to get the diagnosis.
5.  **Report**: View the severity grade, confidence score, and download the **PDF Report**.

---

##  Model Architecture

*   **Backbone**: Swin Transformer V2 (Tiny)
*   **Attention Mechanism**: Coordinate Attention (CA) Block
*   **Input Resolution**: 256x256
*   **Training Data**: Pretrained on Kaggle DR 2015, Fine-tuned on APTOS 2019.

---

##  Project Structure

`
RetiNet/
 app.py                      # Main Streamlit Application
 best_model.pth              # Trained Model Weights (Swin+CA)
 Dockerfile                  # Docker Configuration
 requirements.txt            # Python Dependencies
 dr_detection_transfer_learning.ipynb  # Training Notebook (Transfer Learning)
 dr_detection_custom.ipynb   # Training Notebook (Custom Architecture)
 README.md                   # Project Documentation
`

---

##  Disclaimer
*This tool is a Computer-Aided Diagnosis (CAD) system intended to assist medical professionals. It is **not** a replacement for a doctor. All diagnoses should be verified by a qualified ophthalmologist.*

---

** 2025 RetiNet AI Suite | Created by Rafi Uddin**
