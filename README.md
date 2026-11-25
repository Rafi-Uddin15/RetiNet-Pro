# 👁️ RetiNet AI Suite
### Clinical-Grade Diabetic Retinopathy Detection System

![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Computer Vision](https://img.shields.io/badge/CV-Swin_Transformer-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)

---

## 💡 Project Vision
Diabetic Retinopathy (DR) is a leading cause of blindness. **RetiNet** bridges the gap between complex AI research and clinical application. It is a full-stack **Computer Vision** system that allows doctors to upload retinal scans and receive an instant diagnosis with **Explainable AI (Grad-CAM)** visualizations.

**Why this matters**: Enables early detection in remote areas without needing an on-site specialist.

---

## ⚙️ Technical Innovation
RetiNet fine-tunes an ImageNet-pretrained **Swin Transformer V2 (Base, window 8, 256px)** with a custom MLP head:
*   **Attention Mechanism**: Shifted-window self-attention captures both small lesions (microaneurysms, hemorrhages) and global retinal context.
*   **Evaluation**: 5-fold stratified cross-validation on APTOS 2019, with minority-class oversampling applied only inside each training fold. An earlier evaluation oversampled before splitting, which leaked duplicate images into validation and overstated the scores; the numbers below are from the corrected run.

| Out-of-fold metric (3,662 images, each scored once) | Score |
| :--- | :--- |
| 5-grade accuracy | 83.6% |
| Quadratic Weighted Kappa | 0.907 |
| Any DR (grade 1+): sensitivity / specificity | 99.1% / 98.1% |
| Referable DR (grade 2+): sensitivity / specificity | 89.9% / 96.0% |
| Severe (grade 3) / Proliferative (grade 4) recall | 52.8% / 58.0% |
*   **Explainability**: Generates Grad-CAM heatmaps so doctors can trust *why* the AI made a decision.

---

## 🖥️ Application Features
1.  **Auto-Quality Check**: Algorithms check image sharpness/brightness before processing.
2.  **Live Inference**: Runs in <2 seconds on a standard CPU/GPU.
3.  **PDF Reports**: Generates a professional patient report with diagnosis and next steps.
4.  **Privacy**: Zero-cloud dependency; all processing is local.

## 🚀 Quick Start (Docker)
The easiest way to run the full suite:

```bash
# 1. Build the container
docker build -t retinet-app .

# 2. Run the application
docker run -p 8501:8501 retinet-app
```
Access the dashboard at `http://localhost:8501`.

---

## 📂 Tech Stack Details
*   **Model Training**: PyTorch, Albumentations (Augmentation), Timm (Models).
*   **Frontend**: Streamlit, Plotly (Charts).
*   **Backend Logic**: Python, OpenCV.
*   **Deployment**: Docker.

---
**Author**: Rafi Uddin | [LinkedIn](https://www.linkedin.com/in/rafi-uddin15)
