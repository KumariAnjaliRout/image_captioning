# image_captioning

# 🧠 Image Segmentation & Captioning Web App

An interactive web app that performs **image segmentation** and generates **natural language captions** using deep learning models — all in one place!

🚀 Built with **PyTorch**, **Hugging Face Transformers**, and **Streamlit**  
🌐 Hosted on: [👉 Live App Here](https://your-app-link.streamlit.app)

---

## ✨ Features

- 🔍 **Instance Segmentation** using `maskrcnn_resnet50_fpn` (pre-trained on COCO)
- 🖼️ **Image Captioning** using `BLIP` from Hugging Face Transformers
- 📸 Upload your own images
- 💡 See real-time segmentation masks and descriptive captions

---

## 🧠 Models Used

### 1. `maskrcnn_resnet50_fpn`
- Task: Image Segmentation
- Trained on: **MS-COCO Dataset**
- Provided by: **TorchVision**

### 2. `BLIP (BLIPProcessor + BLIPForConditionalGeneration)`
- Task: Image Captioning
- Trained on: **COCO, Conceptual Captions, Flickr30k**
- Provided by: **Hugging Face Transformers**

---

## 📦 Tech Stack

- `Python`
- `Streamlit` – frontend UI
- `Torch`, `TorchVision` – for segmentation
- `Transformers` – for captioning
- `opencv-python-headless`, `Pillow`, `numpy`

---

## 🛠 How to Run Locally

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
streamlit run app.py
