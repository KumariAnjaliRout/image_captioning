import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from transformers import BlipProcessor, BlipForConditionalGeneration

@st.cache_resource
def load_models():
    seg_model = maskrcnn_resnet50_fpn(pretrained=True)
    seg_model.eval()

    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    return seg_model, caption_model, caption_processor

seg_model, caption_model, caption_processor = load_models()

st.title("🖼️ Image Segmentation & Captioning App")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    img_np = np.array(image)
    img_tensor = F.to_tensor(img_np)

    with torch.no_grad():
        pred = seg_model([img_tensor])[0]

    def apply_masks(img, pred, threshold=0.7):
        img = img.copy()
        for i in range(len(pred["boxes"])):
            score = pred["scores"][i].item()
            if score < threshold:
                continue
            mask = pred["masks"][i, 0].mul(255).byte().cpu().numpy()
            img[mask > 128] = [0, 255, 0]
        return img

    masked_img = apply_masks(img_np, pred)
    st.image(masked_img, caption="Segmented Image", use_column_width=True)

    inputs = caption_processor(images=image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    caption = caption_processor.decode(out[0], skip_special_tokens=True)
    st.markdown(f"**📝 Caption:** _{caption}_")

    result_img = Image.fromarray(masked_img)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    result_img.save(temp_file.name)

    with open(temp_file.name, "rb") as f:
        st.download_button("📥 Download Output", f, file_name="output_result.jpg", mime="image/jpeg")
