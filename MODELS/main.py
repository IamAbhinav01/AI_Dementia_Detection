import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

# ---------------- Streamlit Page Config ----------------
st.set_page_config(
    page_title="Alzheimer MRI Analysis",
    page_icon="🧠",
    layout="wide"
)

# ---------------- App Title ----------------
st.markdown(
    """
    <div style="text-align:center">
        <h1 style='color:#4CAF50'>🧠 Alzheimer's MRI Analysis</h1>
        <p style='color:#555;'>Upload an MRI image to predict Alzheimer’s stage and visualize Grad-CAM</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# ---------------- File Uploader ----------------
file = st.file_uploader("📂 Upload MRI Image", type=["jpg", "jpeg", "png"])

# ---------------- Model Definition ----------------
class AlzheimerModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.base = models.resnet18(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)

model = AlzheimerModel()
model.load_state_dict(torch.load("Alzehimer.pth", map_location="cpu"))
model.eval()

# ---------------- Grad-CAM ----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224,224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

# ---------------- Preprocessing ----------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

classes = ["Mild Impairment", "Moderate Impairment", "No Impairment", "Very Mild Impairment"]

# ---------------- Prediction & Visualization ----------------
if file:
    img = Image.open(file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    # ---------------- Prediction Card ----------------
    st.markdown("### 🔍 Prediction Result")
    st.success(
        f"**Prediction:** {classes[predicted.item()]}  \n**Confidence:** {confidence.item()*100:.2f}%"
    )

    # ---------------- Confidence Scores ----------------
    st.markdown("### 📊 Confidence Scores")
    for i, cls in enumerate(classes):
        st.markdown(f"<b>{cls}</b>", unsafe_allow_html=True)
        st.progress(float(probs[0][i].item()))

    # ---------------- Grad-CAM ----------------
    target_layer = model.base.layer4[-1]
    gradcam = GradCAM(model, target_layer)
    cam, pred_class = gradcam.generate(img_tensor, class_idx=predicted.item())

    img_np = np.array(img.resize((224,224))) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = cv2.addWeighted(np.float32(img_np), 0.5, heatmap, 0.5, 0)

    # ---------------- UI Layout: Side by Side ----------------
    st.markdown("### 🎨 Grad-CAM Visualization")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_np, caption="🖼️ Original MRI", use_container_width=True)
    with col2:
        st.image(cam, caption="🔥 Grad-CAM Heatmap", use_container_width=True, clamp=True)
    with col3:
        st.image(overlay, caption=f"Overlay ({classes[pred_class]})", use_container_width=True)

    # ---------------- Download Overlay ----------------
    overlay_bgr = (overlay*255).astype(np.uint8)
    is_success, im_buf_arr = cv2.imencode(".png", overlay_bgr)
    if is_success:
        st.download_button(
            label="⬇️ Download Overlay Image",
            data=im_buf_arr.tobytes(),
            file_name="gradcam_overlay.png",
            mime="image/png"
        )

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#999;'>🧠 This is a research/demo project. Not for clinical use.</p>",
    unsafe_allow_html=True
)
