import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import numpy as np
from matplotlib.cm import get_cmap

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ----- FastAPI app -----
app = FastAPI(title="AI-Ray Backend", version="0.1.0")

@app.get("/")
def root():
    return {"message": "AI-Ray Backend is running!"}

# ----- CORS -----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Classes -----
CLASS_NAMES = ["normal", "pneumonia"]

# ----- Model Setup -----
MODEL = None

def load_model():
    model_path = r"D:\ai-ray-diagnose-backend\efficientnetb0_best_aug.pth"
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

@app.on_event("startup")
def _init():
    global MODEL
    MODEL = load_model()

# ----- Grad-CAM -----
def generate_gradcam(model, img_tensor, target_class):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Hook into the last conv layer of EfficientNet-B0
    last_conv = model.features[-1][0]
    last_conv.register_forward_hook(forward_hook)
    last_conv.register_backward_hook(backward_hook)

    # Forward + backward pass
    output = model(img_tensor)
    model.zero_grad()
    loss = output[0, target_class]
    loss.backward()

    # Compute Grad-CAM
    grad = gradients[0].detach()
    act = activations[0].detach()
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * act, dim=1).squeeze()
    cam = torch.clamp(cam, min=0)
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cam.cpu().numpy()

    # Convert to heatmap
    cmap = get_cmap("jet")
    heatmap = cmap(cam)[..., :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap)

    return heatmap_img

# ----- Inference with Grad-CAM -----
def predict_with_gradcam(image: Image.Image):
    original_size = image.size  # Save original image size

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = MODEL(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class_idx = torch.argmax(probs, dim=1).item()
        pred_class = CLASS_NAMES[pred_class_idx]
        confidence = float(probs[0, pred_class_idx] * 100)

    # Encode original image
    buf_orig = io.BytesIO()
    image.save(buf_orig, format="PNG")
    original_b64 = base64.b64encode(buf_orig.getvalue()).decode()

    # Generate Grad-CAM
    gradcam_img = generate_gradcam(MODEL, img_tensor, pred_class_idx)

    # Resize Grad-CAM to original image size
    gradcam_img = gradcam_img.resize(original_size, resample=Image.BILINEAR)

    # Convert to RGBA and manipulate channels
    gradcam_colored = gradcam_img.convert("RGBA")
    gradcam_np = np.array(gradcam_colored)

    if pred_class == "normal":
        # Zero out red channel for normal images
        gradcam_np[..., 0] = 0

    gradcam_colored = Image.fromarray(gradcam_np)

    # Overlay Grad-CAM on original
    original_resized = image.convert("RGBA")
    overlayed = Image.blend(original_resized, gradcam_colored, alpha=0.5)

    # Encode overlay
    buf_overlay = io.BytesIO()
    overlayed.save(buf_overlay, format="PNG")
    overlay_b64 = base64.b64encode(buf_overlay.getvalue()).decode()

    return {
        "diagnosis": "Normal" if pred_class=="normal" else "Pneumonia Detected",
        "confidence": confidence,
        "originalImage": f"data:image/png;base64,{original_b64}",
        "gradcamImage": f"data:image/png;base64,{overlay_b64}",
        "emoji": "✅" if pred_class=="normal" else "⚠️"
    }

# ----- Endpoint -----
@app.post("/diagnose-file")
async def diagnose_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    result = predict_with_gradcam(img)
    return result

