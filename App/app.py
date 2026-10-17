import streamlit as st
from PIL import Image
import time, torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms, models
device = "cuda" if torch.cuda.is_available() else "cpu"

classLabels = ["Apple","Banana","Cherry","Coconut","Grapes","Mango","Orange","Pineapple","Strawberry","Watermelon"]
transform = transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )
])
model = models.resnet18(weights = None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
def importModel(path:str):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

st.title("Fruit Classifier App")
st.write("Upload an image for prediction!")
imageUploaded = st.file_uploader(
    label="Image Uploader",
    type=["jpg","png","jpeg"]
)
if st.button("Predict"):
    st.write("Predicting...")
    time.sleep(5)
    if imageUploaded:
        # Image conversion
        image = Image.open(imageUploaded).convert("RGB")
        transformedImage = transform(image).unsqueeze(0)
        imageRGB = image.resize((224,224))
        imageRGB = np.array(imageRGB).astype(np.float32)/255.0

        # Model importing 
        modelImported = importModel(r"D:\AI\CNN Classifier\FruitClassifier\Models\Checkpoints\best_model_2.pth")

        # Predictions
        prediction = modelImported(transformedImage)
        probs = torch.softmax(prediction,dim=1)
        class_idx = torch.argmax(probs,dim=1)
        topProbs, topIDXs = torch.topk(probs, 3)
       
        # Grad Cam
        for params in modelImported.layer4.parameters():
            params.requires_grad = True
        target_layers = [modelImported.layer4[-1]]
        predicted_class = class_idx.item()
        targets = [ClassifierOutputTarget(predicted_class)]
        cam = GradCAM(model = modelImported, target_layers= target_layers)
        grayscaleCAM = cam(input_tensor=transformedImage, targets = targets)[0]
        visualizationCAM = show_cam_on_image(img=imageRGB, mask=grayscaleCAM, use_rgb=True)

        # Showing GradCAM
        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Image:")
            st.image(imageRGB)
        with col2:
            st.subheader("Grad-CAM visualization:")
            st.image(visualizationCAM)
        # Top 3 predictions
        st.subheader("Top 3 Predictions:")
        for probs, idx in zip(topProbs[0], topIDXs[0]):
            st.success(classLabels[idx])
            st.write(f"Confidence: {probs * 100:.2f}%")
            st.progress(float(probs))