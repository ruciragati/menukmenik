import streamlit as st
import torch
import urllib.request
from PIL import Image, ImageOps
from torchvision import transforms
from pillow_heif import register_heif_opener
from src.model import get_dog_classifier

register_heif_opener()

st.title("Menuk or Menik")
st.write("Upload a photo to see which Golden Girl it is!")

@st.cache_resource
def load_model():
    url = "https://huggingface.co/guccirucci/menukmenik/resolve/main/model.pth"
    weights_path = "model.pth"
    urllib.request.urlretrieve(url, weights_path)
    
    model = get_dog_classifier(num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

model = load_model()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader(
        label="Upload Image", 
        type=["jpg", "png", "jpeg", "heic", "HEIC"],
        label_visibility="collapsed"  
    )

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    image = Image.new("RGB", img.size) 
    image.paste(img)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("Identifying...")
    input_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, prediction = torch.max(probabilities, 0)
    
    names = ["Menik", "Menuk"]
    result = names[prediction.item()]
    
    st.success(f"That's **{result}**! ({confidence.item()*100:.2f}% confidence)")
