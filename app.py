import streamlit as st
import torch
import urllib.request
from src.model import get_dog_classifier

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