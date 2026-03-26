# Menuk vs. Menik Classifier
A Deep Learning web application designed to distinguish between my two favorite Golden Retrievers, Menuk and Menik, using Transfer Learning.

## The Stars of the Show

<table>
  <tr>
    <td align="center"><b>Menik</b></td>
    <td align="center"><b>Menuk</b></td>
  </tr>
  <tr>
    <td><img src="images/menik.jpg" width="300"></td>
    <td><img src="images/menuk.jpg" width="300"></td>
  </tr>
</table>

## Link: https://menukmenik.streamlit.app/

## Project Overview
* **Training:** Conducted in Google Colab using PyTorch.
* **Model:** ResNet18 with a custom FC head.
* **Deployment:** Streamlit Cloud.
* **Weights Hosting:** Hugging Face.

## Tech Stack
* **Language:** Python 3.12+
* **Framework:** PyTorch & Torchvision
* **Web Interface:** Streamlit
* **Image Handling:** Pillow & Pillow-Heif

## Installation & Local Setup
1. Clone the repo: git clone https://github.com/yourusername/menukmenik.git
2. Install dependencies: pip install -r requirements.txt
3. Run the app: streamlit run app.py