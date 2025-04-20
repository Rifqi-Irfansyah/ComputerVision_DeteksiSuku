import streamlit as st
import os
from PIL import Image

from Interface.load_dataset import load_data

st.set_page_config(page_title="Face App", layout="wide")

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Dataset", "Augmentasi", "HaarCascade","MCNN", "RetinaFace", "Similarity", "About"])

if menu == "Dataset":
    load_data("Dataset", "null")

elif menu == "Augmentasi":
    load_data("Output/Augmented/Dataset", "Augmented")

elif menu == "HaarCascade":
    load_data("Output/HaarCascade/Dataset", "HaarCascade")

elif menu == "MCNN":
    load_data("Output/MCNN/Dataset", "MCNN")

elif menu == "RetinaFace":
    load_data("Output/RetinaFace/Dataset", "RetinaFace")

elif menu == "Similarity":
    st.subheader("🔍 Similarity Check")
    st.write("Halaman ini untuk melakukan pengecekan kemiripan wajah (fitur akan ditambahkan).")

elif menu == "About":
    st.subheader("ℹ️ Tentang Aplikasi")
    st.write("Aplikasi ini dibuat dengan Streamlit untuk mengeksplorasi dataset wajah dan melakukan face recognition.")
