import streamlit as st
import os
from PIL import Image
from Interface.func_detect_face import detect_face

def load_data(directory, button):
    selected_subfolder = "Pilih Subfolder"
    st.markdown("<h3 style='text-align: center;'>Face Recognition Dashboard</h3>", unsafe_allow_html=True)
    cols = st.columns([1, 1, 0.5])

    if not os.path.exists(directory):
        st.write(f"Belum Pernah Memproses {button}")
        if st.button("Run " + button):
            detect_face(button)
    else:
        if(button != "null"):
            with cols[2]:
                st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
                if st.button("Run " + button):
                    detect_face(button)
        with cols[0]:
            dataset_folder = st.selectbox("Pilih Folder Dataset", ["Pilih Folder"] + sorted(os.listdir(directory)))
        with cols[1]:
            if dataset_folder != "Pilih Folder":
                folder_path = os.path.join(directory, dataset_folder)
                subfolders = sorted([sf for sf in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, sf))])
                selected_subfolder = st.selectbox("Pilih Subfolder", ["Pilih Subfolder"] + subfolders)

        if selected_subfolder != "Pilih Subfolder":
            subfolder_path = os.path.join(folder_path, selected_subfolder)
            images = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            st.subheader(f"📁 {dataset_folder}/{selected_subfolder}")
            st.write(f"Jumlah gambar: {len(images)}")

            cols = st.columns(5)
            for i, img_name in enumerate(images):
                img_path = os.path.join(subfolder_path, img_name)
                try:
                    image = Image.open(img_path)
                    with cols[i % 5]:
                        st.image(image, caption=img_name, use_container_width=True)
                except Exception as e:
                    st.warning(f"Gagal memuat gambar {img_name}: {e}")