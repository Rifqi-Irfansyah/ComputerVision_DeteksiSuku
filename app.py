import streamlit as st
import os
from PIL import Image

from Interface.load_dataset import load_data
from checks.check_readme_examples import face_similarity_check, predict_ethnicity, train_ethnicity_classifier

st.set_page_config(page_title="Face App", layout="wide")

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Dataset", "Augmentasi", "HaarCascade","MCNN", "RetinaFace", "Similarity", "Similarity 2", "Ethnic"])

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
    selected_image2 = None
    selected_image1 = None
    cols = st.columns([1, 1])
    with cols[0]:
        st.title("Gambar 1")

        directory1 = "Dataset"
        if not os.path.exists(directory1):
            st.error(f"Folder '{directory1}' tidak ditemukan.")
        else:
            cols2 = st.columns([1, 1, 1])
            with cols2[0]:
                dataset_folder1 = st.selectbox("📁 Pilih Folder", ["Pilih Folder 1"] + sorted(os.listdir(directory1)))
            if dataset_folder1 != "Pilih Folder 1":
                with cols2[1]:
                    folder_path1 = os.path.join(directory1, dataset_folder1)
                    subfolders1 = sorted([sf for sf in os.listdir(folder_path1)])
                    selected_subfolder1 = st.selectbox("Pilih Subfolder", ["Pilih Subfolder 1"] + subfolders1)

                    if selected_subfolder1 != "Pilih Subfolder 1":
                        subfolder_path1 = os.path.join(folder_path1, selected_subfolder1)
                        images1 = [f for f in os.listdir(subfolder_path1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                        if images1:
                            with cols2[2]:
                                selected_image1 = st.selectbox("Pilih Gambar 1", images1)
                                img_path1 = os.path.join(subfolder_path1, selected_image1)
                            try:
                                image1 = Image.open(img_path1)
                                st.image(image1, caption=f"Gambar 1: {selected_image1}", use_container_width=True)
                            except Exception as e:
                                st.error(f"Gagal membuka gambar 1: {e}")
                        else:
                            st.warning("Tidak ada gambar di subfolder ini.")

    with cols[1]:
        st.title("Gambar 2")

        directory2 = "Dataset"
        if not os.path.exists(directory2):
            st.error(f"Folder '{directory2}' tidak ditemukan.")
        else:
            cols3 = st.columns([1, 1, 1])
            with cols3[0]:
                dataset_folder2 = st.selectbox("📁 Pilih Folder 2", ["Pilih Folder 2"] + sorted(os.listdir(directory2)))
                if dataset_folder2 != "Pilih Folder 2":
                    with cols3[1]:
                        folder_path2 = os.path.join(directory2, dataset_folder2)
                        subfolders2 = sorted([sf for sf in os.listdir(folder_path2)])
                        selected_subfolder2 = st.selectbox("Pilih Subfolder", ["Pilih Subfolder 2"] + subfolders2)

                        if selected_subfolder2 != "Pilih Subfolder 2":
                            subfolder_path2 = os.path.join(folder_path2, selected_subfolder2)
                            images2 = [f for f in os.listdir(subfolder_path2) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                            if images2:
                                with cols3[2]:
                                    selected_image2 = st.selectbox("Pilih Gambar 2", images2)
                                    img_path2 = os.path.join(subfolder_path2, selected_image2)
                                try:
                                    image2 = Image.open(img_path2)
                                    st.image(image2, caption=f"Gambar 2: {selected_image2}", use_container_width=True)
                                except Exception as e:
                                    st.error(f"Gagal membuka gambar 2: {e}")
                            else:
                                st.warning("Tidak ada gambar di subfolder ini.")

    colss = st.columns([1,1,1])
    with colss[1]:
        if selected_image2 and selected_image1:
            if st.button("🔍 Similarity Check"):
                similarity = face_similarity_check(img_path1, "", img_path2, "")
                st.write(f"Similarity kedua wajah = {similarity}")
                if similarity == 100:
                    st.write("❗❗ Wajah tidak terdeteksi ❗❗")
                elif similarity <= 1:
                    st.write("🟢 MATCH (wajah kemungkinan mirip)")
                else:
                    st.write("🔴 NOT MATCH (wajah kemungkinan berbeda)")

elif menu == "Similarity 2":
    selected_image2 = None
    selected_image1 = None
    cols = st.columns([1, 1])
    with cols[0]:
        st.title("Gambar 1")
        uploaded_file1 = st.file_uploader("Unggah Gambar 1", type=["jpg", "jpeg", "png"])
        save_path = "Uploaded_Similarity"

        if uploaded_file1 is not None:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, uploaded_file1.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file1.getbuffer())
            cols2 = st.columns([1, 3, 1])
            with cols2[1]:
                st.image(file_path, caption="Gambar yang diunggah", use_container_width=True)

    with cols[1]:
        st.title("Gambar 2")
        uploaded_file = st.file_uploader("Unggah Gambar 2", type=["jpg", "jpeg", "png"])
        save_path = "Uploaded_Similarity"

        if uploaded_file is not None:
            os.makedirs(save_path, exist_ok=True)
            file_path2 = os.path.join(save_path, uploaded_file.name)
            with open(file_path2, "wb") as f:
                f.write(uploaded_file.getbuffer())
            cols3 = st.columns([1, 3, 1])
            with cols3[1]:
                st.image(file_path2, caption="Gambar yang diunggah", use_container_width=True)


    cols2 = st.columns([1,1,1])
    with cols2[1]:
        if st.button("🔍 Similarity Check", use_container_width=True):
            similarity = face_similarity_check(file_path, "", file_path2, "")
            st.write(f"Similarity kedua wajah = {similarity}")
            if similarity == 100:
                st.write("❗❗ Wajah tidak terdeteksi ❗❗")
            elif similarity <= 1:
                st.write("🟢 MATCH (wajah kemungkinan mirip)")
            else:
                st.write("🔴 NOT MATCH (wajah kemungkinan berbeda)")

elif menu == "Ethnic":
    st.subheader("📸 Upload Foto untuk Deteksi Etnis")

    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
    save_path = "Uploaded_Ethnic"

    if uploaded_file is not None:
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Foto berhasil disimpan di: {file_path}")
        cols = st.columns([1, 1, 1])
        with cols[1]:
            st.image(file_path, caption="Gambar yang diunggah", use_container_width=True)

            if st.button("Deteksi Enis", use_container_width=True):
                model, classes = train_ethnicity_classifier()
                ethnic = predict_ethnicity(file_path, class_indices=classes)
                st.write(f"Prediksi Etnis = {ethnic}" )