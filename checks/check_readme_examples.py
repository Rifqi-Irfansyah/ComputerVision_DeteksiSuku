"""
Script to verify all examples in the readme.
Simply execute
    python test_readme_examples.py


The tests in this file are currently not unittests!
They do plot images.

TODO move this to checks/ ?

"""
from __future__ import print_function, division
import functools
import os

import pandas as pd
print("Current Working Directory:", os.getcwd())
print("File Exists:", os.path.exists("input_images"))

import cv2
import numpy as np
from imgaug import augmenters as iaa
from PIL import Image
from facenet_pytorch import MTCNN
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import torch.nn.functional as F
from retinaface import RetinaFace
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os


def main():
    example_simple_training_setting()
    example_very_complex_augmentation_pipeline()
    example_augment_images_and_keypoints()
    example_augment_images_and_bounding_boxes()
    example_augment_images_and_polygons()
    example_augment_images_and_linestrings()
    example_augment_images_and_heatmaps()
    example_augment_images_and_segmentation_maps()
    example_visualize_augmented_images()
    example_visualize_augmented_non_image_data()
    example_using_augmenters_only_once()
    example_multicore_augmentation()
    example_probability_distributions_as_parameters()
    example_withchannels()
    example_hooks()


def seeded(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import imgaug.random as iarandom
        iarandom.seed(0)
        func(*args, **kwargs)
    return wrapper

os.makedirs("output_images", exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = InceptionResnetV1(pretrained='vggface2').eval()

def load_batch_from_folder(folder_path):
    # Ambil semua file gambar dari folder
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)  # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # konversi ke RGB
            images.append(img)
            filenames.append(filename)
    return images, filenames

#Haar Cascade
def detect_and_crop_faces(images, filenames):
    cropped_faces = []
    cropped_filenames = []
    failed = 0

    for img, filename in zip(images, filenames):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        

        if len(faces) == 0:
            print(f"Tidak ditemukan wajah di gambar: {filename}")
            failed += 1
            continue

        for i, (x, y, w, h) in enumerate(faces):
            face = img[y:y+h, x:x+w]
            new_filename = filename
            cropped_faces.append(face)
            cropped_filenames.append(new_filename)
    print(f"‼️ {failed} Gambar tidak dapat terdeteksi wajahnya dengan HaarCascade ‼️")
    return cropped_faces, cropped_filenames

detector = MTCNN(keep_all=True)
def detect_face_with_mcnn(images, filenames):
    detected_faces = []
    new_filenames = []
    failed = 0

    for img_array, fname in zip(images, filenames):
        img_pil = Image.fromarray(img_array)
        boxes, _ = detector.detect(img_pil)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cropped = img_pil.crop((x1, y1, x2, y2))
                cropped_array = np.array(cropped)
                detected_faces.append(cropped_array)
                new_filename = fname
                new_filenames.append(new_filename)
        else:
            failed += 1
            print(f"Tidak ada wajah ditemukan di {fname}")
        
    print(f"‼️ {failed} Gambar tidak dapat terdeteksi wajahnya dengan MCNN ‼️")
    return detected_faces, new_filenames

def detect_face_with_retina_face(images, filenames):
    detected_faces = []
    new_filenames = []
    failed = 0

    for img_array, fname in zip(images, filenames):
        try:
            faces = RetinaFace.detect_faces(img_array)
        except Exception as e:
            print(f"Gagal mendeteksi wajah pada {fname}: {e}")
            failed += 1
            continue

        if isinstance(faces, dict):
            if not faces:
                print(f"Tidak ada wajah ditemukan di {fname}")
                failed += 1
                continue

            for i, face_key in enumerate(faces):
                face = faces[face_key]
                x1, y1, x2, y2 = map(int, face['facial_area'])

                cropped = img_array[y1:y2, x1:x2]
                detected_faces.append(cropped)

                new_filename = fname
                new_filenames.append(new_filename)
        else:
            failed += 1
            print(f"Tidak ada wajah ditemukan di {fname}")
        
    print(f"‼️ {failed} gambar tidak dapat terdeteksi wajahnya dengan RetinaFace ‼️")
    return detected_faces, new_filenames

def save_images(images_aug, filenames, directory):
    for img_aug, filename in zip(images_aug, filenames):
        save_path = os.path.join("Output", directory, f"{filename}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        img_aug_bgr = cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR)

        success = cv2.imwrite(save_path, img_aug_bgr)
        if success:
            print(f"✅ Berhasil simpan: {save_path}")
        else:
            print(f"❌ Gagal menyimpan gambar: {save_path}")

def save_images2(images_aug, filenames, directory):
    for img_aug, filename in zip(images_aug, filenames):
        save_path = os.path.join(directory, f"{filename}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        img_aug_bgr = cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR)

        success = cv2.imwrite(save_path, img_aug_bgr)
        if success:
            print(f"✅ Berhasil simpan: {save_path}")
        else:
            print(f"❌ Gagal menyimpan gambar: {save_path}")

def read_csv(path_csv):
    df = pd.read_csv(path_csv)
    image_paths = df['path_gambar'].tolist()

    images = []
    filenames = []
    for path in image_paths:
        if path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            filenames.append(path)
            
        print(f"Loaded {len(images)} images.")

    return images, filenames

def save_cropped_faces(images, filenames):
    for img, filename in zip(images, filenames):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        save_path = os.path.join("output_images", filename)
        cv2.imwrite(save_path, img_bgr)

transform = transforms.Compose([
    transforms.Resize((160, 160)),  
    transforms.ToTensor(),          
    transforms.Normalize([0.5], [0.5])  
])

# Fungsi untuk ekstraksi fitur wajah dan menghitung similarity
def get_face_embeddings_and_similarity(images, filenames):
    print("masuk embedding.")
    embeddings = []
    new_filenames = []

    for img, filename in zip(images, filenames):
        img_pil = Image.fromarray(img)
        img_tensor = transform(img_pil).unsqueeze(0)  
        embedding = model(img_tensor)  
        embeddings.append(embedding.detach())
        new_filenames.append(filename)

    print(f"Total {len(embeddings)} embeddings extracted.")
    return embeddings, new_filenames

def face_similarity_check(path_image1, image1_name, path_image2, image2_name, threshold=1.0):
    # Gabungkan path lengkap untuk gambar
    paths = [
        path_image1,
        path_image2
    ]
    images = []

    # Load dan konversi gambar
    for path in paths:
        if not os.path.exists(path):
            print(f"❌ Gambar tidak ditemukan: {path}")
            return 100
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    #Crop Wajah
    images_mcnn, new_path = detect_face_with_mcnn(images, paths)
    save_images2(images_mcnn, new_path, "Uploaded_Similarity/cut")
    if len(new_path) < 1:
        return 100
    # Ekstraksi fitur dari dua gambar wajah
    embeddings, _ = get_face_embeddings_and_similarity(images_mcnn, new_path)

    if len(embeddings) != 2:
        print("❌ Gagal mendapatkan embedding dari kedua gambar.")
        return 100

    similarity = calculate_face_similarity(embeddings[0], embeddings[1])
    print(f"\n✅ Similarity antara wajah:")
    print(f"   {image1_name} dan {image2_name} => {similarity:.4f}")

    if similarity <= threshold:
        print("🟢 MATCH (wajah kemungkinan mirip)")
    else:
        print("🔴 NOT MATCH (wajah kemungkinan berbeda)")
    return similarity

def calculate_face_similarity(embedding1, embedding2):
    distance = F.pairwise_distance(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
    return distance.item()

def train_ethnicity_classifier(dataset_dir="output_images/MCNN/DataSet", model_path="ethnicity_classifier_resnet50.h5"):
    print("✅masuk training")
    batch_size = 5
    target_size = (224, 224)

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    num_ethnicities = len(train_generator.class_indices)

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_ethnicities, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10
    )

    model.save(model_path)
    print(f"✅ Model disimpan ke {model_path}")
    return model, train_generator.class_indices

def predict_ethnicity(image_path, model_path="ethnicity_classifier_resnet50.h5", class_indices=None):
    print("✅masuk predict")
    model = tf.keras.models.load_model(model_path)

    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)
    if class_indices is None:
        raise ValueError("class_indices harus disediakan jika ingin melihat label etnis.")

    ethnicity = list(class_indices.keys())[np.argmax(pred)]
    print(f"Predicted ethnicity: {ethnicity}")
    return ethnicity

def implement_augmented():
    seq = iaa.Sequential([
        iaa.Crop(px=16, keep_size=True),   # crop 16px tapi jaga ukuran
        iaa.Fliplr(1.0),                    # selalu flip horizontal
        iaa.GaussianBlur(sigma=2.0),        # blur 
        iaa.Dropout((0.01, 0.1), per_channel=0.5),
        iaa.Resize((224, 224)),                       # Resize ke ukuran tetap sebelum augmentasi
        iaa.Affine(rotate=(-15, -5)),                 # Rotasi acak antara -15° sampai -5°
        iaa.Fliplr(0.5),                              # 50% kemungkinan flip horizontal
        iaa.Multiply((0.8, 1.2)),                     # Brightness: ±20%
        iaa.LinearContrast((0.8, 1.2)),               # Contrast: ±20%
        iaa.AdditiveGaussianNoise(scale=(0, 0.02*255))  # Gaussian noise ringan
    ])

    seq2 = iaa.Sequential([
        iaa.Crop(px=16, keep_size=True),   # crop 16px tapi jaga ukuran
        iaa.Dropout((0.01, 0.1), per_channel=0.5),
        iaa.Resize((224, 224)),                       # Resize ke ukuran tetap sebelum augmentasi
        iaa.Affine(rotate=(5, 15)),                 # Rotasi acak antara 5° sampai 15°
        iaa.Fliplr(0.5),                              # 50% kemungkinan flip horizontal
        iaa.AdditiveGaussianNoise(scale=(0, 0.02*255))  # Gaussian noise ringan
    ])

    images, filenames = read_csv("metadata.csv")
    filenames_augmented = [f"{os.path.splitext(f)[0]}.2{os.path.splitext(f)[1]}" for f in filenames]

    images_aug = seq(images=images)
    images_aug2 = seq2(images=images)
    save_images(images_aug, filenames, "Augmented")
    save_images(images_aug2, filenames_augmented, "Augmented")

def implement_haarcascade():
    images, filenames = read_csv("metadata.csv")
    cropped_faces, cropped_filenames = detect_and_crop_faces(images, filenames)
    save_images(cropped_faces, cropped_filenames, "HaarCascade")

def implement_mcnn():
    images, filenames = read_csv("metadata.csv")
    images_mcnn, new_filenames = detect_face_with_mcnn(images, filenames)
    save_images(images_mcnn, new_filenames, "MCNN")

def implement_retinaface():
    images, filenames = read_csv("metadata2.csv")
    images_retina, new_filenames = detect_face_with_retina_face(images, filenames)
    save_images(images_retina, new_filenames, "RetinaFace")

def example_simple_training_setting():
    print("Example: Simple Training Setting")

    # Pipeline augmentasi
    seq = iaa.Sequential([
        iaa.Crop(px=16, keep_size=True),   # crop 16px tapi jaga ukuran
        iaa.Fliplr(1.0),                    # selalu flip horizontal
        iaa.GaussianBlur(sigma=2.0),        # blur 
        iaa.Dropout((0.01, 0.1), per_channel=0.5),
        iaa.Resize((224, 224)),                       # Resize ke ukuran tetap sebelum augmentasi
        iaa.Affine(rotate=(-15, 15)),                 # Rotasi acak antara -15° sampai 15°
        iaa.Fliplr(0.5),                              # 50% kemungkinan flip horizontal
        iaa.Multiply((0.8, 1.2)),                     # Brightness: ±20%
        iaa.LinearContrast((0.8, 1.2)),               # Contrast: ±20%
        iaa.AdditiveGaussianNoise(scale=(0, 0.02*255))  # Gaussian noise ringan
    ])

    images, filenames = read_csv("../metadata.csv")

    cropped_faces, cropped_filenames = detect_and_crop_faces(images, filenames)
    save_cropped_faces(cropped_faces, cropped_filenames)

    images_mcnn, new_filenames = detect_face_with_mcnn(images, filenames)
    save_images(images_mcnn, new_filenames, "MCNN")

    images_retina, new_filenames = detect_face_with_retina_face(images[:10], filenames[:10])
    save_images(images_retina, new_filenames, "RetinaFace")
    
    model, classes = train_ethnicity_classifier()
    predict_ethnicity("../output_images/Cek/tes11.png", class_indices=classes)

    # Lakukan augmentasi
    images_aug = seq(images=images)
    # Simpan hasil augmentasi
    save_images(images_aug, filenames, "Augmented")
    print("Augmented images saved")

    face_similarity_check(
        path_image1="../Output/MCNN/DataSet/Sunda/Afriza",
        image1_name="image1.png",
        path_image2="../Output/MCNN/DataSet/Medan/Nashwa",
        image2_name="image1.jpg"
    )

# example_simple_training_setting()

# @seeded
# def example_simple_training_setting():
#     print("Example: Simple Training Setting")
#     import numpy as np
#     import imgaug.augmenters as iaa

#     def load_batch(batch_idx):
#         # dummy function, implement this
#         # Return a numpy array of shape (N, height, width, #channels)
#         # or a list of (height, width, #channels) arrays (may have different image
#         # sizes).
#         # Images should be in RGB for colorspace augmentations.
#         # (cv2.imread() returns BGR!)
#         # Images should usually be in uint8 with values from 0-255.
#         return np.zeros((128, 32, 32, 3), dtype=np.uint8) + (batch_idx % 255)

#     def train_on_images(images):
#         # dummy function, implement this
#         print(f"Training on batch of shape: {images.shape}")
#         pass

#     # Pipeline:
#     # (1) Crop images from each side by 1-16px, do not resize the results
#     #     images back to the input size. Keep them at the cropped size.
#     # (2) Horizontally flip 50% of the images.
#     # (3) Blur images using a gaussian kernel with sigma between 0.0 and 3.0.
#     seq = iaa.Sequential([
#         iaa.Crop(px=16, keep_size=False),  # selalu crop 16px
#         iaa.Fliplr(1.0),                    # selalu flip
#         iaa.GaussianBlur(sigma=2.0)        # selalu blur
#     ])

#     # seq = iaa.Sequential([
#     #     iaa.Crop(px=(1, 16), keep_size=True),
#     #     iaa.Fliplr(0.5),
#     #     iaa.GaussianBlur(sigma=(0, 3.0))
#     # ])

#     for batch_idx in range(100):
#         images = load_batch(batch_idx)
#         images_aug = seq(images=images)  # done by the library
#         train_on_images(images_aug)

#         # -----
#         # Make sure that the example really does something
#         if batch_idx == 0:
#             assert not np.array_equal(images, images_aug)


@seeded
def example_very_complex_augmentation_pipeline():
    print("Example: Very Complex Augmentation Pipeline")
    import numpy as np
    import imgaug as ia
    import imgaug.augmenters as iaa

    # random example images
    images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.LinearContrast((0.5, 2.0))
                        )
                    ]),
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    images_aug = seq(images=images)

    # -----
    # Make sure that the example really does something
    assert not np.array_equal(images, images_aug)


@seeded
def example_augment_images_and_keypoints():
    print("Example: Augment Images and Keypoints")
    import numpy as np
    import imgaug.augmenters as iaa

    images = np.zeros((2, 128, 128, 3), dtype=np.uint8)  # two example images
    images[:, 64, 64, :] = 255
    points = [
        [(10.5, 20.5)],  # points on first image
        [(50.5, 50.5), (60.5, 60.5), (70.5, 70.5)]  # points on second image
    ]

    seq = iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=0.05*255),
        iaa.Affine(translate_px={"x": (1, 5)})
    ])

    # augment keypoints and images
    images_aug, points_aug = seq(images=images, keypoints=points)

    print("Image 1 center", np.argmax(images_aug[0, 64, 64:64+6, 0]))
    print("Image 2 center", np.argmax(images_aug[1, 64, 64:64+6, 0]))
    print("Points 1", points_aug[0])
    print("Points 2", points_aug[1])


@seeded
def example_augment_images_and_bounding_boxes():
    print("Example: Augment Images and Bounding Boxes")
    import numpy as np
    import imgaug as ia
    import imgaug.augmenters as iaa

    images = np.zeros((2, 128, 128, 3), dtype=np.uint8)  # two example images
    images[:, 64, 64, :] = 255
    bbs = [
        [ia.BoundingBox(x1=10.5, y1=15.5, x2=30.5, y2=50.5)],
        [ia.BoundingBox(x1=10.5, y1=20.5, x2=50.5, y2=50.5),
         ia.BoundingBox(x1=40.5, y1=75.5, x2=70.5, y2=100.5)]
    ]

    seq = iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=0.05*255),
        iaa.Affine(translate_px={"x": (1, 5)})
    ])

    images_aug, bbs_aug = seq(images=images, bounding_boxes=bbs)


@seeded
def example_augment_images_and_polygons():
    print("Example: Augment Images and Polygons")
    import numpy as np
    import imgaug as ia
    import imgaug.augmenters as iaa

    images = np.zeros((2, 128, 128, 3), dtype=np.uint8)  # two example images
    images[:, 64, 64, :] = 255
    polygons = [
        [ia.Polygon([(10.5, 10.5), (50.5, 10.5), (50.5, 50.5)])],
        [ia.Polygon([(0.0, 64.5), (64.5, 0.0), (128.0, 128.0), (64.5, 128.0)])]
    ]

    seq = iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=0.05*255),
        iaa.Affine(translate_px={"x": (1, 5)})
    ])

    images_aug, polygons_aug = seq(images=images, polygons=polygons)


@seeded
def example_augment_images_and_linestrings():
    print("Example: Augment Images and LineStrings")
    import numpy as np
    import imgaug as ia
    import imgaug.augmenters as iaa

    images = np.zeros((2, 128, 128, 3), dtype=np.uint8)  # two example images
    images[:, 64, 64, :] = 255
    ls = [
        [ia.LineString([(10.5, 10.5), (50.5, 10.5), (50.5, 50.5)])],
        [ia.LineString([(0.0, 64.5), (64.5, 0.0), (128.0, 128.0), (64.5, 128.0),
                        (128.0, 0.0)])]
    ]

    seq = iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=0.05*255),
        iaa.Affine(translate_px={"x": (1, 5)})
    ])

    images_aug, ls_aug = seq(images=images, line_strings=ls)


@seeded
def example_augment_images_and_heatmaps():
    print("Example: Augment Images and Heatmaps")
    import numpy as np
    import imgaug.augmenters as iaa

    # Standard scenario: You have N RGB-images and additionally 21 heatmaps per
    # image. You want to augment each image and its heatmaps identically.
    images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
    heatmaps = np.random.random(size=(16, 64, 64, 1)).astype(np.float32)

    seq = iaa.Sequential([
        iaa.GaussianBlur((0, 3.0)),
        iaa.Affine(translate_px={"x": (-40, 40)}),
        iaa.Crop(px=(0, 10))
    ])

    images_aug, heatmaps_aug = seq(images=images, heatmaps=heatmaps)


@seeded
def example_augment_images_and_segmentation_maps():
    print("Example: Augment Images and Segmentation Maps")
    import numpy as np
    import imgaug.augmenters as iaa

    # Standard scenario: You have N=16 RGB-images and additionally one segmentation
    # map per image. You want to augment each image and its heatmaps identically.
    images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
    segmaps = np.random.randint(0, 10, size=(16, 64, 64, 1), dtype=np.int32)

    seq = iaa.Sequential([
        iaa.GaussianBlur((0, 3.0)),
        iaa.Affine(translate_px={"x": (-40, 40)}),
        iaa.Crop(px=(0, 10))
    ])

    images_aug, segmaps_aug = seq(images=images, segmentation_maps=segmaps)


@seeded
def example_visualize_augmented_images():
    print("Example: Visualize Augmented Images")
    import numpy as np
    import imgaug.augmenters as iaa

    images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
    seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.GaussianBlur((0, 3.0))])

    # Show an image with 8*8 augmented versions of image 0 and 8*8 augmented
    # versions of image 1. Identical augmentations will be applied to
    # image 0 and 1.
    seq.show_grid([images[0], images[1]], cols=8, rows=8)


@seeded
def example_visualize_augmented_non_image_data():
    print("Example: Visualize Augmented Non-Image Data")
    import numpy as np
    import imgaug as ia

    image = np.zeros((64, 64, 3), dtype=np.uint8)

    # points
    kps = [ia.Keypoint(x=10.5, y=20.5), ia.Keypoint(x=60.5, y=60.5)]
    kpsoi = ia.KeypointsOnImage(kps, shape=image.shape)
    image_with_kps = kpsoi.draw_on_image(image, size=7, color=(0, 0, 255))
    ia.imshow(image_with_kps)

    # bbs
    bbsoi = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=10.5, y1=20.5, x2=50.5, y2=30.5)
    ], shape=image.shape)
    image_with_bbs = bbsoi.draw_on_image(image)
    image_with_bbs = ia.BoundingBox(
        x1=50.5, y1=10.5, x2=100.5, y2=16.5
    ).draw_on_image(image_with_bbs, color=(255, 0, 0), size=3)
    ia.imshow(image_with_bbs)

    # polygons
    psoi = ia.PolygonsOnImage([
        ia.Polygon([(10.5, 20.5), (50.5, 30.5), (10.5, 50.5)])
    ], shape=image.shape)
    image_with_polys = psoi.draw_on_image(
        image, alpha_points=0, alpha_face=0.5, color_lines=(255, 0, 0))
    ia.imshow(image_with_polys)

    # heatmaps
    # pick first result via [0] here, because one image per heatmap channel
    # is generated
    hms = ia.HeatmapsOnImage(np.random.random(size=(32, 32, 1)).astype(np.float32),
                             shape=image.shape)
    image_with_hms = hms.draw_on_image(image)[0]
    ia.imshow(image_with_hms)


@seeded
def example_using_augmenters_only_once():
    print("Example: Using Augmenters Only Once")
    from imgaug import augmenters as iaa
    import numpy as np

    images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

    # always horizontally flip each input image
    images_aug = iaa.Fliplr(1.0)(images=images)

    # vertically flip each input image with 90% probability
    images_aug = iaa.Flipud(0.9)(images=images)

    # blur 50% of all images using a gaussian kernel with a sigma of 3.0
    images_aug = iaa.Sometimes(0.5, iaa.GaussianBlur(3.0))(images=images)


@seeded
def example_multicore_augmentation():
    print("Example: Multicore Augmentation")
    import skimage.data
    import imgaug as ia
    import imgaug.augmenters as iaa
    from imgaug.augmentables.batches import UnnormalizedBatch

    # Number of batches and batch size for this example
    nb_batches = 10
    batch_size = 32

    # Example augmentation sequence to run in the background
    augseq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.CoarseDropout(p=0.1, size_percent=0.1)
    ])

    # For simplicity, we use the same image here many times
    astronaut = skimage.data.astronaut()
    astronaut = ia.imresize_single_image(astronaut, (64, 64))

    # Make batches out of the example image (here: 10 batches, each 32 times
    # the example image)
    batches = []
    for _ in range(nb_batches):
        batches.append(UnnormalizedBatch(images=[astronaut] * batch_size))

    # Show the augmented images.
    # Note that augment_batches() returns a generator.
    for images_aug in augseq.augment_batches(batches, background=True):
        ia.imshow(ia.draw_grid(images_aug.images_aug, cols=8))


@seeded
def example_probability_distributions_as_parameters():
    print("Example: Probability Distributions as Parameters")
    import numpy as np
    from imgaug import augmenters as iaa
    from imgaug import parameters as iap

    images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

    # Blur by a value sigma which is sampled from a uniform distribution
    # of range 10.1 <= x < 13.0.
    # The convenience shortcut for this is: GaussianBlur((10.1, 13.0))
    blurer = iaa.GaussianBlur(10 + iap.Uniform(0.1, 3.0))
    images_aug = blurer(images=images)

    # Blur by a value sigma which is sampled from a gaussian distribution
    # N(1.0, 0.1), i.e. sample a value that is usually around 1.0.
    # Clip the resulting value so that it never gets below 0.1 or above 3.0.
    blurer = iaa.GaussianBlur(iap.Clip(iap.Normal(1.0, 0.1), 0.1, 3.0))
    images_aug = blurer(images=images)


@seeded
def example_withchannels():
    print("Example: WithChannels")
    import numpy as np
    import imgaug.augmenters as iaa

    # fake RGB images
    images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

    # add a random value from the range (-30, 30) to the first two channels of
    # input images (e.g. to the R and G channels)
    aug = iaa.WithChannels(
      channels=[0, 1],
      children=iaa.Add((-30, 30))
    )

    images_aug = aug(images=images)


@seeded
def example_hooks():
    print("Example: Hooks")
    import numpy as np
    import imgaug as ia
    import imgaug.augmenters as iaa

    # Images and heatmaps, just arrays filled with value 30.
    # We define the heatmaps here as uint8 arrays as we are going to feed them
    # through the pipeline similar to normal images. In that way, every
    # augmenter is applied to them.
    images = np.full((16, 128, 128, 3), 30, dtype=np.uint8)
    heatmaps = np.full((16, 128, 128, 21), 30, dtype=np.uint8)

    # add vertical lines to see the effect of flip
    images[:, 16:128-16, 120:124, :] = 120
    heatmaps[:, 16:128-16, 120:124, :] = 120

    seq = iaa.Sequential([
      iaa.Fliplr(0.5, name="Flipper"),
      iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
      iaa.Dropout(0.02, name="Dropout"),
      iaa.AdditiveGaussianNoise(scale=0.01*255, name="MyLittleNoise"),
      iaa.AdditiveGaussianNoise(loc=32, scale=0.0001*255, name="SomeOtherNoise"),
      iaa.Affine(translate_px={"x": (-40, 40)}, name="Affine")
    ])

    # change the activated augmenters for heatmaps,
    # we only want to execute horizontal flip, affine transformation and one of
    # the gaussian noises
    def activator_heatmaps(images, augmenter, parents, default):
        if augmenter.name in ["GaussianBlur", "Dropout", "MyLittleNoise"]:
            return False
        else:
            # default value for all other augmenters
            return default
    hooks_heatmaps = ia.HooksImages(activator=activator_heatmaps)

    # call to_deterministic() once per batch, NOT only once at the start
    seq_det = seq.to_deterministic()
    images_aug = seq_det(images=images)
    heatmaps_aug = seq_det(images=heatmaps, hooks=hooks_heatmaps)

    # -----------
    ia.show_grid(images_aug)
    ia.show_grid(heatmaps_aug[..., 0:3])


if __name__ == "__main__":
    example_simple_training_setting()
