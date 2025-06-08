import cv2
import numpy as np
from deepface import DeepFace
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sklearn.preprocessing import normalize
import time
import json
import requests


URL_API = "http://localhost:3333/api/python"
# Hubungkan ke Milvus
connections.connect("default", host="localhost", port="19530")
# Konstanta
COLLECTION_NAME = "faces"

collection = Collection(name=COLLECTION_NAME)
collection.load()

# =====================================
# 1. Fungsi untuk Encode Wajah (Normalize)
# =====================================
def encode_face(img) -> np.ndarray:
    try:
        # DeepFace return banyak hasil, kita ambil satu
        embedding_obj = DeepFace.represent(img_path=img, model_name='Facenet', enforce_detection=True)[0]
        embedding = np.array(embedding_obj["embedding"], dtype=np.float32)
        norm_embedding = normalize([embedding])[0]  # Normalisasi
        return norm_embedding
    except Exception as e:
        print(f"Encoding Error: {e}")
        return None

# =====================================
# 2. Buat Dataset dari Webcam dan Kirim ke Milvus
# =====================================
def augment_image(image):
    augmented_images = []

    # Flip horizontal
    flip = cv2.flip(image, 1)
    augmented_images.append(('flip', flip))

    # Brightness adjustment
    bright = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    augmented_images.append(('bright', bright))

    # Rotate 15 degrees
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 15, 1)
    rotated = cv2.warpAffine(image, M, (w, h))
    augmented_images.append(('rotate', rotated))

    # Gaussian blur (optional)
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    augmented_images.append(('blur', blur))

    return augmented_images

def create_dataset_from_image(image, nim: str, nama: str):
    try:
        success = False
        pesan_list = []

        # Cek apakah gambar valid
        if image is None:
            return json.dumps({"status": False, "pesan": "[ERROR] Gambar kosong atau tidak valid."})

        # List untuk menyimpan semua variasi gambar (original + augmentasi)
        all_images = [('original', image)] + augment_image(image)

        for aug_type, img in all_images:
            try:
                embedding = encode_face(img)
                if embedding is not None:
                    collection.insert([[nim], [nama], [embedding], [aug_type]])
                    pesan_list.append(f"Wajah {nama} ({aug_type}) berhasil disimpan.")
                    success = True
                else:
                    pesan_list.append(f"[WARNING] Wajah tidak terdeteksi pada augmentasi '{aug_type}'.")
            except Exception as e:
                pesan_list.append(f"[ERROR] Gagal memproses augmentasi '{aug_type}': {str(e)}")

        if not success:
            return json.dumps({"status": False, "pesan": "[ERROR] Tidak ada wajah yang terdeteksi dalam semua gambar."})

        return json.dumps({"status": True, "pesan": pesan_list})

    except Exception as e:
        return json.dumps({"status": False, "pesan": f"[EXCEPTION] Terjadi kesalahan: {str(e)}"})

def search_face_from_face(filepath: str, threshold=0.7):
    embedding = encode_face(filepath)
    
    if embedding is None:
        return json.dumps({"status": False,"pesan": "[ERROR] Wajah tidak terdeteksi dalam gambar."})

    search_result = collection.search(
        data=[embedding],
        anns_field="encoding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=1,
        output_fields=["nim", "nama"]
    )

    for hits in search_result:
        for hit in hits:
            distance = hit.distance
            if distance >= threshold:
                return json.dumps({"status": True,"pesan": f" {hit.entity.get('nama')} Score: {distance:.4f}", "data" : {
                    "nama" : hit.entity.get('nama'),
                    "nim" : hit.entity.get('nim'),
                }})
                print()
            else:
                return json.dumps({"status": False,"pesan": f"[NO MATCH] Score: {distance:.4f}"})
    return json.dumps({"status": False,"pesan": "[ERROR] Wajah tidak ditemukan."})
def curl_post(data: dict):
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(URL_API, data=json.dumps(data), headers=headers)
        response.raise_for_status()
        print(f"[INFO] POST berhasil: {response.status_code}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Gagal POST ke {URL_API}: {e}")
        return []
#create_dataset_from_livecam("2155201014","Amir")
#search_face_from_livecam()
#search_face_from_file("gabung.jpg")