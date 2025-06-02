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
        embedding_obj = DeepFace.represent(img_path=img, model_name='Facenet512', enforce_detection=True)[0]
        embedding = np.array(embedding_obj["embedding"], dtype=np.float32)
        norm_embedding = normalize([embedding])[0]  # Normalisasi
        return norm_embedding
    except Exception as e:
        print(f"Encoding Error: {e}")
        return None

# =====================================
# 2. Buat Dataset dari Webcam dan Kirim ke Milvus
# =====================================
def create_dataset_from_livecam(nim: str, nama: str):
    cam = cv2.VideoCapture(0)
    print("Tekan 's' untuk simpan wajah. Tekan 'q' untuk keluar.")
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        cv2.imshow("Ambil Wajah", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            embedding = encode_face(frame)
            if embedding is not None:
                collection.insert([[nim], [nama], [embedding],["original"]])
                print(f"Wajah {nama} disimpan ke Milvus.")
                break
            else:
                print("Gagal mendeteksi wajah.")
        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# =====================================
# 3. Pencarian Wajah dari Webcam (Live)
# =====================================
def search_face_from_livecam(threshold=0.7):
    cam = cv2.VideoCapture(0)
    print("Tekan 'q' untuk keluar pencarian.")
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        embedding = encode_face(frame)
        if embedding is not None:
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
                        print(f"{distance} >= {threshold}")
                        print(f"[MATCH] Nama: {hit.entity.get('nama')} | NIM: {hit.entity.get('nim')} | Score: {distance:.4f}")
                    else:
                        print(f"[NO MATCH] Score: {distance:.4f}")

        cv2.imshow("Search Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
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