import cv2
import numpy as np
from deepface import DeepFace
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sklearn.preprocessing import normalize
import time
import json
import base64
import requests
import platform
import sys

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
        all_images = [('original', image)]# + augment_image(image) // buat tanpa augmentasi dulu

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

def search_face_from_face(filepath: str, threshold=0.75):
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
                    "score" : hit.entity.get('distance'),
                }})
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
def is_yolo_face_valid(x1, y1, x2, y2, frame_width, frame_height,
                       min_ratio=0.2, max_ratio=0.6):
    # Konversi ke w dan h
    w = x2 - x1
    h = y2 - y1

    # Cek apakah bounding box di dalam frame
    if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
        return False

    # Cek apakah ukurannya sesuai proporsi
    face_area = w * h
    frame_area = frame_width * frame_height
    ratio = face_area / frame_area

    return min_ratio <= ratio <= max_ratio
def base64_to_ndarray(base64_str):
    # Jika base64 ada prefix "data:image/jpeg;base64,...", hapus itu dulu
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    # Decode base64 ke bytes
    img_bytes = base64.b64decode(base64_str)

    # Convert bytes ke numpy array
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)

    # Decode menjadi gambar (BGR image)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img
def show_toast(text: str, duration: int = 5):
    system = platform.system()

    if system == "Windows":
        try:
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast("Notifikasi", text, duration=duration)
        except ImportError:
            print("win10toast belum terinstall. Install dengan: pip install win10toast")

    elif system == "Linux":
        try:
            import notify2
            notify2.init("Python App")
            n = notify2.Notification("Notifikasi", text)
            n.set_timeout(duration * 1000)  # duration dalam milidetik
            n.show()
        except ImportError:
            print("notify2 belum terinstall. Install dengan: pip install notify2")

    else:
        print(f"Toast belum didukung di OS ini: {system}")
#create_dataset_from_livecam("2155201014","Amir")
#search_face_from_livecam()
#search_face_from_file("gabung.jpg")