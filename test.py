import os
import cv2
import json
from ultralytics import YOLO
import time
import threading
from amr_helper.face_recog import create_dataset_from_image

def warmup_model():
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy)


model = YOLO('yolov11n-face.pt')
threading.Thread(target=warmup_model).start()

def process_folder(folder_path):
    time.sleep(2)
    hasil_dataset = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # cek hanya file gambar
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                base_name = os.path.splitext(file_name)[0]  # buang ekstensi
                nama, nim = base_name.rsplit("_", 1)  # pecah berdasarkan "_"
                nama = nama.strip()
                nim = nim.strip()
            except Exception:
                print(f"[WARNING] Format nama file tidak sesuai (nama_nim): {file_name}")
                continue

            image = cv2.imread(file_path)
            start_time = time.time()
            predict = model.predict(source=image)
            end_time = time.time()  # ⬅️ selesai prediksi
            durasi = end_time - start_time
            boxes = predict[0].boxes
            if boxes and len(boxes.xyxy) > 0:
                box = boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_img = image[y1:y2, x1:x2]
                result_json = create_dataset_from_image(face_img, nim, nama)
                result = json.loads(result_json)
                if result.get("status"):
                    hasil_dataset.append({
                        "nim": nim,
                        "nama": nama,
                        "waktu_yolo": round(durasi, 4)  # detik
                    })
                else:
                    print(f"Gagal memproses {file_name}: {result['pesan']}")

    return hasil_dataset

create_data = (process_folder("dataset_test"))
output_path = "dataset_test/dataset.json"
with open(output_path, "w") as f_json:
    json.dump(create_data, f_json, indent=2)

print(f"Dataset berhasil disimpan ke {output_path}, total {len(create_data)} data.")

