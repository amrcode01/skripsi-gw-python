import cv2
from amr_helper.face_recog import search_face_from_face,curl_post,create_dataset_from_image,base64_to_ndarray,is_yolo_face_valid,show_toast,delete_by_nim
import time
import json
import base64
from ultralytics import YOLO
from flask import Flask, request, jsonify, send_from_directory
import threading
import numpy as np

model = YOLO('yolov11n-face.pt')
cap = cv2.VideoCapture(0)

prev_count = 0
stable_count = 0
countdown_start = None
countdown_duration = 3  # detik
paused = True
last_paused_state = paused  # untuk mendeteksi perubahan state

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({"message": "YOLO Flask API Ready!"})

@app.route('/delete_data', methods=['POST'])
def delete_data():
    try:
        data = request.get_json()
        nim = data.get("nim")
        post_delete = delete_by_nim(nim)
        if not nim:
            return jsonify({"status" : False,"pesan": "Nim tidak ditemukan"})
        else:
            return post_delete
    except Exception as e:
        print(e)
        return jsonify({"status": False, "pesan":"Error Exception"})
@app.route('/add_data', methods=['POST'])
def upload_base64():
    try:
        data = request.get_json()
        nama = data.get("nama")
        nim = data.get("nomor_identitas")
        base64_str = data.get("image")
        if not base64_str:
            return jsonify({"status" : False,"pesan": "Gambar tidak ditemukan"})
        elif not nama:
            return jsonify({"status" : False,"pesan": "Nama tidak boleh kosong"})
        elif not nim:
            return jsonify({"status" : False,"pesan": "Nomor identitas tidak boleh kosong"})
        else: 
            img = base64_to_ndarray(base64_str)
            create_result = create_dataset_from_image(img,nim,nama)
            create_result = json.loads(create_result)
            if create_result.get("status") == True:
                return jsonify({"status": True, "pesan":"Dataset Berhasil dibuat"})
            else:
                return jsonify({"status": False, "pesan":"Dataset Gagal dibuat"})
    except Exception as e:
        print(e)
        return jsonify({"status": False, "pesan":"Error Exception"})


def run_flask():
    app.run(host="0.0.0.0", port=5000)

# Jalankan Flask di thread terpisah
threading.Thread(target=run_flask, daemon=True).start()

while True:
    if paused != last_paused_state:
        # Jika state berubah, tutup semua jendela sebelumnya
        cv2.destroyAllWindows()
        last_paused_state = paused

    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        results = model.predict(source=frame, conf=0.4, verbose=False)
        boxes = results[0].boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)
        # Hitung wajah (ganti ID 0 jika model custom)
        curr_count = sum(1 for cid in class_ids if cid == 0)

        # Jika jumlah berubah dan ≠ 0
        if curr_count != stable_count and curr_count != 0:
            if countdown_start is None or curr_count != prev_count:
                countdown_start = time.time()  # mulai hitung mundur
                prev_count = curr_count
        else:
            countdown_start = None
            stable_count = curr_count  # tidak berubah atau kembali ke 0

        # Tampilkan countdown jika sedang berlangsung
        if countdown_start:
            elapsed = time.time() - countdown_start
            remaining = countdown_duration - elapsed
            if remaining <= 0: #Disini waktu habis
                stable_count = curr_count
                countdown_start = None
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face_img = frame[y1:y2, x1:x2]
                    
                    cek_valid_face = is_yolo_face_valid(x1, y1, x2, y2, w, h)
                    print(f"Valid Face {cek_valid_face}")
                    search_result = search_face_from_face(face_img)
                    search_result = json.loads(search_result)

                    _, buffer = cv2.imencode('.jpg', face_img)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')

                    if(search_result.get("status")):                        
                        #is_live = liveness_model.predict(face_img) nanti tambahkan deteksi muka orang atau gambar
                        #label = "LIVE" if is_live > 0.8 else "PHOTO"
                        print(f"Ditemukan {search_result.get("data").get("nama")} dengan score {search_result.get("data").get("score")}")
                        
                        data_hadir = {
                            "nim" : search_result.get("data").get("nim"),
                            "score" : search_result.get("data").get("score"),
                            "img_code" : img_base64,
                            "is_user" : True
                        }
                        try:
                            post_hadir = curl_post(data_hadir)
                            show_toast(post_hadir.get("pesan"))
                        except Exception as e:
                            print(e)
                            show_toast("Gagal Post Kehadiran")

                        
                    else:
                        data_register = {
                            "img_code" : img_base64,
                            "is_user" : False
                        }
                        post_register = curl_post(data_register)
                        print(f"Post register {post_register}")
                        
            else:
                cv2.putText(frame, f"Stabil dalam {int(remaining)+1}s",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)

        # Tambahkan info wajah terdeteksi
        cv2.putText(frame, f"Wajah stabil: {stable_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        annotated_frame = results[0].plot()
        annotated_frame[0:100, 0:400] = frame[0:100, 0:400]  # tempel teks di pojok

        cv2.imshow("YOLO Wajah + Countdown", annotated_frame)

    else:
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, f"Yolo Proses Dipause",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        cv2.putText(blank_frame, f"untuk melanjutkan pencet p",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.imshow("Yolo STOP",blank_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('p'):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
