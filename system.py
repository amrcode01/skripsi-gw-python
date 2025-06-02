import cv2
from amr_helper.face_recog import search_face_from_face,curl_post
import time
import json
import base64
from ultralytics import YOLO

model = YOLO('yolov11n-face.pt')
cap = cv2.VideoCapture(0)

prev_count = 0
stable_count = 0
countdown_start = None
countdown_duration = 3  # detik
paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
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

                    search_result = search_face_from_face(face_img)
                    search_result = json.loads(search_result)
                    if(search_result.get("status")):                        
                        #is_live = liveness_model.predict(face_img) nanti tambahkan deteksi muka orang atau gambar
                        #label = "LIVE" if is_live > 0.8 else "PHOTO"
                        print("Ditemukan ",search_result.get("data").get("nama"))
                        cv2.imshow(search_result.get("data").get("nama"), face_img)
                    else:
                        _, buffer = cv2.imencode('.jpg', face_img)
                        img_base64 = base64.b64encode(buffer).decode('utf-8')
                        data_register = {
                            "img_code" : img_base64,
                            "is_user" : False
                        }
                        post_register = curl_post(data_register)
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
        cv2.imshow("YOLO Wajah + Countdown", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('p'):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
