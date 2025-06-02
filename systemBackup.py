from ultralytics import YOLO
from amr_helper.face_recog import create_dataset_from_livecam,search_face_from_livecam,search_face_from_face,create_dataset_from_face,curl_post
import cv2
import json
import time
import base64

model = YOLO("yolov11n-face.pt")  # pastikan file ini ada di direktori kerja

faces_data = []

prev_face_count = 0
countdown_duration = 5  # hitungan mundur 5 detik
countdown_start_time = None
is_counting = False
last_face_change_time = time.time()
debounce_delay = 0.5  # delay 0.5 detik untuk menghindari reset spam

for result in model(source=0, stream=True,verbose = False):
    frame = result.orig_img
    boxes = result.boxes
    current_face_count = len(boxes)

    # Reset hitungan jika jumlah wajah berubah (dengan debounce delay)
    if current_face_count != prev_face_count:
        current_time = time.time()
        if current_time - last_face_change_time > debounce_delay:
            is_counting = False
            prev_face_count = current_face_count
            last_face_change_time = current_time

    # Mulai hitungan mundur jika ada wajah dan tidak sedang menghitung
        if current_face_count > 0 and not is_counting:
            is_counting = True
            countdown_start_time = time.time()

    if is_counting:
        elapsed_time = time.time() - countdown_start_time
        remaining_time = max(0, countdown_duration - elapsed_time)
        
        # Tampilkan hitungan mundur di frame
        cv2.putText(frame, f"Countdown: {int(remaining_time)}s", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if remaining_time <= 0:
            print("Loading,Cari Muka")
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                face_img = frame[y1:y2, x1:x2]
                
                # Cari wajah di database
                search_result = search_face_from_face(face_img)
                search_result = json.loads(search_result)
                print(search_result)
                if(search_result.get("status")):
                    cv2.putText(frame, f'{search_result.get("data").get("nama")} - {search_result.get("data").get("nim")}', 
                        (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
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
                    
            is_counting = False  # Reset setelah aksi selesai

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        
        cv2.imshow("YOLOv11m Face Detection + Countdown", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()


#create_dataset_from_livecam("Amir sarifudin","2155201014")
#search_face_from_livecam()
#search_face_from_face("2155201014.jpg")
#create_dataset_from_face("2155201014.jpg","Amir sarifudin","2155201014")