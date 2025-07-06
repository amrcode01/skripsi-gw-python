import cv2
import mediapipe as mp
import requests
import numpy as np
from collections import defaultdict
import time

last_capture_time = {'left': 0, 'center': 0, 'right': 0}
delay_between_frames = 0.2  # jeda 0.5 detik antar capture

temp_frames = defaultdict(list)
max_frames_per_step = 3
captures = defaultdict(list)  # sekarang setiap step simpan list, bukan satu frame

BACKEND_URL = 'http://localhost:8000/api/upload'  # Ganti dengan URL backend-mu

# Inisialisasi FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def is_image_blurry(image, threshold=100.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance

def calculate_yaw(landmarks, image_width):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose_tip = landmarks[1]
    eye_center_x = (left_eye.x + right_eye.x) / 2
    nose_x = nose_tip.x
    offset = (nose_x - eye_center_x) * image_width
    yaw_angle = offset * 0.5
    return yaw_angle

# Variabel penyimpanan sementara
steps = ['left', 'center', 'right']
current_step = 0
captures = {
    'left': None,
    'center': None,
    'right': None
}

def reset():
    global current_step, captures
    current_step = 0
    captures = {'left': None, 'center': None, 'right': None}
    print("🔄 Reset proses")

def post_images(captures):
    try:
        _, left_img = cv2.imencode('.jpg', captures['left'])
        _, center_img = cv2.imencode('.jpg', captures['center'])
        _, right_img = cv2.imencode('.jpg', captures['right'])

        files = {
            'left': ('left.jpg', left_img.tobytes(), 'image/jpeg'),
            'center': ('center.jpg', center_img.tobytes(), 'image/jpeg'),
            'right': ('right.jpg', right_img.tobytes(), 'image/jpeg'),
        }

        response = requests.post(BACKEND_URL, files=files)
        print(f"✅ Data terkirim: {response.status_code}")
    except Exception as e:
        print(f"❌ Gagal kirim ke backend: {e}")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape

    direction_text = ""
    status_text = ""

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            yaw = calculate_yaw(landmarks, w)

            step_name = steps[current_step]
    
            if step_name == 'left' and yaw < -15:
                now = time.time()
                if now - last_capture_time['left'] >= delay_between_frames:
                    blurry, score = is_image_blurry(frame)
                    if not blurry:
                        temp_frames['left'].append((frame.copy(), score))
                        last_capture_time['left'] = now
                        status_text = f"📷 Kiri Frame {len(temp_frames['left'])}/3 (sharpness: {int(score)})"

                        if len(temp_frames['left']) >= max_frames_per_step:
                            captures['left'] = [img for img, _ in temp_frames['left']]
                            status_text = f"✅ 3 Gambar Kiri Disimpan"
                            current_step += 1
                            temp_frames['left'].clear()
                    else:
                        status_text = f"⚠️ Gambar Kiri Buram (sharpness: {int(score)})"

            elif step_name == 'center' and yaw > 0.5:
                now = time.time()
                if now - last_capture_time['center'] >= delay_between_frames:
                    blurry, score = is_image_blurry(frame)
                    if not blurry:
                        temp_frames['center'].append((frame.copy(), score))
                        last_capture_time['center'] = now
                        status_text = f"📷 Tengah Frame {len(temp_frames['center'])}/3 (sharpness: {int(score)})"

                        if len(temp_frames['center']) >= max_frames_per_step:
                            captures['center'] = [img for img, _ in temp_frames['center']]
                            status_text = f"✅ 3 Gambar Tengah Disimpan"
                            current_step += 1
                            temp_frames['center'].clear()
                    else:
                        status_text = f"⚠️ Gambar Tengah Buram (sharpness: {int(score)})"


            elif step_name == 'right' and yaw > 15:
                now = time.time()
                if now - last_capture_time['right'] >= delay_between_frames:
                    blurry, score = is_image_blurry(frame)
                    if not blurry:
                        temp_frames['right'].append((frame.copy(), score))
                        last_capture_time['right'] = now
                        status_text = f"📷 Kanan Frame {len(temp_frames['right'])}/3 (sharpness: {int(score)})"

                        if len(temp_frames['right']) >= max_frames_per_step:
                            captures['right'] = [img for img, _ in temp_frames['right']]
                            status_text = f"✅ 3 Gambar Kanan Disimpan"
                            current_step += 1
                            temp_frames['right'].clear()
                    else:
                        status_text = f"⚠️ Gambar Kanan Buram (sharpness: {int(score)})"


            if current_step >= 3:
                status_text = "📤 Mengirim ke backend..."
                cv2.putText(frame, status_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                for step in ['left', 'center', 'right']:
                    for i, img in enumerate(captures[step]):
                        window_name = f"{step.capitalize()}_{i+1}"
                        cv2.imshow(window_name, img)


                cv2.imshow("Face Capture", frame)
                cv2.waitKey(500)  # beri jeda agar terlihat
                post_images(captures)
                reset()

    # Instruksi tampilan
    if current_step < 3:
        direction = steps[current_step]
        direction_text = f"▶ Arahkan wajah ke {direction.upper()}"
    else:
        direction_text = "Tunggu..."

    # Tampilkan info di layar
    cv2.putText(frame, direction_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    if status_text:
        cv2.putText(frame, status_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Capture", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == ord('r'):
        reset()

cap.release()
cv2.destroyAllWindows()
