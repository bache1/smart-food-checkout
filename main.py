import cv2
from ultralytics import YOLO

print("Memuat model YOLO...")

# Model YOLO kecil untuk pengujian
model = YOLO("yolo26n.pt")

print("Membuka webcam...")

# 0 = webcam utama
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam tidak dapat dibuka.")
    exit()

print("Webcam berhasil dibuka.")
print("Tekan Q untuk keluar.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Tidak dapat membaca kamera.")
        break

    # Deteksi objek
    results = model(
        frame,
        conf=0.5,
        verbose=False
    )

    # Tampilkan bounding box
    annotated_frame = results[0].plot()

    cv2.imshow(
        "Smart Food Checkout - YOLO",
        annotated_frame
    )

    # Tekan Q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()