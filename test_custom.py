import cv2
from ultralytics import YOLO

print("Memuat model custom bakery...")
# Kita panggil file best.pt hasil trainingmu sendiri!
model = YOLO("runs/detect/bakery_v2/weights/best.pt")

print("Membuka webcam...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame kamera.")
        break

    results = model(frame, conf=0.4, verbose=False)
    
    # Ambil frame yang sudah digambar kotaknya oleh YOLO
    annotated_frame = results[0].plot()
    cv2.imshow("Smart Food Checkout - Custom Model", annotated_frame)

    # LOOPING UNTUK MEMOTONG SETIAP OBJEK YANG TERDETEKSI
    for box in results[0].boxes:
        # Ambil koordinat kotak: x1 (kiri), y1 (atas), x2 (kanan), y2 (bawah)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # Potong (crop) gambar asli menggunakan koordinat tersebut
        # OpenCV menggunakan format frame[y_start:y_end, x_start:x_end]
        cropped_food = frame[int(y1):int(y2), int(x1):int(x2)]
        
        # Tampilkan hasil potongan di jendela baru
        cv2.imshow("Hasil Crop (Akan dikirim ke Embedding)", cropped_food)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()