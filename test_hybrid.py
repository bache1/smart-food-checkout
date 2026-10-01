import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
import os

print("=== Inisialisasi AI Smart Food Checkout ===")
print("1. Memuat model Deteksi (YOLO v2)...")
# Memuat model YOLO yang sudah kamu training untuk mendeteksi lokasi makanan
yolo_model = YOLO("runs/detect/bakery_v2/weights/best.pt")

print("2. Memuat model Pengenalan/Embedding (MobileNetV3)...")
# Memuat MobileNet untuk menghasilkan 1000 angka 'Sidik Jari Digital'
weights = models.MobileNet_V3_Small_Weights.DEFAULT
embed_model = models.mobilenet_v3_small(weights=weights)
embed_model.eval() # Mode evaluasi agar model tidak berubah-ubah

# Aturan standardisasi ukuran gambar agar cocok untuk MobileNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# === SISTEM DATABASE PERMANEN ===
DB_FILE = "database_makanan.pt"

# Cek apakah file database sudah ada di hardisk laptop
if os.path.exists(DB_FILE):
    product_database = torch.load(DB_FILE, weights_only=True)
    print(f"📂 Database dimuat! Ada {len(product_database)} produk yang tersimpan.")
else:
    product_database = {}
    print("⚠️ File database belum ada. Database baru dibuat kosong.")

print("\n✅ Sistem Siap!")
print("Webcam Aktif! Tekan 'R' (Register), 'S' (Scan Kasir), 'Q' (Keluar).")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. TAHAP DETEKSI UTAMA
    results = yolo_model(frame, conf=0.4, verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("Smart Food Checkout - Hybrid", annotated_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # ==========================================================
    # FITUR MANAGEMENT APP: REGISTER MULTI-ANGLE (ALA FACE ID)
    # ==========================================================
    if key == ord('r'):
        nama_produk = input("\n[REGISTER] Masukkan nama produk baru: ")
        
        # Daftar instruksi pengambilan sudut gambar
        instruksi = [
            "Tampak ATAS (Top View)", 
            "Tampak SAMPING (Side View)", 
            "Tampak SERONG/SUDUT LAIN (Angle View)"
        ]
        
        vektor_terkumpul = []
        
        # Looping instruksi untuk setiap sudut
        for tahap in instruksi:
            print(f"\n👉 TUGAS: Tunjukkan {tahap}!")
            print("Tekan 'C' (Capture) jika posisi sudah pas di kamera.")
            
            # Loop kamera internal khusus untuk saat instruksi berjalan
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                results_reg = yolo_model(frame, conf=0.4, verbose=False)
                annotated_frame_reg = results_reg[0].plot()
                
                # Menampilkan instruksi UI di atas kamera
                cv2.putText(annotated_frame_reg, f"Instruksi: Tunjukkan {tahap}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated_frame_reg, "Tekan 'C' untuk Capture", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Smart Food Checkout - Hybrid", annotated_frame_reg)
                
                k = cv2.waitKey(1) & 0xFF
                if k == ord('c'): # Tombol jepret ditekan
                    if len(results_reg[0].boxes) > 0:
                        box = results_reg[0].boxes[0]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cropped_food = frame[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Ekstraksi vektor untuk sudut ini
                        rgb_image = cv2.cvtColor(cropped_food, cv2.COLOR_BGR2RGB)
                        input_tensor = preprocess(Image.fromarray(rgb_image)).unsqueeze(0)
                        
                        with torch.no_grad():
                            embedding = embed_model(input_tensor)
                        
                        vektor_terkumpul.append(embedding)
                        print(f"✅ {tahap} berhasil direkam!")
                        break # Pindah ke sudut berikutnya
                    else:
                        print("❌ Makanan tidak terdeteksi oleh AI. Posisikan lebih baik dan tekan 'C' lagi!")
        
        # Jika ke-3 sudut sudah terekam, buat SUPER VEKTOR
        if len(vektor_terkumpul) == 3:
            print("\n🔄 Memproses dan menggabungkan semua sudut pandang...")
            # Menghitung rata-rata dari ke-3 vektor agar kebal dari perubahan posisi
            super_vektor = torch.mean(torch.cat(vektor_terkumpul, dim=0), dim=0, keepdim=True)
            
            # Simpan permanen ke file hardisk
            product_database[nama_produk] = super_vektor
            torch.save(product_database, DB_FILE)
            print(f"🎉 SUCCESS! '{nama_produk}' berhasil didaftarkan secara permanen!")

    # ==========================================================
    # FITUR CASHIER APP: MULTI-OBJECT SCAN
    # ==========================================================
    elif key == ord('s'):
        print("\n--- MULAI SCAN KASIR ---")
        if len(product_database) == 0:
            print("Database kosong! Register makanan dulu pakai tombol 'R'.")
            continue
            
        if len(results[0].boxes) == 0:
            print("Tidak ada makanan yang terdeteksi di nampan.")
            continue

        # Looping setiap objek yang ada di nampan (contoh: 2 Roti, 1 Croissant)
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cropped_food = frame[int(y1):int(y2), int(x1):int(x2)]
            if cropped_food.size == 0: continue
            
            rgb_image = cv2.cvtColor(cropped_food, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(Image.fromarray(rgb_image)).unsqueeze(0)
            
            # Ekstrak sidik jari kasir
            with torch.no_grad():
                scan_embedding = embed_model(input_tensor)
            
            best_match = "Tidak Dikenali"
            highest_score = 0.0
            
            # Cocokkan dengan seluruh data di dalam Hardisk
            for db_nama, db_embedding in product_database.items():
                score = F.cosine_similarity(scan_embedding, db_embedding).item()
                if score > highest_score:
                    highest_score = score
                    best_match = db_nama
            
            # Terapkan Ambang Batas (Threshold)
            THRESHOLD = 0.75 # Minimal harus mirip 75%
            
            if highest_score >= THRESHOLD:
                print(f"Objek {i+1} terdeteksi: {best_match} ({highest_score*100:.1f}%)")
            else:
                print(f"Objek {i+1}: ❌ TIDAK DIKENAL (Mirip {best_match} tapi hanya {highest_score*100:.1f}%)")
                
    elif key == ord('q'):
        print("\nKeluar dari program. Sampai jumpa!")
        break

cap.release()
cv2.destroyAllWindows()