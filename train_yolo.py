from ultralytics import YOLO

# 1. Kita muat model dasar yang sudah ada (Transfer Learning)
model = YOLO("yolo26n.pt")

print("Memulai proses training AI...")

# 2. Proses training dimulai
results = model.train(
    data="C:/Users/LOQ/smart-food-checkout/datasets/bakery_data/bakery.v5i.yolov8/data.yaml",
    epochs=100,       # Kita naikkan menjadi 100 putaran agar AI benar-benar belajar
    imgsz=640,
    batch=4,
    name="bakery_v2"  # Kita simpan di folder baru (v2) agar hasil sebelumnya tidak tertimpa
)

print("Training selesai! Silakan cek folder 'runs/detect/bakery_v2'")   