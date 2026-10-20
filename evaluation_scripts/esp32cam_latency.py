# ============================================================
# ESP32-CAM ↔ YOLO SERVER LATENCY TEST
# ============================================================
# FITUR:
# 1. Warm-up stream sebelum pengujian
# 2. Preview deteksi YOLO realtime
# 3. Latency dicatat per detik
# 4. Latency dalam satuan ms
# 5. Grafik latency vs waktu
# ============================================================

import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

ESP32_STREAM_URL = "http://10.149.241.19:80/stream"

MODEL_PATH = 'E:/Kuliah/Skripsi/Project/BISINDO-Vowels-Detection-YOLOv8/train_results/yolov8m/weights/best.pt'

TEST_DURATION = 31  # detik

WARMUP_DURATION = 5  # detik sebelum recording latency

CONF_THRESHOLD = 0.25

# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL_PATH)

# ============================================================
# CONNECT TO STREAM
# ============================================================

cap = cv2.VideoCapture(ESP32_STREAM_URL)

if not cap.isOpened():
    print("Failed to connect to ESP32-CAM")
    exit()

print("Connected to ESP32-CAM")

# ============================================================
# WARM-UP STREAM
# ============================================================

print(f"\nWarming up stream for {WARMUP_DURATION} seconds...")

warmup_start = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    elapsed_warmup = time.time() - warmup_start

    display_frame = frame.copy()

    remaining = int(
        np.ceil(WARMUP_DURATION - elapsed_warmup)
    )

    # ========================================================
    # DISPLAY WARM-UP TEXT
    # ========================================================

    cv2.putText(
        display_frame,
        f"Starting in {max(0, remaining)}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        3,
        cv2.LINE_AA
    )

    cv2.imshow("ESP32-CAM Warmup", display_frame)

    if elapsed_warmup >= WARMUP_DURATION:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

print("\nStarting latency recording...")

# ============================================================
# STORAGE
# ============================================================

timestamps = []
latencies_ms = []

# ============================================================
# START TEST
# ============================================================

global_start = time.time()

last_recorded_second = -1

while True:

    current_global = time.time()

    elapsed_global = current_global - global_start

    # ========================================================
    # STOP CONDITION
    # ========================================================

    if elapsed_global >= TEST_DURATION:
        break

    # ========================================================
    # START LATENCY TIMER
    # ========================================================

    start_latency = time.time()

    # ========================================================
    # READ FRAME
    # ========================================================

    ret, frame = cap.read()

    if not ret:
        continue

    # ========================================================
    # YOLO INFERENCE
    # ========================================================

    results = model.predict(
        frame,
        conf=CONF_THRESHOLD,
        verbose=False
    )[0]

    # ========================================================
    # END LATENCY TIMER
    # ========================================================

    end_latency = time.time()

    latency_ms = (end_latency - start_latency) * 1000

    # ========================================================
    # RECORD LATENCY PER SECOND
    # ========================================================

    current_second = int(elapsed_global)

    if current_second != last_recorded_second:

        timestamps.append(current_second)
        latencies_ms.append(latency_ms)

        print(
            f"Second: {current_second}s | "
            f"Latency: {latency_ms:.2f} ms"
        )

        last_recorded_second = current_second

    # ========================================================
    # DRAW YOLO RESULT
    # ========================================================

    plotted = results.plot()

    # ========================================================
    # DISPLAY LATENCY
    # ========================================================

    cv2.putText(
        plotted,
        f"Latency: {latency_ms:.2f} ms",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        plotted,
        f"Time: {elapsed_global:.1f}s",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # ========================================================
    # SHOW PREVIEW
    # ========================================================

    cv2.imshow("YOLO Detection Preview", plotted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================================
# RELEASE
# ============================================================

cap.release()

cv2.destroyAllWindows()

# ============================================================
# CREATE DATAFRAME
# ============================================================

latency_df = pd.DataFrame({
    "time_sec": timestamps,
    "latency_ms": latencies_ms
})

# ============================================================
# SAVE CSV
# ============================================================

latency_df.to_csv(
    "esp32cam_latency_test.csv",
    index=False
)

# ============================================================
# STATISTICS
# ============================================================

avg_latency = np.mean(latencies_ms)
min_latency = np.min(latencies_ms)
max_latency = np.max(latencies_ms)

print("\n==============================")
print("LATENCY TEST RESULTS")
print("==============================")

print(f"Average Latency : {avg_latency:.2f} ms")
print(f"Minimum Latency : {min_latency:.2f} ms")
print(f"Maximum Latency : {max_latency:.2f} ms")

# ============================================================
# LATENCY GRAPH
# ============================================================

plt.figure(figsize=(12,6))

plt.plot(
    timestamps,
    latencies_ms,
    marker='o'
)

plt.xlabel("Time (seconds)")
plt.ylabel("Latency (ms)")
plt.title("ESP32-CAM to YOLO Server Latency")

plt.grid(True)

plt.show()