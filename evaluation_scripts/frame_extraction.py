import os
import cv2


# CONFIGURATION
WS_PATH = os.path.dirname(os.getcwd())
SRC_PATH = os.path.join(WS_PATH, 'src')
VIDEO_DIR = os.path.join(SRC_PATH, 'esp32cam_videos')
OUTPUT_DIR = os.path.join(SRC_PATH, 'extracted_frames')
TARGET_FRAMES = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get video files
video_files = [
    f for f in os.listdir(VIDEO_DIR)
    if f.endswith(('.mp4', '.avi', '.mov'))
]

print(f'Found {len(video_files)} videos')

# Process each video
for video_file in video_files:

    print(f'\nProcessing: {video_file}')

    video_path = os.path.join(VIDEO_DIR, video_file)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print('Failed to open video')
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // TARGET_FRAMES, 1)

    print(f'Total frames: {total_frames}')

    # Video naming
    video_name = os.path.splitext(video_file)[0]

    saved_count = 0
    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Save each frame
        if frame_idx % frame_interval == 0:

            frame_name = f'{video_name}-frame_{saved_count:03d}.jpg'  # Custom

            frame_path = os.path.join(
                OUTPUT_DIR,
                frame_name
            )

            cv2.imwrite(frame_path, frame)

            saved_count += 1

            # stop jika gambar sudah lebih dari jumlah target per video
            if saved_count >= TARGET_FRAMES:
                break

        frame_idx += 1

    cap.release()

    print(f'Saved {saved_count} frames')

print('\nFrame extraction completed.')