import kagglehub
import os
import shutil
import cv2
import torch
import numpy as np
import time
import os
from ultralytics import YOLO


# Define the new destination directory
new_destination = "dataset"  # Example: change this to your desired path

# Download the dataset to the default location
path = kagglehub.dataset_download(
    "mehantkammakomati/atm-anomaly-video-dataset-atmav"
)

print("Path to dataset files (default location):", path)

# Check if the new directory exists, and create it if it doesn't
if not os.path.exists(new_destination):
    os.makedirs(new_destination)

# Move the downloaded dataset to the new directory
try:
    # List the files in the source path
    files = os.listdir(path)

    # Move each file to the new destination
    for file in files:
        source_file = os.path.join(path, file)
        destination_file = os.path.join(new_destination, file)
        shutil.move(source_file, destination_file)
        print(f"Moved '{file}' from '{path}' to '{new_destination}'")

    # Remove the now empty source directory
    shutil.rmtree(path)
    print(f"Removed empty directory: '{path}'")

    #change the dataset_path variable
    dataset_path = new_destination
    print(f"The dataset path has now been set to '{dataset_path}'")
except FileNotFoundError as e:
    print(f"Error moving dataset: {e}")

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Using the nano version for efficiency

# Define classes of interest
classes_of_interest = ['person', 'hand', 'weapon', 'card', 'phone', 'wallet']

# Suspicious activity flags
suspicious_activities = {
    'touching_non_screen_area': False,
    'bringing_weapon': False,
    'blocking_camera': False,
}

# Dataset path
dataset_path = r"/tmp/atm_dataset/ATMA-V Dataset"
videos_path = os.path.join(dataset_path, "videos")
labels_path = os.path.join(dataset_path, "labels")

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
recording = False
out = None
last_suspicious_time = 0
record_duration = 10  # seconds


def check_suspicious_activities(detections, frame):
    height, width, _ = frame.shape
    atm_screen_area = (width // 3, height // 3, width * 2 // 3, height * 2 // 3)  # Define ROI for screen
    suspicious_activities.update({'touching_non_screen_area': False, 'bringing_weapon': False, 'blocking_camera': False})

    for detection in detections:
        label = detection["name"]
        x1, y1, x2, y2 = map(int, [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])

        if label == 'hand':
            if not (atm_screen_area[0] < (x1 + x2) / 2 < atm_screen_area[2] and atm_screen_area[1] < (y1 + y2) / 2 < atm_screen_area[3]):
                suspicious_activities['touching_non_screen_area'] = True

        if label == 'weapon':
            suspicious_activities['bringing_weapon'] = True

        if label == 'person' and y1 < height // 4:  # Person covering the camera
            suspicious_activities['blocking_camera'] = True

    return any(suspicious_activities.values())

# Process each video in the dataset
for video_file in os.listdir(videos_path):
    if video_file.endswith(('.mp4', '.avi', '.mov')):  # Check for video file extensions
        video_path = os.path.join(videos_path, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_file}")
            continue

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)  # Run YOLOv8 model
            detections = results[0].boxes.data.cpu().numpy()  # Get detections

            detection_list = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                label = model.names[int(cls)]
                if label in classes_of_interest:
                    detection_list.append({
                        'name': label,
                        'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2
                    })
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label}: {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if check_suspicious_activities(detection_list, frame):
                cv2.putText(frame, "DANGER", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                current_time = time.time()
                if not recording:
                    last_suspicious_time = current_time
                    output_filename = f"ATM_alert_{int(time.time())}.avi"
                    out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                    recording = True

            if recording:
                out.write(frame)
                if time.time() - last_suspicious_time > record_duration:
                    recording = Falsqqqe
                    out.release()

            cv2.imshow("ATM Security System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        if recording:
            out.release()

print("Processing complete.")
