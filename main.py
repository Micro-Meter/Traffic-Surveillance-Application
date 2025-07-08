import cv2
import numpy as np
import tensorflow as tf
import time
import psutil
from scipy.spatial import distance as dist
import os
import gdown

model_path = "model/vehicle_cnn_model.h5"
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1u0x2qzGriNfNIiAz3J8oDvi7nHKgbwrd"  # ← Replace with your real ID
    gdown.download(url, model_path, quiet=False)


LABELS = ['Car', 'Bike', 'Truck']
model = tf.keras.models.load_model('model/vehicle_cnn_model.h5')
IMG_WIDTH = 100
IMG_HEIGHT = 120

next_object_id = 1
tracked_objects = {}
store_time = []
def update_tracked_objects(input_centroids):
    global next_object_id, tracked_objects

    if len(tracked_objects) == 0:
        for centroid in input_centroids:
            tracked_objects[next_object_id] = {
                "current": centroid,
                "prev": centroid,
                "start_frame": 0,
                "end_frame": 0,
                "crossed_240": False,
                "crossed_200": False
            }
            next_object_id += 1
        return tracked_objects

    object_ids = list(tracked_objects.keys())
    object_centroids = [v["current"] for v in tracked_objects.values()]

    D = dist.cdist(np.array(object_centroids), np.array(input_centroids))
    rows = D.min(axis=1).argsort()
    cols = D.argmin(axis=1)[rows]

    used_rows = set()
    used_cols = set()
    new_tracked = {}

    for row, col in zip(rows, cols):
        if row in used_rows or col in used_cols:
            continue
        if D[row][col] > 50:
            continue

        object_id = object_ids[row]
        matched_centroid = input_centroids[col]
        old_data = tracked_objects[object_id]

        new_tracked[object_id] = {
            "prev": old_data["current"],
            "current": matched_centroid,
            "start_frame": old_data["start_frame"],
            "end_frame": old_data["end_frame"],
            "crossed_240": old_data["crossed_240"],
            "crossed_200": old_data["crossed_200"]
        }
        used_rows.add(row)
        used_cols.add(col)

    for i, centroid in enumerate(input_centroids):
        if i not in used_cols:
            new_tracked[next_object_id] = {
                "prev": centroid,
                "current": centroid,
                "start_frame": 0,
                "end_frame": 0,
                "crossed_240": False,
                "crossed_200": False
            }
            next_object_id += 1

    tracked_objects = new_tracked
    return tracked_objects

cap = cv2.VideoCapture("Resources/traffic_5.mp4")
ret, prev_frame = cap.read()
if not ret:
    print("Error reading video.")
    exit()

prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
threshold = 20
time_1 = time.time()

frame_width = 640
frame_height = 360

video_fps = cap.get(cv2.CAP_PROP_FPS)
out =  cv2.VideoWriter('new_output2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (frame_width, frame_height))
frame_count = 0
while True:
    ret, curr_frame = cap.read()
    if not ret:
        break
    frame_count += 1
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = curr_gray.astype(np.int16) - prev_frame.astype(np.int16)

    dvs_frame = np.full_like(curr_gray, 128, dtype=np.uint8)
    dvs_frame[diff > threshold] = 255
    dvs_frame[diff < -threshold] = 0

    gray_bg = np.full_like(dvs_frame, 128)
    mask = np.zeros_like(dvs_frame)
    points = np.array([[445, 45], [322, 50], [159, 288], [630, 285]])
    cv2.fillPoly(mask, [points], 255)
    roi_gray = np.where(mask == 255, dvs_frame, gray_bg)
    cv2.imshow('DVS', roi_gray)

    binary1 = cv2.inRange(roi_gray, 255, 255)
    binary0 = cv2.inRange(roi_gray, 0, 0)
    binary = cv2.bitwise_or(binary1, binary0)
    kernel = np.ones((7, 7), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.medianBlur(binary, 5)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    input_centroids = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 150:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2

        roi = dvs_frame[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi, (IMG_HEIGHT, IMG_WIDTH))
        roi_input = roi_resized.astype('float32') / 255.0
        roi_input = np.expand_dims(roi_input, axis=(0, -1))

        predictions = model.predict(roi_input)
        vehicle_index = np.argmax(predictions)
        vehicle_name = LABELS[vehicle_index]
        confidence = np.max(predictions)

        if confidence < 0.7:
            continue

        input_centroids.append((cx, cy))
        cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(curr_frame, f"{vehicle_name}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 0, 0), 1)


    tracked = update_tracked_objects(input_centroids)

    for object_id, obj in tracked.items():
        x_prev, y_prev = obj["prev"]
        x_curr, y_curr = obj["current"]

        if not obj["crossed_240"] and y_curr <= 210:
            print(f"{object_id}: {x_prev}, {y_prev}, {x_curr}, {y_curr}")
            tracked[object_id]["start_frame"] = frame_count
            tracked[object_id]["crossed_240"] = True

        if obj["crossed_240"] and not obj["crossed_200"] and y_curr <= 100:
            print(f"{object_id}: {x_prev}, {y_prev}, {x_curr}, {y_curr}")
            tracked[object_id]["end_frame"] = frame_count
            tracked[object_id]["crossed_200"] = True
            total_time = (tracked[object_id]["end_frame"] - tracked[object_id]["start_frame"])/video_fps
            store_time.append(total_time)
            print(f"ID {object_id} took {total_time:.5f} seconds from y=240 to y=200")

        vx = x_prev
        vy = 4 * y_curr - 3 * y_prev
        cv2.arrowedLine(dvs_frame, (x_prev, y_prev), (x_prev, vy), 255, 1, tipLength=0.4)
        cv2.arrowedLine(curr_frame, (x_prev, y_prev), (x_prev, vy), (0, 0, 255), 1, tipLength=0.4)
        cv2.putText(curr_frame, f"ID {object_id}", (x_curr + 8, y_curr),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    for i in range (len(store_time)):
        relative_speed = store_time[i]/store_time[0]
        cv2.putText(curr_frame, f"ID : {i+1} relative speed: {relative_speed:0.2f}xx", (10, 120 + i * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 1)

    #cv2.line(curr_frame, (0, 240), (curr_frame.shape[1], 240), (0, 255, 0), 1)
    #cv2.line(curr_frame, (0, 200), (curr_frame.shape[1], 200), (0, 0, 255), 1)

    cpu_fraction = psutil.cpu_percent() / 100
    bars = '|' * int(60 * cpu_fraction)
    memory_fraction = psutil.virtual_memory().percent / 100
    time_0 = time.time()
    latency = (time_0 - time_1) * 1000
    time_1 = time_0


    cv2.putText(curr_frame, f"Input file FPS: {video_fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(curr_frame, f"Latency: {latency:.0f} ms", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(curr_frame, f"CPU Usage: {cpu_fraction:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(curr_frame, f"{bars}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 3)
    cv2.putText(curr_frame, f"Memory: {memory_fraction:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    out.write(curr_frame)
    cv2.imshow('Recognition', curr_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    prev_frame = curr_gray

cap.release()
out.release()
cv2.destroyAllWindows()
