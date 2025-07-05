import cv2
import time
import numpy as np
import os

THRESHOLD = 10
LABELS = ['Car', 'Bike', 'truck']
SAVE_DIR = 'vehicledataset'


for vehicles in LABELS:
    os.makedirs(os.path.join(SAVE_DIR, 'LABELS'), exist_ok=True)

count = 0
vehicle_index = 0

threshold = 20
cap = cv2.VideoCapture("Resources/traffic_5.mp4")
ret, prev_frame = cap.read()

if not ret:
    print("Error reading video.")
    exit()


prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('outputdvs.mp4', cv2.VideoWriter_fourcc(*'XVID'),fps, (640, 360))

while True:
    ret, curr_frame= cap.read()
    if not ret:
        break

    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = curr_gray.astype(np.int16) - prev_frame.astype(np.int16)
    dvs_frame = np.full_like(curr_gray, 128, dtype=np.uint8)
    dvs_frame[diff > threshold] = 255
    dvs_frame[diff < -threshold] = 0


    # Define ROAD-ONLY coordinates
    gray_bg = np.full_like(dvs_frame, 128)  # full gray frame
    mask = np.zeros_like(dvs_frame)
    points = np.array([[445, 45], [322, 50], [159, 288], [630, 285]])
    cv2.fillPoly(mask, [points], 255)
    # Combine ROI with gray background
    road_gray = np.where(mask == 255, dvs_frame, gray_bg)
    cv2.imshow("road only", road_gray)
    road_bgr = cv2.cvtColor(road_gray, cv2.COLOR_GRAY2BGR)
    out.write(road_bgr)

    binary1 = cv2.inRange(road_gray, 255, 255)
    binary0 = cv2.inRange(road_gray, 0, 0)
    binary = cv2.bitwise_or(binary1, binary0)
    # remove noise
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
        cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi = dvs_frame[y:y + h, x:x + w]

        if key == ord('s'):
        # Save current event frame as training image
            filename = os.path.join(SAVE_DIR, LABELS[vehicle_index], f"{int(time.time() * 1000)}.png")
            cv2.imwrite(filename, roi)
            print(f"Saved {filename}")
            count += 1

    cv2.putText(curr_frame, f"vehicle: {LABELS[vehicle_index]} | saved image count: ({count})", (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(curr_frame, f"press 'n' to change vehicle, 's' to save, 'p' to pause, 'q' to exit", (100, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("curr", curr_frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        vehicle_index = (vehicle_index + 1) % len(LABELS)
        print(f"Switched to: {LABELS[vehicle_index]}")

    paused = False
    if key == ord('p'):  # Pause when 'p' is pressed
        if paused:
            cv2.waitKey(30)
            paused = False
        else:
            paused = True
            print("Paused. Press any key to continue...")
            cv2.waitKey(0)


    prev_frame = curr_gray


cap.release()
cv2.destroyAllWindows()
