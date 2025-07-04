# Traffic-Surveillance-Application with Object Tracking and Detection using OpenCV & TensorFlow

An real time DVS-style traffic monitoring system for:
- Real-time vehicle detection and tracking
- Vehicle classification (Car, Bike, Truck) using a self trained CNN
- Event-frame generation inspired by Dynamic Vision Sensors (DVS)
- Speed estimation based on movement across predefined virtual lines

# What's Special?
Unlike traditional frame-based detection, this project uses DVS-inspired motion detection — using pixel-level intensity changes between consecutive frames to generate event frames, making it very efficient and responsive to motion.

# Features
- Motion detection via event-based frame differencing (DVS-inspired)
- Real-time object tracking with contour detection and centroid association
- Vehicle classification (Car, Bike, Truck) using a trained CNN model
- Relative speed estimation using timestamps between virtual lines across the road.
- Overlay visualizations with object IDs, motion arrows, and system metrics (CPU, latency, memory).

# Tech Stack and Libraries
- Python 3.10+
- TensorFlow – Vehicle classification
- OpenCV – Image processing & tracking
- NumPy, SciPy – Centroid distance computation
- psutil – System performance monitoring
- os, time – Standard library utilities

## 📁 Project Structure

```
traffic-surveillance/
├── collect_data.py           # Save DVS-style ROIs from video
├── train_model.py            # Train CNN on vehicle images
├── main.py                   # Real-time detection, classification & speed tracking
├── model/
│   └── vehicle_cnn_model.h5  # Saved trained model
├── vehicle_dataset/
│   ├── Car/
│   ├── Bike/
│   └── Truck/
├── Resources/
│   ├── traffic_3.mp4         # For data collection
│   └── traffic_5.mp4         # For real-time inference
└── README.md  
```

---

##  Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/traffic-surveillance.git
   cd traffic-surveillance
   ```

2. **Install dependencies**  
   ```bash
   pip install opencv-python tensorflow numpy scipy psutil
   ```

# Step 1: Collect Training Data (DVS ROIs) 
Extract grayscale event-based ROIs from a video for each class label.
```bash
   python collect_data.py
   ```
Controls:<br>
Press s → Save current ROI (image)<br>
Press n → Switch label (Car → Bike → Truck)<br>
Press p → Pause and resume<br>
Press q → Quit<br>

🛠 Make sure Resources/traffic_3.mp4 exists before running.
