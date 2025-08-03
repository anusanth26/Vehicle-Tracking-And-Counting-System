# Vehicle Tracking and Counting System

A robust, real-time vehicle tracking and counting system developed using YOLOv8 for object detection and SORT algorithm for tracking. This application processes video streams to detect, track, and count vehicles crossing a designated line.

## Key Features 🚗

* **Real-Time Detection**: Utilizes the high-performance YOLOv8 model for accurate, real-time vehicle detection.
* **Multi-Object Tracking**: Implements the Simple Online and Realtime Tracking (SORT) algorithm to maintain unique IDs for each vehicle across frames.
* **Vehicle Counting**: Accurately counts vehicles as they cross a user-defined virtual line in the video feed.
* **Object Masking**: Focuses detection on specific regions of interest, improving performance and accuracy by ignoring irrelevant areas.
* **Customizable**: Easily adaptable for various video sources and detection classes.

## Technology Stack

* **Primary Language**: Python
* **Computer Vision**: OpenCV
* **Object Detection**: YOLOv8 (Ultralytics)
* **Object Tracking**: SORT (Simple Online and Realtime Tracking)
* **Core Libraries**: NumPy, supervision

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/anusanth26/Vehicle-Tracking-And-Counting-System.git](https://github.com/anusanth26/Vehicle-Tracking-And-Counting-System.git)
    cd Vehicle-Tracking-And-Counting-System
    ```

2.  **Install the required packages:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download YOLOv8 weights:**
    The necessary YOLOv8 model weights (`yolov8l.pt`) will be automatically downloaded by the script on its first run.

## Usage

You can run the system on a video file. Place your input video in the `videos` directory.

Execute the main script from the project's root directory:
```bash
python main.py
