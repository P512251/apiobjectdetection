import cv2
import pyttsx3
import torch
from fastapi import FastAPI
import threading

# Initialize FastAPI app
app = FastAPI()

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()

# Load YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_objects():
    cap = cv2.VideoCapture(0)  # Open webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model
        results = model(frame)
        detected_objects = results.pandas().xyxy[0]['name'].tolist()  # Get detected object names

        if detected_objects:
            object_names = ', '.join(detected_objects)
            print(f"Detected: {object_names}")  # Print detected objects
            engine.say(f"Detected {object_names}")  # Convert to speech
            engine.runAndWait()

        # Show camera feed
        cv2.imshow("Live Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.get("/start_detection")
async def start_detection():
    thread = threading.Thread(target=detect_objects)
    thread.start()
    return {"message": "Object detection started"}
