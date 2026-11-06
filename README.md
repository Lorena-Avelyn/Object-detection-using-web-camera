# WORKSHOP 2- Object detection using web camera

- **Name:** Lorena Avelyn R
- **Register Number:** 212224040174

## Overview

In this project, real-time object detection is implemented using the YOLOv4 deep learning model with OpenCV. The main objective is to capture frames from a webcam, process them using the YOLOv4 model, detect objects, and display the results with bounding boxes and labels.

Additionally, a screenshot of the detected output is captured and displayed using Matplotlib for better visualization.

**Note:** The following files are required to execute YOLOv4 object detection:

- `yolov4.weights`
- `yolov4.cfg`
- `coco.names`

These files can be downloaded from the official YOLOv4 GitHub repository.

## Aim

To implement real-time object detection using the YOLOv4 deep learning model and OpenCV by capturing video from a webcam and identifying objects with bounding boxes and labels.

## Software Required

- Anaconda
- Python 3.7
- OpenCV
- NumPy
- Matplotlib
- YOLOv4 Model

## Algorithm

**Step 1:** Import the required libraries such as OpenCV, NumPy, and Matplotlib.

**Step 2:** Load the YOLOv4 network using the configuration file (`yolov4.cfg`) and weight file (`yolov4.weights`).

**Step 3:** Read the COCO class labels from the `coco.names` file.

**Step 4:** Access the webcam using `cv2.VideoCapture(0)` to capture live video frames.

**Step 5:** Convert each captured frame into a blob using `cv2.dnn.blobFromImage()`.

**Step 6:** Pass the blob through the YOLOv4 network to detect objects and obtain their class IDs, confidence scores, and bounding box coordinates.

**Step 7:** Apply Non-Maximum Suppression using `cv2.dnn.NMSBoxes()` to remove overlapping bounding boxes.

**Step 8:** Draw bounding boxes and display the detected object names with their confidence scores.

**Step 9:** Display the real-time object detection output using `cv2.imshow()`.

**Step 10:** Capture and save the detection output as a screenshot and display it using Matplotlib.

**Step 11:** Press the `q` key to stop the video, release the webcam, and close all output windows.

## Program

### Developed By: Lorena Avelyn R

### Register Number: 212224040174

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLOv4 network
net = cv2.dnn.readNet(
    "yolov4.weights",
    "yolov4.cfg"
)

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()

output_layers = [
    layer_names[i - 1]
    for i in net.getUnconnectedOutLayers().flatten()
]

# Set up webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    height, width, channels = frame.shape

    # Prepare image for YOLOv4
    blob = cv2.dnn.blobFromImage(
        frame,
        1 / 255.0,
        (416, 416),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)

    # Get YOLO output
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Process detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        0.5,
        0.4
    )

    # Draw bounding boxes and labels
    if len(indexes) > 0:
        for i in np.array(indexes).flatten():
            x, y, w, h = boxes[i]

            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            color = (0, 255, 0)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    # Display detected objects
    cv2.imshow(
        "YOLOv4 Real-Time Object Detection",
        frame
    )

    # Save screenshot by pressing 's'
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("detected_output.jpg", frame)

    # Exit by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()

# Display saved detection output
captured_image = cv2.imread("detected_output.jpg")

if captured_image is not None:
    captured_image = cv2.cvtColor(
        captured_image,
        cv2.COLOR_BGR2RGB
    )

    plt.imshow(captured_image)
    plt.title("Object Detection Output")
    plt.axis("off")
    plt.show()
```

## Output

<img width="1600" height="1078" alt="61422019-2a69-429a-b6bf-68f7b480c9e9" src="https://github.com/user-attachments/assets/5d4163c2-c03a-4436-95e1-2ed398cd45e4" />

## Result

Thus, real-time object detection was successfully implemented using the YOLOv4 deep learning model and OpenCV. Objects were detected from the webcam feed and displayed with bounding boxes and labels. The detection output was also captured and displayed using Matplotlib.

