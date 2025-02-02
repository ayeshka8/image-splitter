from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from ultralytics import YOLO
import torch

app = Flask(__name__)

UPLOAD_FOLDER = "../backend/uploads/"
PROCESSED_FOLDER = "../backend/uploads/processed/"

os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Using the nano version for faster inference

def non_max_suppression_fast(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        suppress = [last]

        for pos in idxs[:-1]:
            xx1 = max(x1[last], x1[pos])
            yy1 = max(y1[last], y1[pos])
            xx2 = min(x2[last], x2[pos])
            yy2 = min(y2[last], y2[pos])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[pos]

            if overlap > overlap_thresh:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    return boxes[pick].astype("int")

@app.route("/process", methods=["POST"])
def process_image():
    data = request.json
    image_filename = data["image_path"]
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)

    # Load the image
    image = cv2.imread(image_path)
    original = image.copy()

    # Perform object detection using YOLO
    results = model(image)

    # Extract bounding boxes
    boxes = [box.xyxy[0].cpu().numpy() for box in results[0].boxes]

    # Apply Non-Maximum Suppression (NMS)
    nms_boxes = non_max_suppression_fast(boxes)

    # Extract and save detected regions
    images_saved = []
    for i, (x1, y1, x2, y2) in enumerate(nms_boxes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Filter small detections
        if (x2 - x1) > image.shape[1] * 0.2 and (y2 - y1) > image.shape[0] * 0.2:
            ROI = original[y1:y2, x1:x2]
            output_path = f"{PROCESSED_FOLDER}split_{i}.png"
            cv2.imwrite(output_path, ROI)
            images_saved.append(output_path.replace(PROCESSED_FOLDER, "processed/"))

    return jsonify({"images": images_saved})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)