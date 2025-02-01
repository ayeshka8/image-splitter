from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "../backend/uploads/"
PROCESSED_FOLDER = "../backend/uploads/processed/"

os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route("/process", methods=["POST"])
def process_image():
    data = request.json
    image_filename = data["image_path"]
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)

    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection and contour finding
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    images_saved = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:  # Ignore small detections
            cropped = image[y:y+h, x:x+w]
            output_path = f"{PROCESSED_FOLDER}split_{i}.png"
            cv2.imwrite(output_path, cropped)
            images_saved.append(output_path.replace(PROCESSED_FOLDER, "processed/"))

    return jsonify({ "images": images_saved })

if __name__ == "__main__":
    app.run(port=5001)
