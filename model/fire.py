from pathlib import Path
from ultralytics import YOLO
import cvzone
import cv2
import math

# Running real time from webcam
# cap = cv2.VideoCapture("pic.jpg")
cap = cv2.VideoCapture("fire2.mp4")

model_path = Path("./model/fire.pt")
model = YOLO(model_path)


# Reading the classes
classnames = ["fire"]


def detect_fire(image_path: Path):
    # Load the image using OpenCV
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, (640, 480))

    # Run the YOLO model on the image with stream=True for consistent results with your script
    results = model(img, stream=True)

    # Getting bbox, confidence, and class names information to work with
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0].item()
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50:  # Only draw boxes if confidence is greater than 50%
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(
                    img,
                    f"{classnames[Class]} {confidence}%",
                    [x1 + 8, y1 + 100],
                    scale=1.5,
                    thickness=2,
                )

    # Save the image with the detections
    output_path = image_path.with_name("detected_" + image_path.name)
    cv2.imwrite(str(output_path), img)

    return output_path


def get_best_detection(image_path: Path):
    # Load the image using OpenCV
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, (640, 480))

    # Run the YOLO model on the image with stream=True for consistent results with your script
    results = model(img, stream=True)

    max_confidence = 0
    best_box = None
    best_class = None

    # Getting bbox, confidence, and class names information to work with
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0].item()
            confidence = math.ceil(confidence * 100)
            if confidence > max_confidence:
                max_confidence = confidence
                best_box = box
                best_class = int(box.cls[0])

    # If a box with confidence greater than 50% is found, draw it on the image
    if max_confidence > 50 and best_box is not None and best_class is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
        cvzone.putTextRect(
            img,
            f"{classnames[best_class]} {max_confidence}%",
            [x1 + 8, y1 + 100],
            scale=1.5,
            thickness=1,
        )

    # Save the image with the detection
    output_path = image_path.with_name("detected_" + image_path.name)
    cv2.imwrite(str(output_path), img)

    return output_path


if __name__ == "__main__":
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        result = model(frame, stream=True)

        # Getting bbox,confidence and class names information to work with
        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 50:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cvzone.putTextRect(
                        frame,
                        f"{classnames[Class]} {confidence}%",
                        [x1 + 8, y1 + 100],
                        scale=1.5,
                        thickness=2,
                    )

        cv2.imshow("frame", frame)
        cv2.waitKey(1)
