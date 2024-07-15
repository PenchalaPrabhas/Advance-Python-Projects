from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# Define the paths to the prototxt and model files
prototxt_path = r"C:\Users\DELL\Desktop\myprojects\kant\MobileNetSSD_deploy.prototxt.txt"
model_path = r"C:\Users\DELL\Desktop\myprojects\kant\MobileNetSSD_deploy.caffemodel"

def main():
    # Load the class labels
    CLASSES = ["aeroplane", "background", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # Load the pre-trained model
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Initialize the video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # Start the FPS counter
    fps = FPS().start()

    while True:
        frame = vs.read()
        frame = cv2.flip(frame, 1)  # Flip horizontally

        frame = imutils.resize(frame, width=400)

        (h, w) = frame.shape[:2]
        resized_image = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(resized_image, (1 / 127.5), (300, 300), 127.5, swapRB=True)

        net.setInput(blob)
        predictions = net.forward()

        for i in np.arange(0, predictions.shape[2]):
            confidence = predictions[0, 0, i, 2]
            if confidence > 0.2:  # Minimum confidence level
                idx = int(predictions[0, 0, i, 1])
                box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        fps.update()

    fps.stop()
    print("[INFO] Elapsed Time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approximate FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()
