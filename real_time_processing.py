import cv2
import numpy as np
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import math

# Load trained model
model = load_model("sign_language_model.h5")

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

# Define labels for model output (based on the signs you trained on)
labels = ['A', 'B']  # Add more labels for each sign

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Adjust bounding box
        x -= offset
        y -= offset
        w += 2 * offset
        h += 2 * offset

        # Preprocess hand image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y:y+h, x:x+w]
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Convert image to grayscale and reshape for model input
        imgGray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
        imgGray = imgGray.reshape(1, 300, 300, 1) / 255.0  # Normalize

        # Predict the sign
        prediction = model.predict(imgGray)
        classIndex = np.argmax(prediction)
        sign = labels[classIndex]

        # Display the result
        cv2.putText(img, sign, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
