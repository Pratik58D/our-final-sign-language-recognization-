import cv2 
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300


#save images
folder = ".//data//A"
counter = 0

# Create directory if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    #crop the image
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Expand the bounding box to include all fingers
        x -= offset
        y -= offset
        w += 2 * offset
        h += 2 * offset
        
        #custom image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        imgCrop = img[y:y+h, x:x+w]
        
        imgCropShape = imgCrop.shape
        
        
        aspectRatio = h / w 
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            
            wGap = int(math.ceil((imgSize - wCal) / 2))
        
            imgWhite[0:imgResizeShape[0], wGap:wCal+wGap] = imgResize
        
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            
            hGap = int(math.ceil((imgSize - hCal) / 2))
        
            imgWhite[hGap:hCal+hGap, :] = imgResize
        
        
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("imgWhite", imgWhite)
    
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter +=1
        filename = f'{folder}//Image_{counter}.jpg'
        cv2.imwrite(filename,imgWhite)
        print(counter)
        
    elif key == ord("q"):
        break
        

    
