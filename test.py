import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier


# Constants
offset = 20
imgsize = 300
folder = "Data/B"
counter = 0

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
classifier=Classifier("Model/keras_model.h5","Module/labels.txt")
detector = HandDetector(maxHands=1)
label=["A","B","C"]

while True:
    success, img = cap.read()
    imgoutput=img.copy()
    if not success:
        continue

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure cropping stays within bounds
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        # Create white image
        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255

        aspectRatio = h / w

        if aspectRatio > 1:  # Height is greater than width
            k = imgsize / h
            wcal = int(k * w)
            imgResize = cv2.resize(imgCrop, (wcal, imgsize))
            wGap = (imgsize - wcal) // 2
            imgwhite[:, wGap:wGap + wcal] = imgResize
            prediction, index = classifier.getPrediction(imgwhite,draw=False)
            print(prediction,index)

        else:  # Width is greater than height
            k = imgsize / w
            hcal = int(k * h)
            imgResize = cv2.resize(imgCrop, (imgsize, hcal))
            hGap = (imgsize - hcal) // 2
            imgwhite[hGap:hGap + hcal, :] = imgResize
            prediction, index = classifier.getPrediction(imgwhite,draw=False)

        cv2.rectangle(imgoutput,(x-offset+100,y-offset-50+50),(150,50),(255,255,255),cv2.FILLED)
        cv2.putText(imgoutput,label[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,0,255),2)
        cv2.rectangle(imgoutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)


        # Show images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgwhite)

    cv2.imshow("Image", imgoutput)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
