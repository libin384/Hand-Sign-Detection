import numpy as np
import math
import cv2
from cvzone.HandTrackingModule import HandDetector
import time

# Constants
offset = 20
imgsize = 300
folder = "Data/B"
counter = 0

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

while True:
    success, img = cap.read()
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

        else:  # Width is greater than height
            k = imgsize / w
            hcal = int(k * h)
            imgResize = cv2.resize(imgCrop, (imgsize, hcal))
            hGap = (imgsize - hcal) // 2
            imgwhite[hGap:hGap + hcal, :] = imgResize

        # Show images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgwhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgwhite)
        print(f"Saved Image {counter}")

cap.release()
cv2.destroyAllWindows()
