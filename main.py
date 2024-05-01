import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
# variables
width, height = 1280, 720
folderPath = "PPT_Images"

# camera setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# get the list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

# variables
imgNumber = 0
hs, ws = 120, 213  # dimensions de la webcam
gestureThreshold = 300
buttonPressed = False
buttonCounter = 0
buttonDelay = 30

#Hand detector

detector = HandDetector(detectionCon=0.8,maxHands=1)

while True:
    # import images
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # Redimensionner l'image de la diapositive pour s'adapter à l'écran
    imgCurrent_resized = cv2.resize(imgCurrent, (width, height))
    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (255,255,255), 4)

    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList']

        # Constrain values for easier drawing
        #indexFinger = lmList[8][0], lmList[8][1]
        xVal = int(np.interp(lmList[8][0], [width//2, w], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height-150], [0, height]))
        indexFinger = xVal, yVal
        #print(fingers)

        if cy <= gestureThreshold: # if hand is at the height of the face
            # gesture 1 - left
            if fingers == [1,0,0,0,0]:
                print("Left")
                if imgNumber > 0:
                    buttonPressed = True
                    imgNumber -= 1
            # gesture 2 - right
            if fingers == [0, 0, 0, 0, 1]:
                print("Right")
                if imgNumber<len(pathImages)-1:
                    buttonPressed = True
                    imgNumber += 1

        # fingers 3 - Show Pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent_resized, indexFinger, 10, (127, 0, 255), cv2.FILLED)

    # Button pressed iterations
    if buttonPressed:
        buttonCounter +=1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False


    # Superposer l'image de la webcam en haut à droite de l'image de la diapositive
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent_resized.shape
    imgCurrent_resized[0:hs, w - ws:w] = imgSmall
    cv2.imshow("Image", img)
    cv2.imshow("Slides", imgCurrent_resized)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
