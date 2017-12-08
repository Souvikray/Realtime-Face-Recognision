import cv2
import numpy as np
from IdentifyFace import *

face_person = {1: "Harry", 2: "John", 3: "Chang", 4: "Zhao", 5: "Alex", 6: "Chow",
               7: "Raju", 8: "Hatori", 9: "David", 10: "Subhash", 11: "Amy", 12: "Harry",
               13: "Subramaniyam", 14: "Xing", 15: "Mike", 16: "Souvik"}

img = cv2.imread("mypic.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.05, 8)
for face in faces:
    x, y, w, h = face
    face_region = gray[y:y + h, x:x + w]
    person_number, confidence = face_recognizer.predict(face_region)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    if face_person[person_number] is "Souvik":
        cv2.putText(img, face_person[person_number], (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))  # (image, text, text-coordinate, font, font-size, text-color)

# display the image
cv2.imshow("Face Recognized", img)
# hold the window
cv2.waitKey(0)
# destroy all windows
cv2.destroyAllWindows()