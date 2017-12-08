from IdentifyFace import *
import cv2
import numpy as np

face_person = {1: "Harry", 2: "John", 3: "Chang", 4: "Zhao", 5: "Alex", 6: "Chow",
               7: "Raju", 8: "Hatori", 9: "David", 10: "Subhash", 11: "Amy", 12: "Harry",
               13: "Subramaniyam", 14: "Xing", 15: "Mike", 16: "Souvik"}
# capture video from a live video stream
video_cap = cv2.VideoCapture(0)
while True:
    # get the frame
    ret, frame = video_cap.read()
    # resize the frame to half to speed up the processing
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.05, 6)
    for face in faces:
        x, y, w, h = face
        # our face recognision algorithm has been trained on grayscale images
        face_region = gray[y:y+h, x:x+w]
        person_number, confidence = face_recognizer.predict(face_region)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if face_person[person_number] is "Souvik":
            cv2.putText(frame, face_person[person_number], (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0)) #(image, text, text-coordinate, font, font-size, text-color)
        '''    
        else:
            cv2.putText(frame, str(face_person[person_number]), (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
        '''
    # show the present frame
    cv2.imshow("Running Face Recognision", frame)
    # cv2.waitKey(1) returns a value of -1 which is masked using & 0xFF to get char value
    key = cv2.waitKey(1) & 0xFF
    if key is ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()


