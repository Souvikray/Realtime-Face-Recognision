# FisherFaces algorithm is relatively inaccurate
import time
import os
import cv2
import numpy as np
from PIL import Image

# train, verify, test
def get_training_data(face_cascade, data_dir):
    # get the faces
    images = []
    # get the label associated with the faces
    labels = []
    # get list of all the files
    # we exclude one property or expression for the testing purpose in the later stage
    image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if not f.endswith('.wink')]
    for image_file in image_files:
        # OpenCV cannot read .gif files so we use an alternate library PIL
        # read the image and convert it to grayscale
        img = Image.open(image_file).convert('L')
        # represent the image in matrix format for the OpenCV to work on it
        img = np.array(img)
        filename = os.path.split(image_file)[1]
        true_person_number = int(filename.split(".")[0].replace("subject", ""))

        # detect faces in the image
        faces = face_cascade.detectMultiScale(img, 1.05, 6)
        for face in faces:
            x, y, w, h = face
            face_region = img[y:y+h, x:x+w]
            # FisherFaces expects the training image sets to be of equal size
            face_region = cv2.resize(face_region, (150, 150))
            images.append(face_region)
            labels.append(true_person_number)

    return images, labels

def evaluate(face_recognizer, face_cascade, data_dir):
    # get the images which the face_recognizer hasn't seen before
    image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wink')]
    num_correct = 0
    for image_file in image_files:
        # OpenCV cannot read .gif files so we use an alternate library PIL
        # read the image and convert it to grayscale
        img = Image.open(image_file).convert('L')
        # represent the image in matrix format for the OpenCV to work on it
        img = np.array(img)
        filename = os.path.split(image_file)[1]
        true_person_number = int(filename.split(".")[0].replace("subject", ""))

        # detect faces in the image
        faces = face_cascade.detectMultiScale(img, 1.05, 6)
        for face in faces:
            x, y, w, h = face
            face_region = img[y:y + h, x:x + w]
            # FisherFaces expects the image to be of equal size
            face_region = cv2.resize(face_region, (150, 150))
            person_number, confidence = face_recognizer.predict(face_region)
            if person_number == true_person_number:
                num_correct += 1
                print("Correctly identified person {} with confidence {}".format(true_person_number, confidence))
            else:
                print("Incorrectly identified real person {} to false person {}".format(true_person_number, person_number))
    accuracy = (num_correct / len(image_files)) * 100
    print(accuracy)


# get the features from the file and pass it to the Cascade Classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# We choose Local Binary Pattern as the face recognision algorithm
face_recognizer = cv2.face.FisherFaceRecognizer_create()
images, labels = get_training_data(face_cascade, 'yalefaces')
# pass the training data to the face recognision algorithm
face_recognizer.train(images, np.array(labels))
evaluate(face_recognizer, face_cascade, 'yalefaces')




