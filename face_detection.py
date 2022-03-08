import cv2
from random import randrange

#Machine learning opencv program for detecting faces
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to detect faces in
webcam_video = cv2.VideoCapture("faces.mp4")

while True:
    successful_frame_read, frame = webcam_video.read()
    #Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Draw Rectangles around the faces
    for (x, y, w, h)  in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)


    #displaying image
    cv2.imshow('FACE DETECTOR' ,frame)

    #processing time
    key = cv2.waitKey(1)
    print("Code Completed")

    #if 0 is pressed Break the Code
    if key==81 or key==113:
        break


