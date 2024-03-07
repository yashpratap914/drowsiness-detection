from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

mixer.init()
mixer.music.load("music.wav")


def eye_aspect_ratio(eye):  #using formula to calculate ear
    A = distance.euclidean(eye[1], eye[5]) #Vertical dist
    B = distance.euclidean(eye[2], eye[4]) #Verical dist
    C = distance.euclidean(eye[0], eye[3]) #horizontal dist
    ear = (A + B) / (2.0 * C)
    return ear  #val of ear remains const when eye->open , drops ->eye closes


thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()  #built in func used for bettter effectiveness
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #68 landmarks from face

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]  #left eye landmark
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0) #built in func -> return frames detected from camera(0->primary camera)
flag = 0 #frame count
while True: # infinite while loop
    ret, frame = cap.read() #built in func return 2 vals : bool, imagery vector
    frame = imutils.resize(frame, width=450) #resized frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converted to grayscale
    subjects = detect(gray, 0) #detector
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]   # passed landmarks
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye) #individual ear
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0  # average ear
        leftEyeHull = cv2.convexHull(leftEye) #completely enclose / wrap object
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)  #cover points on boundary (IMAGE, HULL, CONTOUR INDEX(EXACT POINT TO BE DRAWN), COLOR OF LINE, THICKNESS OF LINE)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if ear < thresh:  #CONDITION
            flag += 1
            print(flag)
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            flag = 0
    cv2.imshow("Frame", frame) #display image on our window: window_name, image(frame)
    key = cv2.waitKey(1) & 0xFF  # display window for a millisecond
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.release() 