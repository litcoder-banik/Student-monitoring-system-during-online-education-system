import cv2
import dlib
import numpy as np
from math import hypot
import time

font = cv2.FONT_HERSHEY_PLAIN

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


def get_blining_ration(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(
        eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(
        eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(
        eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(
        eye_points[5]), facial_landmarks.part(eye_points[4]))

    # cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    # cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_length = hypot(
        (left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot(
        (center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_length/ver_line_length
    return ratio


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:

        landmarks = predictor(gray, face)

        # Detect Blinking
        left_eye_ratio = get_blining_ration(
            [36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blining_ration(
            [42, 43, 44, 45, 46, 47], landmarks)
        blinking_ration = (left_eye_ratio + right_eye_ratio)/2

        if blinking_ration > 4.5:
            cv2.putText(frame, "Blinking", (50, 150), font, 3, (255, 0, 0))

        # Gaze Detection
        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
        # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)
        # print(left_eye_region)

        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        left_eye = cv2.bitwise_and(gray, gray, mask=mask)

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])

        gray_eye = left_eye[min_y: max_y, min_x: max_x]
        #gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        #gray_eye = cv2.equalizeHist(gray_eye)
        # _, threshold_eye = cv2.threshold(gray_eye_equalized, 50, 255, cv2.THRESH_BINARY)
        # threshold_eye = cv2.adaptiveThreshold(gray_eye, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        threshold_eye = cv2.adaptiveThreshold(gray_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        height, width = threshold_eye.shape
        left_side_threshold = threshold_eye[0 : height, 0 : int(width/2)]
        left_side_white = cv2.countNonZero(left_side_threshold)

        right_side_threshold = threshold_eye[0 : height, int(width/2) : width]
        right_side_white = cv2.countNonZero(right_side_threshold)

        gaze_ratio = left_side_white/right_side_white

        threshold_eye = cv2.resize(threshold_eye, None, fx=10, fy=10)
        eye = cv2.resize(gray_eye, None, fx=10, fy=10)
        left_side_threshold = cv2.resize(left_side_threshold, None, fx=15, fy=10)
        right_side_threshold = cv2.resize(right_side_threshold, None, fx=15, fy=10)
        
        
        # cv2.putText(frame, str(left_side_white), (50, 100), font, 2, (0, 0, 255), 3)
        # cv2.putText(frame, str(right_side_white), (50, 150), font, 2, (0, 0, 255), 3)
        cv2.putText(frame, str(gaze_ratio), (50, 150), font, 2, (0, 0, 255), 3)
        cv2.imshow("Threshold", threshold_eye)
        cv2.imshow("Left", left_side_threshold)
        cv2.imshow("Right", right_side_threshold)

        start_time = time.time()
        end_time = time.time()
        frame_duration = end_time - start_time
        frame_rate = int(1 / frame_duration) if frame_duration != 0 else 0

        left_duration = 0
        center_duration = 0
        right_duration = 0
        #threshold_time = 1 * frame_rate

        if gaze_ratio <= 0.9:
            left_duration += 1
            center_duration = 0
            right_duration = 0
        elif gaze_ratio > 0.9 and gaze_ratio <= 1.1:
            center_duration += 1
            left_duration = 0
            right_duration = 0
        else:
            right_duration += 1
            left_duration = 0
            center_duration = 0

        # # Check for attentiveness
        # if left_duration >= 5:
        #     attentiveness = "Less Attentive"
        # elif center_duration >= 2:
        #     attentiveness = "Attentive"
        # elif right_duration >= 5:
        #     attentiveness = "Less Attentive"
        # else:
        #     attentiveness = "Neutral"

        #cv2.putText(frame, attentiveness, (50, 200), font, 2, (0, 0, 255), 3)

        if gaze_ratio <= 1.1:
            cv2.putText(frame, str("Left"), (50, 100), font, 2, (0, 0, 255), 3)
        elif (gaze_ratio > 1.1) and (gaze_ratio < 1.2):
             cv2.putText(frame, str("Center"), (50, 100), font, 2, (0, 0, 255), 3)
        elif (gaze_ratio >= 1.2):
            cv2.putText(frame, str("Right"), (50, 100), font, 2, (0, 0, 255), 3)
        

    cv2.imshow("Eye_Detecting", frame)

    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
