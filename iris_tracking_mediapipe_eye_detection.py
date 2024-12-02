import cv2 as cv
import numpy as np
import mediapipe as mp 
import math
import time
import serial

# Left eyes indices 
RIGHT_IRIS = [474,475,476,477]
R_H_LEFT = [362]
R_H_RIGHT = [263]

# right eyes indices
LEFT_IRIS = [469,470,471,472]
L_H_LEFT = [33] #left eye left most landmark
L_H_RIGHT =[133]

# Eye blink detection landmarks
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145

ser = serial.Serial('COM8', 9600)
def euclidean_distance(p1,p2):
    x1, y1 = p1.ravel()
    x2, y2 = p2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance
def eyes_pos(center, right, left):
    center_to_right_dist = euclidean_distance(center, right)
    total_distance = euclidean_distance(right, left)
    avg_ratio = center_to_right_dist/total_distance
    iris_pos= ""
    if avg_ratio <= 0.42:
        iris_pos = "right"
    elif avg_ratio > 0.42 and avg_ratio < 0.57:
        iris_pos = "center"
    elif avg_ratio >= 0.57:
        iris_pos = "left"

    return iris_pos, avg_ratio

def detect_blink(face_landmarks, frame, blink_threshold=0.02):
    # Get the landmarks for both eyes
    left_eye_top = face_landmarks.landmark[LEFT_EYE_TOP]
    left_eye_bottom = face_landmarks.landmark[LEFT_EYE_BOTTOM]
    right_eye_top = face_landmarks.landmark[RIGHT_EYE_TOP]
    right_eye_bottom = face_landmarks.landmark[RIGHT_EYE_BOTTOM]

    # Calculate the vertical distances for each eye
    left_eye_dist = np.linalg.norm(
        np.array([left_eye_top.x, left_eye_top.y]) - np.array([left_eye_bottom.x, left_eye_bottom.y])
    )
    right_eye_dist = np.linalg.norm(
        np.array([right_eye_top.x, right_eye_top.y]) - np.array([right_eye_bottom.x, right_eye_bottom.y])
    )

    # Check if both eyes meet the blink threshold
    if left_eye_dist < blink_threshold and right_eye_dist < blink_threshold:
        cv.putText(frame, "Blink Detected", (50, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return True
    return False

mp_face_mesh = mp.solutions.face_mesh

cap = cv.VideoCapture(0)

wheel_state = False
flag = False
blink_detected = False  # Flag for blink detection
iris_positions = []  # List to store iris positions with timestamps
iris_ratios = []

start_time = time.time()  # Initialize the start time

with mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence = 0.5,
    min_tracking_confidence =0.5
    )as face_mesh:
    while True:
        ret, frame = cap.read()
        frame = cv.flip(frame,1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            
            mesh_points = np.array([np.multiply([p.x,p.y], [img_w,img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark] )
           
            # Check for blink detection
            blink_detected = detect_blink(results.multi_face_landmarks[0], frame)

            if blink_detected:
                
                # If a blink is detected, display the iris position from 1 second ago
                if len(iris_positions) > 0:
                    # Filter out positions older than 1 second
                    blink_time = time.time()
                    iris_positions_blink= [pos for pos in iris_positions if blink_time - pos[1] <= 1]
                    iris_ratios_blink= [pos for pos in iris_ratios if blink_time - pos[1] <= 1]
                    
                    #toggle wheel_state when eyes closed for more than 1 seconds
                    print(blink_time - open_eyes_time)
                    while blink_time - open_eyes_time > 1 and flag == False:
                        flag = True
                        wheel_state = not wheel_state
                        break
                    if flag == True:
                        print(wheel_state)
                        cv.putText(frame, "toggle done", (30, 30),cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)

                    if iris_positions_blink:
                        last_iris_position, _ = iris_positions_blink[-1]  # Get the last position from within 1 second
                        last_iris_ratio, _ = iris_ratios_blink[-1] 
                        cv.putText(frame, f'iris pos: {last_iris_position}, {last_iris_ratio:.2f}', (30, 30),cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)
                        ser.write(last_iris_ratio)
                        
            else:      
                open_eyes_time = time.time()
                flag = False
                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                center_left = np.array([l_cx, l_cy], dtype = np.int32)
                center_right = np.array([r_cx, r_cy], dtype = np.int32)
                cv.circle(frame, center_left, int(l_radius), (255,0,255), 1 , cv.LINE_AA)
                cv.circle(frame, center_right, int(r_radius), (255,0,255), 1 , cv.LINE_AA)
                # cv.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255,255,255), -1 , cv.LINE_AA)
                # cv.circle(frame, mesh_points[R_H_LEFT][0], 3, (0,255,255), -1 , cv.LINE_AA)
                iris_position_right, ratio_right = eyes_pos(center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0])
                iris_position_left, ratio_left = eyes_pos(center_left, mesh_points[L_H_RIGHT], mesh_points[L_H_LEFT][0])
                avg_ratio = (ratio_right+ratio_left)/2
                # ser.write(avg_ratio)
                if avg_ratio <= 0.45:
                    iris_pos = "right"
                elif avg_ratio > 0.45 and avg_ratio < 0.55:
                    iris_pos = "center"
                elif avg_ratio >= 0.55:
                    iris_pos = "left"


                # Store the current iris position and average ratio with the current timestamp
                iris_positions.append((iris_pos, time.time()))
                iris_ratios.append((avg_ratio, time.time()))
                if wheel_state == True: 
                    # ser.write("M")
                    ser.write(avg_ratio)
                    print(iris_pos, avg_ratio)
                    cv.putText(frame, "moving", (30, 60),cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)
                    cv.putText(frame,f'iris pos: {iris_pos}, {avg_ratio:.2f}', (30,30), cv.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 1, cv.LINE_AA)
                else:
                    ser.write("S")
                    cv.putText(frame, "stopping", (30, 60),cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)
                    cv.putText(frame,f'iris pos: {iris_pos}, {avg_ratio:.2f}', (30,30), cv.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 1, cv.LINE_AA)
                # cv.putText(frame,f'iris pos: {iris_position_left}, {ratio_right:.2f}', (30,50), cv.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 1, cv.LINE_AA)
                # print(iris_position_left, ratio_left)
        cv.imshow('img', frame)
        key = cv.waitKey(1)

        if key == 27: #esc
            break

    cap.release()
    cv.destroyAllWindows()
