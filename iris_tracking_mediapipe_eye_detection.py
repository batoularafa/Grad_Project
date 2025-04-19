import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import time
# from picamera2 import Picamera2
# import serial

# right eyes indices
RIGHT_IRIS = [474, 475, 476, 477]
R_H_LEFT = [362]
R_H_RIGHT = [263]
# Eye blink detection landmarks
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145

# left eyes indices
LEFT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]  # left eye left most landmark
L_H_RIGHT = [133]
# Eye blink detection landmarks
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374

#start the picamera
# picam2 = Picamera2()
# picam2.preview_configuration.main.size = (800,800)
# picam2.preview_configuration.main.format = "RGB888"
# picam2.preview_configuration.align()
# picam2.configure("preview")
# picam2.start()

# ser = serial.Serial('COM6', 9600, timeout=1)

#ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

def euclidean_distance(p1, p2):
    x1, y1 = p1.ravel()
    x2, y2 = p2.ravel()
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def eyes_pos(center, right, left):
    center_to_right_dist = euclidean_distance(center, right)
    total_distance = euclidean_distance(right, left)
    avg_ratio = center_to_right_dist / total_distance
    iris_pos = ""
    if avg_ratio <= 0.42:
        iris_pos = "right"
    elif avg_ratio > 0.42 and avg_ratio < 0.57:
        iris_pos = "center"
    elif avg_ratio >= 0.57:
        iris_pos = "left"

    return iris_pos, avg_ratio

def both_eye_blink(face_landmarks, frame, blink_threshold=0.02):
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
        print("Blink Detected")
        return True
    return False

def one_eye_blink(face_landmarks, frame, blink_threshold=0.02):
    # Get the landmarks for both eyes
    left_eye_top = face_landmarks.landmark[LEFT_EYE_TOP]
    left_eye_bottom = face_landmarks.landmark[LEFT_EYE_BOTTOM]
    right_eye_top = face_landmarks.landmark[RIGHT_EYE_TOP]
    right_eye_bottom = face_landmarks.landmark[RIGHT_EYE_BOTTOM]
    left_eye_dist = np.linalg.norm(
        np.array([left_eye_top.x, left_eye_top.y]) - np.array([left_eye_bottom.x, left_eye_bottom.y])
    )
    right_eye_dist = np.linalg.norm(
        np.array([right_eye_top.x, right_eye_top.y]) - np.array([right_eye_bottom.x, right_eye_bottom.y])
    )

    # Check if one eye meet the blink threshold
    if left_eye_dist < blink_threshold or right_eye_dist < blink_threshold:
        cv.putText(frame, "One eyed Blink Detected", (50, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print("One eyed Blink Detected")
        return True
    return False


mp_face_mesh = mp.solutions.face_mesh

cap = cv.VideoCapture(0)

wheel_state = False
flag = False
blink_detected = False  # Flag for blink detection
iris_positions = []  # List to store iris positions with timestamps
iris_ratios = []
speed = 1  # Speed level (1 or 2)

# For single-eye blink timing
one_eye_closed_time = 0
one_eye_blink_flag = False

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:
    while True:
        # frame = picam2.capture_array()
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        current_time = time.time()

        if results.multi_face_landmarks:
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            # Check for blink detection (both eyes)
            blink_detected = both_eye_blink(results.multi_face_landmarks[0], frame)

            if blink_detected:
                if len(iris_positions) > 0:
                    blink_time = time.time()
                    iris_positions_blink = [pos for pos in iris_positions if blink_time - pos[1] <= 1]
                    iris_ratios_blink = [pos for pos in iris_ratios if blink_time - pos[1] <= 1]

                    while blink_time - open_eyes_time > 1 and flag == False:
                        flag = True
                        wheel_state = not wheel_state
                        break
                    if flag == True:
                        cv.putText(frame, "toggle done", (30, 30), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1,  cv.LINE_AA)
                        print("toggle done")

                    if iris_positions_blink and wheel_state == True:
                        last_iris_position, _ = iris_positions_blink[-1]
                        last_iris_ratio, _ = iris_ratios_blink[-1]
                        cv.putText(frame, f'Speed: {speed}', (30, 120), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 255), 1)
                        cv.putText(frame, f'iris pos: {last_iris_position}, {last_iris_ratio:.2f}', (30, 30),cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)
                        print(f'Speed: {speed}',f'iris pos: {last_iris_position}, {last_iris_ratio:.2f}')
                        # ser.write(f"{last_iris_ratio}\n".encode())
                        #if speed == 1:
                            #ser.write(f"s1@{last_iris_ratio}\n".encode())
                        #else:
                            #ser.write(f"s2@{last_iris_ratio}\n".encode())
                        # time.sleep(0.1)
                        # if ser.in_waiting > 0:  # Check if there is data available to read
                        #     data = ser.readline().decode('utf-8').strip()  # Decode and strip newline
                        #     print(f"Received: {data}")
                        # # time.sleep(0.1)

            else:
                open_eyes_time = time.time()
                flag = False
                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
                iris_position_right, ratio_right = eyes_pos(center_right, mesh_points[R_H_RIGHT],mesh_points[R_H_LEFT][0])
                iris_position_left, ratio_left = eyes_pos(center_left, mesh_points[L_H_RIGHT],mesh_points[L_H_LEFT][0])
                avg_ratio = (ratio_right + ratio_left) / 2

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
                    #if speed == 1:
                        #ser.write(f"s1@{avg_ratio}\n".encode())
                    #else:
                        #ser.write(f"s2@{avg_ratio}\n".encode())
                    # time.sleep(0.1)
                    # if ser.in_waiting > 0:  # Check if there is data available to read
                    #     data = ser.readline().decode('utf-8').strip()  # Decode and strip newline
                    #     print(f"Received: {data}")  
                    # time.sleep(0.1)
                   
                    cv.putText(frame, "moving", (30, 60), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)
                    cv.putText(frame, f'Speed: {speed}', (30, 120), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 255), 1)
                    cv.putText(frame, f'iris pos: {iris_pos}, {avg_ratio:.2f}', (30, 30), cv.FONT_HERSHEY_PLAIN, 1.2,(0, 255, 0), 1, cv.LINE_AA)
                    print("moving",f'Speed: {speed}', f'iris pos: {iris_pos}, {avg_ratio:.2f}')

                    # Check for single-eye blink detection
                    if one_eye_blink(results.multi_face_landmarks[0], frame):
                        if one_eye_closed_time == 0:
                            one_eye_closed_time = current_time
                        elif current_time - one_eye_closed_time > 1 and not one_eye_blink_flag:
                            # Toggle speed if one eye blinked for 1 second and wheelchair is moving 
                            speed = 2 if speed == 1 else 1
                            one_eye_blink_flag = True
                            cv.putText(frame, "Speed Toggle", (30, 90), cv.FONT_HERSHEY_PLAIN, 1.2,(0, 255, 255), 1)
                            print("Speed Toggle")
                    else:
                        one_eye_closed_time = 0
                        one_eye_blink_flag = False
                else:
                    one_eye_closed_time = 0

                    # ser.write(f"5.0\n".encode())
                    # time.sleep(0.1)
                    # if ser.in_waiting > 0:  # Check if there is data available to read
                    #     data = ser.readline().decode('utf-8').strip()  # Decode and strip newline
                        # print(f"Received: {data}")  
                    # time.sleep(0.1)
                    cv.putText(frame, "stopping", (30, 60), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)
                    cv.putText(frame, f'iris pos: {iris_pos}, {avg_ratio:.2f}', (30, 30), cv.FONT_HERSHEY_PLAIN, 1.2,(0, 255, 0), 1, cv.LINE_AA)   
                    print("stopping", f'iris pos: {iris_pos}, {avg_ratio:.2f}')
                if not one_eye_blink(results.multi_face_landmarks[0], frame):
                    one_eye_blink_flag = False
           
        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key == 27:  # esc
            break

    cap.release()
    cv.destroyAllWindows()
