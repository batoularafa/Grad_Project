import cv2 as cv
import numpy as np
import mediapipe as mp 
import math
import time

# Left eyes indices 
RIGHT_IRIS = [474,475,476,477]
R_H_LEFT = [362]
R_H_RIGHT = [263]
# right eyes indices
LEFT_IRIS = [469,470,471,472]
L_H_LEFT = [33] #left eye left most landmark
L_H_RIGHT =[133]

def euclidean_distance(p1,p2):
    x1, y1 = p1.ravel()
    x2, y2 = p2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance
def eyes_pos(center, right, left):
    center_to_right_dist = euclidean_distance(center, right)
    total_distance = euclidean_distance(right, left)
    ratio = center_to_right_dist/total_distance
    iris_pos= ""
    if ratio <= 0.42:
        iris_pos = "right"
    elif ratio > 0.42 and ratio < 0.57:
        iris_pos = "center"
    elif ratio >= 0.57:
        iris_pos = "left"

    return iris_pos, ratio

mp_face_mesh = mp.solutions.face_mesh

cap = cv.VideoCapture(0)
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
            # print(results.multi_face_landmarks[0].landmark )
            # [print(p.x,p.y) for p in results.multi_face_landmarks[0].landmark] 
            mesh_points = np.array([np.multiply([p.x,p.y], [img_w,img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark] )
            # print(mesh_points.shape)
            # cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
            # cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype = np.int32)
            center_right = np.array([r_cx, r_cy], dtype = np.int32)
            cv.circle(frame, center_left, int(l_radius), (255,0,255), 1 , cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255,0,255), 1 , cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255,255,255), -1 , cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_LEFT][0], 3, (0,255,255), -1 , cv.LINE_AA)
            iris_position1, ratio1 = eyes_pos(center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0])
            iris_position2, ratio2 = eyes_pos(center_left, mesh_points[L_H_RIGHT], mesh_points[L_H_LEFT][0])
            ratio = (ratio1+ratio2)/2
            if ratio <= 0.45:
                iris_pos = "right"
            elif ratio > 0.45 and ratio < 0.55:
                iris_pos = "center"
            elif ratio >= 0.55:
                iris_pos = "left"
            print(iris_pos, ratio)
            cv.putText(frame,f'iris pos: {iris_pos}, {ratio:.2f}', (30,30), cv.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 1, cv.LINE_AA)
            # cv.putText(frame,f'iris pos: {iris_position2}, {ratio1:.2f}', (30,50), cv.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 1, cv.LINE_AA)
            # print(iris_position2, ratio2)
        cv.imshow('img', frame)
        key = cv.waitKey(1)

        if key == 27: #esc
            break

    cap.release()
    cv.destroyAllWindows()
