#! /usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np

# video_1_path = "/media/keyi/Data/Research/traffic/detection/object_detection_2D/experiment/SSD300_traffic_001/live_results/GTA/GTA_4.mkv"
# video_2_path = "/media/keyi/Data/Research/traffic/detection/object_detection_2D/experiment/SSD512_traffic_002/live_results/GTA/GTA_4.mkv"
# video_out_path = "/media/keyi/Data/Research/traffic/detection/object_detection_2D/experiment/SSD512_traffic_002/live_results/GTA_4_cat.mkv"

video_1_path = "/media/keyi/Data/Research/traffic/detection/object_detection_2D/experiment/SSD300_traffic_001/live_results/Hwy7/20200224_153147_demo4.mkv"
video_2_path = "/media/keyi/Data/Research/traffic/detection/object_detection_2D/experiment/SSD512_traffic_002/live_results/Hwy7/20200224_153147_demo4.mkv"
video_out_path = "/media/keyi/Data/Research/traffic/detection/object_detection_2D/experiment/SSD512_traffic_002/live_results/20200224_153147_demo4_cat.mkv"

cap_left = cv2.VideoCapture(video_1_path)
cap_right = cv2.VideoCapture(video_2_path)

width = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap_left.get(cv2.CAP_PROP_FPS))

video_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'),
                            20, (width * 2, height))

while True:
    ret1, frame1 = cap_left.read()
    ret2, frame2 = cap_right.read()

    if ret1 and ret2:
        img_concat = np.concatenate((frame1, frame2), axis=1)

        text1 = 'left SSD300 Citycam  vs.  right SSD512 DETRAC'

        # For pm demo
        cv2.putText(img_concat, text1, (60, 60),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 200), 1, cv2.LINE_AA)

        video_out.write(img_concat)
    else:
        break

    cv2.imshow('frame cat', img_concat)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
video_out.release()
