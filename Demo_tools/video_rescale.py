import cv2
import os
import sys

original_video_path = '/media/keyi/Data/Research/traffic/data/Hwy7/20200224_153147_demo4.mkv'
rescaled_output = '/media/keyi/Data/Research/traffic/data/Hwy7/20200224_153147_demo8.mkv'

cap_video = cv2.VideoCapture(original_video_path)
width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap_video.get(cv2.CAP_PROP_FPS))

downscale_factor = 2

video_out = cv2.VideoWriter(rescaled_output,
                            cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps,
                            (width // downscale_factor, height // downscale_factor))

while True:
    ret, frame = cap_video.read()

    if not ret:
        print('Image rendering done. Save to:', rescaled_output)
        break

    rescaled_frame = cv2.resize(frame, dsize=(width // downscale_factor, height // downscale_factor))

    video_out.write(rescaled_frame)

cap_video.release()
print('output image dim:', width // downscale_factor, height // downscale_factor)
