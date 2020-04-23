import cv2
import os
import sys
# import skvideo.io
# import tqdm

original_video_path = '/media/keyi/Data/Research/traffic/data/Hwy7/20200224_153147_9FD8.mkv'
rescaled_output = '/media/keyi/Data/Research/traffic/data/Hwy7/20200224_153147_demo4.avi'

cap_video = cv2.VideoCapture(original_video_path)
width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap_video.get(cv2.CAP_PROP_FPS))

downscale_factor = 4

video_out = cv2.VideoWriter(rescaled_output,
                            cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps,
                            (width // downscale_factor, height // downscale_factor))

frame_id = 0
num_frames = cap_video.get(cv2.CAP_PROP_FRAME_COUNT)
print(num_frames)

# video_data = skvideo.io.vread(original_video_path)

while frame_id < num_frames:
# for frame_id in tqdm(range(num_frames)):
    frame_id += 1
    ret, frame = cap_video.read()

    if not ret:
        # print('Image rendering done. Save to:', rescaled_output)
        print('Skipping frame: ', frame_id)
        # cv2.imshow('skipped frames:', frame)
        # cv2.waitKey()
        continue
    # frame = video_data[frame_id]

    rescaled_frame = cv2.resize(frame, dsize=(width // downscale_factor, height // downscale_factor))

    video_out.write(rescaled_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_video.release()
print(frame_id, ' output image dim:', width // downscale_factor, height // downscale_factor)
