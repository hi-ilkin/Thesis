import os.path

import cv2
import gradio as gr
import numpy as np


def sepia(video_path):
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20 # round(cap.get(cv2.CAP_PROP_FPS))
    print(f'w: {w}, h: {h}, fps: {fps}')
    print(video_path)
    output_path = f'{os.path.dirname(video_path)}/output.avi'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mjpg'), fps, (w, h))
    counter = 0
    while True:
        counter += 1
        ret, frame = cap.read()
        if not ret:
            break
        # cv2.imshow('window', frame)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break

        out.write(frame*2)
    print(counter)
    out.release()
    print(output_path)
    return video_path


iface = gr.Interface(sepia, gr.inputs.Video(type='avi'), "video")
iface.launch()
# sepia('C:/Users/husey/Downloads/FAKE.mp4')
