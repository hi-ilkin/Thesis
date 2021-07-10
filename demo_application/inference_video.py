import os

import cv2
import gradio as gr


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    print(f'w: {w}, h: {h}, fps: {fps}')

    output_path = f'{os.path.dirname(video_path)}/output.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    counter = 0
    while True:
        counter += 1
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    out.release()
    return output_path


iface = gr.Interface(process_video, gr.inputs.Video(type='mp4'), "video")
iface.launch()
# sepia('C:/Users/husey/Downloads/FAKE.mp4')
