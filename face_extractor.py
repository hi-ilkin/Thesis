import mmcv
import cv2
from PIL import Image, ImageDraw

from utils import get_frames, save_video


class FaceExtractor:
    def __init__(self, use_mtcnn=True):
        if use_mtcnn:
            import torch
            from facenet_pytorch import MTCNN

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print('Running on device: {}'.format(device))
            self.mtcnn = MTCNN(margin=50, keep_all=True, device=device, post_process=False)

        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.FPS = 30
        self.batch_size = 16

    def extract_faces(self, video_path):
        print(f"\nExtracting faces from: {video_path}")

        frames = get_frames(video_path)

        frames_tracked = []
        face_coordinates = {}
        for i, frame in enumerate(frames):
            # Detect faces
            boxes, _ = self.mtcnn.detect(frame)

            # Draw faces
            # processed_frame = self.draw_on_frame(frame, boxes)

            if boxes is not None:
                face_coordinates[i] = boxes.tolist()
                processed_frame = self.crop_faces(frame, boxes)

                # Add to frame list
                frames_tracked.append(processed_frame)
            else:
                face_coordinates[i] = None

        save_video(frames_tracked, video_path, self.fourcc, self.FPS, output_size=(240, 240), output_path='data/faces_48')

        return {'face_coordinates': face_coordinates}

    def extract_faces_batch(self, video_path):
        video = mmcv.VideoReader(video_path)
        frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
        processed_frames = []

        batch_start = 0
        for idx in range(16, len(frames) + self.batch_size - 5, self.batch_size):
            batch = frames[batch_start:idx]
            detection_results = self.mtcnn(batch)

            for p in detection_results:
                processed_frames.append(p[0].permute(1, 2, 0).int().numpy())
            batch_start = idx

        save_video(self.fourcc, self.FPS, processed_frames, video_path)

    def draw_on_frame(self, frame, boxes):
        frame_draw = frame.copy()
        draw = ImageDraw.Draw(frame_draw)
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

        return frame_draw

    def crop_faces(self, frame, boxes):
        """
        Crops faces and returns .
        !!! Currently returning only first detected face
        :param frame: single frame image
        :param boxes: coordinates of detected face
        :return: cropped face image
        """
        for box in boxes:
            return frame.crop(box.tolist())
