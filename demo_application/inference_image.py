import cv2
import numpy as np
import gradio as gr
from facenet_pytorch.models.mtcnn import MTCNN

import config
from demo_application.demo_models import DFDCSmallModels
from training.transformers import get_transformer
from utils import extract_box

mtcnn = MTCNN(**config.FACE_DETECTOR_KWARGS)
transform = get_transformer('test', size=224)

model = DFDCSmallModels.load_from_checkpoint(
    'model=tf_efficientnet_b0_ns-run_id=2yvkagkm-epoch=00-val_loss=1.2081.ckpt')


def process_image(image):
    boxes, probabilities = mtcnn.detect(image)
    face_img = extract_box(image, boxes[0])
    input_image = transform(image=np.array(face_img))
    result = model.forward(input_image['image'].unsqueeze(0))

    # return face detection probability too
    return face_img, {'real': 0.1, 'fake': 0.75}, str(result)


if __name__ == '__main__':
    iface = gr.Interface(process_image,
                         inputs=gr.inputs.Image(type='pil'),
                         outputs=["image", gr.outputs.Label(num_top_classes=2), "textbox"],
                         examples=[['examples/fake.jpg']])
    iface.launch(debug=True)
