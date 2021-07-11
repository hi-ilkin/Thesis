import numpy as np
import gradio as gr
import torch
from facenet_pytorch.models.mtcnn import MTCNN

import config
from demo_application.demo_models import DFDCSmallModels
from training.transformers import get_transformer
from utils import extract_box

mtcnn = MTCNN(**config.FACE_DETECTOR_KWARGS)
transform = get_transformer('test', size=224)

model = DFDCSmallModels.load_from_checkpoint(
    'model=tf_efficientnet_b0_ns-run_id=2yvkagkm-epoch=00-val_loss=1.2081.ckpt')


def process_image(image, model_name):
    print(f"Starting processing. {model_name} model was chosen.")

    boxes, probabilities = mtcnn.detect(image)
    confidence_face = probabilities.astype('float')[0]
    print(f'{len(boxes)} face(s) were detected with {confidence_face} probability.')

    face_img = extract_box(image, boxes[0])
    input_image = transform(image=np.array(face_img))
    print(f'Face extraction is done!')

    y_preds = model.forward(input_image['image'].unsqueeze(0))
    confidence = torch.softmax(y_preds, dim=1)
    fake_confidence, real_confidence = np.round(confidence.cpu().detach().numpy().squeeze().astype('float'), 2)
    print('Inference completed!')

    return face_img, '0.44', {'real': real_confidence, 'fake': fake_confidence}, f'Chosen model: {model_name}'


if __name__ == '__main__':
    iface = gr.Interface(process_image,
                         inputs=[
                             gr.inputs.Image(type='pil', label='Input Image'),
                             gr.inputs.Dropdown(
                                 choices=['Xception', 'EfficientNet-B0', 'EfficientNet-B4'], type='value')
                         ],
                         outputs=[
                             gr.outputs.Image(label='Detected Face'),
                             gr.outputs.Label(label='Face detection confidence'),
                             gr.outputs.Label(num_top_classes=2, label='Fake/Real confidences'),
                             'text'
                         ],
                         examples=[
                             ['examples/fake.jpg', 'Xception'],
                             ['examples/fake.jpg', 'EfficientNet-B0'],
                             ['examples/fake.jpg', 'EfficientNet-B4'],
                             ['examples/aaeflzzhvy.mp4_9_.png', 'Xception'],
                             ['examples/aaghqpewzx.mp4_8.png', 'DenseNet161'],
                             ['examples/abfnyenqdw.mp4_1.png', 'Inception-ResNet-v2']
                         ],
                         title='DeepFake Face Detection Demo')
    iface.launch(debug=True)
