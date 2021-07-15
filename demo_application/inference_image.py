import random

import numpy as np
import gradio as gr
import pandas as pd
import torch
import config
from facenet_pytorch.models.mtcnn import MTCNN
from demo_application.demo_models import DFDCSmallModels
from training.transformers import get_transformer

from utils import extract_box, timeit

np.random.seed(1)
torch.manual_seed(1)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

use_real = True  # set false to use mocked data
if not use_real:
    print("[WARNING] Using mocked data! Set `use_real` to True to use real outputs.")
else:
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)

    transform = get_transformer('test', size=240)

if use_real:
    mtcnn = MTCNN(**config.FACE_DETECTOR_KWARGS)
print('Face detector initialized!')

models_root = 'models'
model_metadatas = {
    'EfficientNet-B0': (
        'models/model=tf_efficientnet_b0_ns-run_id=c3x0rzgt-epoch=03-val_loss=0.3987.ckpt', 'tf_efficientnet_b0_ns'),
    'EfficientNet-B4': (
        'models/model=tf_efficientnet_b4_ns-run_id=2u1txf0a-epoch=02-val_loss=0.3715.ckpt', 'tf_efficientnet_b4_ns'),
    'Xception': ('models/model=xception-run_id=4jnbnk3u-epoch=01-val_loss=0.4701.ckpt', 'xception'),
    'Inception-V4': ('models/model=inception_v4-run_id=djinh5d4-epoch=04-val_loss=0.4442.ckpt', 'inception_v4'),
    'Inception-Resnet-V2': (
        'models/model=inception_resnet_v2-run_id=ay5l7q3q-epoch=01-val_loss=0.4274.ckpt', 'inception_resnet_v2'),
    'MobileNet-V3': (
        'models/model=mobilenetv3_large_100-run_id=6ihp9jw7-epoch=05-val_loss=0.4229.ckpt', 'mobilenetv3_large_100'),
    'ResNet50': ('models/model=resnet50-run_id=np7d6t9p-epoch=04-val_loss=0.4705.ckpt', 'resnet50'),
    'DenseNet161': ('models/model=densenet161-run_id=v2cq0a1n-epoch=01-val_loss=0.3785.ckpt', 'densenet161')
}

models = {}
print('Initializing models', end='...')
if use_real:
    for name, (path, run_name) in model_metadatas.items():
        _model = DFDCSmallModels(run_name)
        checkpoint = torch.load(path)
        _model.load_state_dict(checkpoint['state_dict'])
        _model.eval()
        models[name] = _model
print('DONE!')


def get_face(in_image):
    """
        Detects face from the input image and returns only first face
        - face-only image
        - resized, padded, normalized input image
        - face detection confidence
    """

    boxes, probabilities = mtcnn.detect(in_image)
    confidence_face = round(probabilities.astype('float')[0] * 100, 2)
    print(f'{len(boxes)} face(s) were detected with {confidence_face} probability.')
    face_img = extract_box(in_image, boxes[0])
    normalized_image = transform(image=np.array(face_img))
    print(f'Face extraction is done!')

    return face_img, normalized_image['image'].unsqueeze(0), confidence_face


def mocked_process_image(image, *weights):
    face_img = image
    confidence_face = random.randint(0, 101)
    results = []

    for _ in range(8):
        fake = random.randint(0, 101) / 100
        results.append({'fake': fake, 'real': 1 - fake})

    print(results)
    df = pd.DataFrame(results)
    df.insert(0, column='models', value=model_metadatas.keys())

    f, r = np.mean(df[['fake', 'real']].mul(weights, axis=0))
    f = f / (f+r)
    r = 1 - f
    weighted_overall_results = {'fake': f, 'real': r}

    return face_img, confidence_face, df, weighted_overall_results


@timeit
def process_image(image, *weights):
    print(f"Starting processing.")
    face_img, normalized_image, confidence_face = get_face(image)

    results = []
    for model_name, model in models.items():
        y_preds = model.forward(normalized_image)
        confidence = torch.softmax(y_preds, dim=1)
        real_confidence, fake_confidence = np.round(confidence.cpu().detach().numpy().squeeze().astype('float'), 2)
        results.append({'real': real_confidence, 'fake': fake_confidence})
    print(f'Inference completed!')

    df = pd.DataFrame(results)
    df.insert(0, column='models', value=model_metadatas.keys())

    f, r = np.mean(df[['fake', 'real']].mul(weights, axis=0))
    f = f / (f + r)
    r = 1 - f
    weighted_overall_results = {'fake': f, 'real': r}

    return face_img, confidence_face, df, weighted_overall_results


if __name__ == '__main__':
    model_names = model_metadatas.keys()
    fn = process_image if use_real else mocked_process_image

    iface = gr.Interface(fn,
                         inputs=[
                             gr.inputs.Image(type='pil', label='Input Image'),
                             *[gr.inputs.Slider(minimum=0, maximum=1, step=0.05, default=1, label=m) for m in
                               model_names]
                         ],
                         outputs=[
                             gr.outputs.Image(label='Detected Face'),
                             gr.outputs.Label(label='Face detection confidence'),
                             # 'dataframe',
                             gr.outputs.Dataframe(label='Results', type='pandas', headers=['models,fake,real']),
                             gr.outputs.Label(num_top_classes=2, label='Weighted Results')
                         ],
                         css="""custom.css""",
                         # examples='examples', - looks upgly, better upload
                         title='DeepFake Face Detection Demo')
    iface.launch(debug=True)
