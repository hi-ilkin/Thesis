# Environment
- Python 3.7+
- CUDA 11.0 Update 1 or CUDA 10.1
- Install proper [pytorch](https://pytorch.org/) version from  recommended way.

for  better experience with notebooks install jupyterlab and [nbextensions](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html). 

### Hardware specs
Computer 1:
- Intel i7-9750H 2.6 GHz
- 16 GB RAM
- Nvidia GTX 1650 Max-Q

Computer 2:
- Intel i7-9750H 2.6 GHz
- 16 GB RAM
- Nvidia RTX 2070 Max-Q

Computer 3: TODO: Fill
- Intel i7-8700K 3.7 GHz
- 64 GB RAM
- Nvidia Titan V

### Third party tools
- [Facenet-pytorch](https://github.com/timesler/facenet-pytorch)


### Known Problems
n (such as opencv-python-headless, opencv-python, opencv-contrib-python or opencv-contrib-python-headless) installed 
in your Python environment, you can force Albumentations installation with following command:

```shell script
pip install -U albumentations --no-binary imgaug,albumentations
```