# Environment
- Python 3.8
- CUDA 11.0 Update 1
  
### Hardware specs
- CPU: Intel i7-9750H 2.6 GHz
- Nvidia GTX 1650 Max-Q

### Third party tools
- [Facenet-pytorch](https://github.com/timesler/facenet-pytorch)


### Known Problems
n (such as opencv-python-headless, opencv-python, opencv-contrib-python or opencv-contrib-python-headless) installed 
in your Python environment, you can force Albumentations installation with following command:

```shell script
pip install -U albumentations --no-binary imgaug,albumentations
```