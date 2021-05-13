<pre>
├── data_analysis
│   └── face_data_analysis.py
├── images - Graphics and images
│   ├──  ...
├── notebooks
│   ├── torch_test.ipynb
│   └── video_tool.ipynb
├── <strong>training</strong> - Main training folder 
│   ├── config-defaults.yaml - Training specific hyperparameter configs in wandb format
│   ├── datasets.py - data loading modules
│   ├── inference.py - Experimental, inference only
│   ├── <a href="../training/lightining_training.py">lightining_training.py</a> - Main training file, run this file to start training
│   ├── model_zoo.py - All heavy load, loading models, optimizers, schedulers, train/test/val steps happens here
│   └── transformers.py - augmentations
├── cleaning_with_coordinate_clustering.py
├── cleaning_with_face_recognition.py
├── clear_dataset.py
├── compare_real_vs_fake.py
├── config.py - except training hyperparameters, all paths, directories and other relevant information
├── Data Analysis.ipynb - Analyzing various aspects of data 
├── face_extraction_with_tracker.py - used to extract face images from test and validation sets.
├── face_extractor.py
├── face_extractor_v2.py
├── iou.py
├── _local_properties.py - template for local\_properties
├── local_properties.py - machine specific properties and secret keys
├── main.py - main file face extraction of training data. Refactoring needed here
├── npz2img.py
├── Playground.ipynb - Playing around with data, merging with data analysis?
├── README.md
├── requirments.txt
├── tmp_wsdan_results.ipynb
├── torch_test.py
├── training-deit-for-dfdc.ipynb
├── train.py
└── utils.py
</pre>
