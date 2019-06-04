# Kissing Detector
Detect kissing scenes in a movie using both audio and video features.

Project for [Stanford CS231N](http://cs231n.stanford.edu)

## Running the code
Use Python 3.6+
```bash
python3 experiments.py
```

this will run the experiments in `params.py` specified by the `experiments` dictionary.

## Build dataset
The following will build the dataset for training. You need to provide path to video segments.
```python
from pipeline import BuildDataset

videos_and_labels = [
    # (file name in base_path, label) where label is 1 for kissing and 0 for not kissing
    ('movies_casino_royale_2006_kissing_1.mp4', 1),
    ('movies_casino_royale_2006_kissing_2.mp4', 1),
    ('movies_casino_royale_2006_kissing_3.mp4', 1),
    ('movies_casino_royale_2006_not_1.mp4', 0),
    ('movies_casino_royale_2006_not_2.mp4', 0),
    ('movies_casino_royale_2006_not_3.mp4', 0),
    
    ('movies_goldeneye_1995_kissing_1.mp4', 1),
    ('movies_goldeneye_1995_kissing_2.mp4', 1),
    ('movies_goldeneye_1995_kissing_3.mp4', 1),
    ('movies_goldeneye_1995_not_1.mp4', 0),
    ('movies_goldeneye_1995_not_2.mp4', 0),
    ('movies_goldeneye_1995_not_3.mp4', 0),
]

builder = BuildDataset(base_path='path/to/movies',
                 videos_and_labels=videos_and_labels,
                 output_path='/path/to/output',
                 test_size=1 / 3)  # set aside 1 / 3 of data for validation
builder.build_dataset()
```

## Heavily used the following resources:
- [Video Classification Using 3D ResNet](https://github.com/kenshohara/video-classification-3d-cnn-pytorch)
- [CS231N assignment 3](http://cs231n.github.io/assignments2019/assignment3/)
