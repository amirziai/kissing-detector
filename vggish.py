from typing import Tuple

import torch.nn as nn
from torch import hub

VGGISH_WEIGHTS = (
    "https://users.cs.cf.ac.uk/taylorh23/pytorch/models/vggish-cbfe8f1c.pth"
)
PCA_PARAMS = (
    "https://users.cs.cf.ac.uk/taylorh23/pytorch/models/vggish_pca_params-4d878af3.npz"
)


class VGGishParams:
    """
    These should not be changed. They have been added into this file for convenience.
    """

    NUM_FRAMES = (96,)  # Frames in input mel-spectrogram patch.
    NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
    EMBEDDING_SIZE = 128  # Size of embedding layer.

    # Hyperparameters used in feature and example generation.
    SAMPLE_RATE = 16000
    STFT_WINDOW_LENGTH_SECONDS = 0.025
    STFT_HOP_LENGTH_SECONDS = 0.010
    NUM_MEL_BINS = NUM_BANDS
    MEL_MIN_HZ = 125
    MEL_MAX_HZ = 7500
    LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
    EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
    EXAMPLE_HOP_SECONDS = 0.96  # with zero overlap.

    # Parameters used for embedding postprocessing.
    PCA_EIGEN_VECTORS_NAME = "pca_eigen_vectors"
    PCA_MEANS_NAME = "pca_means"
    QUANTIZE_MIN_VAL = -2.0
    QUANTIZE_MAX_VAL = +2.0


"""
VGGish
Input: 96x64 1-channel spectrogram
Output:  128 Embedding 
"""


class VGGish(nn.Module):
    def __init__(self):
        super(VGGish, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, VGGishParams.NUM_BANDS, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(VGGishParams.NUM_BANDS, VGGishParams.EMBEDDING_SIZE, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 24, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, VGGishParams.EMBEDDING_SIZE),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embeddings(x)
        return x


def vggish() -> Tuple[VGGish, int]:
    """
    VGGish is a PyTorch implementation of Tensorflow's VGGish architecture used to create embeddings
    for Audioset. It produces a 128-d embedding of a 96ms slice of audio. Always comes pretrained.
    """
    model = VGGish()
    model.load_state_dict(hub.load_state_dict_from_url(VGGISH_WEIGHTS), strict=True)
    return model, VGGishParams.EMBEDDING_SIZE
