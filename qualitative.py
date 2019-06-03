from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
import numpy as np

from pipeline import BuildDataset


class QualitativeAnalysis:
    def __init__(self,
                 model: nn.Module,
                 img_size: int,
                 videos_and_frames: Dict[str, List[int]],
                 class_names: Dict[int, str]):
        self.class_names = class_names
        self.img_size = img_size
        self.model = model
        self.videos_and_frames = videos_and_frames

        self.features = {
            vid: BuildDataset.one_video_extract_audio_and_stills(vid)
            for vid in self.videos_and_frames}

    # method is adapted from stanford cs231n assignment 3 available at:
    # http://cs231n.github.io/assignments2019/assignment3/
    @staticmethod
    def _compute_saliency_maps(A, I, y, model):
        """
        Compute a class saliency map using the model for images X and labels y.

        Input:
        - A: Input audio; Tensor of shape (N, 1, 96, 64)
        - I: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the saliency map.

        Returns:
        - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
        images.
        """
        # Make sure the model is in "test" mode
        model.eval()

        # Make input tensor require gradient
        # A.requires_grad_()
        I.requires_grad_()

        scores = model(A, I).gather(1, y.view(-1, 1)).squeeze()
        scores.backward(torch.ones(scores.size()))
        saliency, _ = torch.max(I.grad.abs(), dim=1)

        return saliency

    # also adapted from cs231n assignment 3
    def _show_saliency_maps(self, A, I, y):
        # Convert X and y from numpy arrays to Torch Tensors
        I_tensor = torch.cat([
            BuildDataset.transformer(self.img_size)(Image.fromarray(i)).unsqueeze(0)
            for i in I], dim=0)
        A_tensor = torch.cat([a.unsqueeze(0) for a in A])
        y_tensor = torch.LongTensor(y)

        # Compute saliency maps for images in X
        saliency = self._compute_saliency_maps(A_tensor, I_tensor, y_tensor, self.model)

        # Convert the saliency map from Torch Tensor to numpy array and show images
        # and saliency maps together.
        saliency = saliency.numpy()
        N = len(I)
        for i in range(N):
            plt.subplot(2, N, i + 1)
            plt.imshow(I[i])
            plt.axis('off')
            plt.title(self.class_names[y[i]])
            plt.subplot(2, N, N + i + 1)
            plt.imshow(saliency[i], cmap=plt.cm.hot)
            plt.axis('off')
            plt.gcf().set_size_inches(12, 5)
        plt.show()

    @staticmethod
    def _img_transform_reverse_to_np(x: torch.Tensor) -> np.array:
        rev = BuildDataset.transform_reverse(x)
        return np.array(rev)

    def saliency_maps(self):
        for vid, indices in self.videos_and_frames.items():
            A = [self.features[vid][0][idx] for idx in indices]
            I = [self._img_transform_reverse_to_np(self.features[vid][1][idx])
                 for idx in indices]
            y = [1 if 'kissing' in vid else 0] * len(A)
            self._show_saliency_maps(A, I, y)
            print('=' * 10)
