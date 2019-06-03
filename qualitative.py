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

    # next few methods taken from cs231n
    @staticmethod
    def jitter(X, ox, oy):
        """
        Helper function to randomly jitter an image.

        Inputs
        - X: PyTorch Tensor of shape (N, C, H, W)
        - ox, oy: Integers giving number of pixels to jitter along W and H axes

        Returns: A new PyTorch Tensor of shape (N, C, H, W)
        """
        if ox != 0:
            left = X[:, :, :, :-ox]
            right = X[:, :, :, -ox:]
            X = torch.cat([right, left], dim=3)
        if oy != 0:
            top = X[:, :, :-oy]
            bottom = X[:, :, -oy:]
            X = torch.cat([bottom, top], dim=2)
        return X

    def create_class_visualization(target_y, model, dtype, **kwargs):
        """
        Generate an image to maximize the score of target_y under a pretrained model.

        Inputs:
        - target_y: Integer in the range [0, 1000) giving the index of the class
        - model: A pretrained CNN that will be used to generate the image
        - dtype: Torch datatype to use for computations

        Keyword arguments:
        - l2_reg: Strength of L2 regularization on the image
        - learning_rate: How big of a step to take
        - num_iterations: How many iterations to use
        - blur_every: How often to blur the image as an implicit regularizer
        - max_jitter: How much to gjitter the image as an implicit regularizer
        - show_every: How often to show the intermediate result
        """
        model.type(dtype)
        l2_reg = kwargs.pop('l2_reg', 1e-3)
        learning_rate = kwargs.pop('learning_rate', 25)
        num_iterations = kwargs.pop('num_iterations', 100)
        blur_every = kwargs.pop('blur_every', 10)
        max_jitter = kwargs.pop('max_jitter', 16)
        show_every = kwargs.pop('show_every', 25)

        # Randomly initialize the image as a PyTorch Tensor, and make it requires gradient.
        img = torch.randn(1, 3, 224, 224).mul_(1.0).type(dtype).requires_grad_()

        for t in range(num_iterations):
            # Randomly jitter the image a bit; this gives slightly nicer results
            ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
            img.data.copy_(jitter(img.data, ox, oy))

            ########################################################################
            # TODO: Use the model to compute the gradient of the score for the     #
            # class target_y with respect to the pixels of the image, and make a   #
            # gradient step on the image using the learning rate. Don't forget the #
            # L2 regularization term!                                              #
            # Be very careful about the signs of elements in your code.            #
            ########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            target = model(img)[0, target_y]
            target.backward()
            g = img.grad.data
            g -= 2 * l2_reg * img.data
            img.data += learning_rate * (g / g.norm())
            img.grad.zero_()

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            ########################################################################
            #                             END OF YOUR CODE                         #
            ########################################################################

            # Undo the random jitter
            img.data.copy_(jitter(img.data, -ox, -oy))

            # As regularizer, clamp and periodically blur the image
            for c in range(3):
                lo = float(-SQUEEZENET_MEAN[c] / SQUEEZENET_STD[c])
                hi = float((1.0 - SQUEEZENET_MEAN[c]) / SQUEEZENET_STD[c])
                img.data[:, c].clamp_(min=lo, max=hi)
            if t % blur_every == 0:
                blur_image(img.data, sigma=0.5)

            # Periodically show the image
            if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
                plt.imshow(deprocess(img.data.clone().cpu()))
                class_name = class_names[target_y]
                plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
                plt.gcf().set_size_inches(4, 4)
                plt.axis('off')
                plt.show()

        return deprocess(img.data.cpu())
