import math
import os
import pickle
import shutil
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from moviepy.editor import VideoFileClip

from torchvision import transforms

import params
import vggish_input


class BuildDataset:
    def __init__(self,
                 base_path: str,
                 videos_and_labels: List[Tuple[str, str]],
                 output_path: str,
                 n_augment: int = 1,
                 test_size: float = 1 / 3):
        assert 0 < test_size < 1
        self.videos_and_labels = videos_and_labels
        self.test_size = test_size
        self.output_path = output_path
        self.base_path = base_path
        self.n_augment = n_augment

        self.sets = ['train', 'val']

    def _get_set(self):
        return np.random.choice(self.sets, p=[1 - self.test_size, self.test_size])

    def build_dataset(self):
        # wipe
        for set_ in self.sets:
            path = f'{self.output_path}/{set_}'
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass
            os.makedirs(path)

        for file_name, label in self.videos_and_labels:
            name, _ = file_name.split('.')
            path = f'{self.base_path}/{file_name}'
            audio, images = self.one_video_extract_audio_and_stills(path)
            set_ = self._get_set()
            target = f"{self.output_path}/{set_}/{label}_{name}.pkl"
            pickle.dump((audio, images, label), open(target, 'wb'))

    @staticmethod
    def transform_reverse(img: torch.Tensor) -> Image:
        return transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0], std=(1.0 / params.std).tolist()),
            transforms.Normalize(mean=(-params.mean).tolist(), std=[1, 1, 1]),
            transforms.ToPILImage()])(img)

    @staticmethod
    def transformer(img_size: int):
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(params.mean, params.std)
        ])

    @classmethod
    def one_video_extract_audio_and_stills(cls,
                                           path_video: str,
                                           img_size: int = 224) -> Tuple[List[torch.Tensor],
                                                                         List[torch.Tensor]]:
        # return a list of image(s), audio tensors
        cap = cv2.VideoCapture(path_video)
        frame_rate = cap.get(5)
        images = []

        transformer = cls.transformer(img_size)

        # process the image
        while cap.isOpened():
            frame_id = cap.get(1)
            success, frame = cap.read()

            if not success:
                print('Something went wrong!')
                break

            if frame_id % math.floor(frame_rate * params.vggish_frame_rate) == 0:
                frame_pil = Image.fromarray(frame, mode='RGB')
                images.append(transformer(frame_pil))
                # images += [transformer(frame_pil) for _ in range(self.n_augment)]

        cap.release()

        # process the audio
        # TODO: hack to get around OpenMP error
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

        tmp_audio_file = 'tmp.wav'
        VideoFileClip(path_video).audio.write_audiofile(tmp_audio_file)
        # fix if n_augment > 1 by duplicating each sample n_augment times
        audio = vggish_input.wavfile_to_examples(tmp_audio_file)
        # audio = audio[:, None, :, :]  # add dummy dimension for "channel"
        # audio = torch.from_numpy(audio).float()  # Convert input example to float

        min_sizes = min(audio.shape[0], len(images))
        audio = [torch.from_numpy(audio[idx][None, :, :]).float() for idx in range(min_sizes)]
        images = images[:min_sizes]
        # images = [torch.from_numpy(img).permute((2, 0, 1)) for img in images[:min_sizes]]

        return audio, images
