import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL.Image import Image
from matplotlib.pyplot import figure, imshow, axis
from pytube import YouTube
from torch import nn

from pipeline import BuildDataset

# images constituting a segments and the length in seconds
Segment = Tuple[List[Image], int]


class Segmentor:
    def __init__(self,
                 model: nn.Module,
                 min_frames: int,
                 threshold: float):
        self.model = model
        self.min_frames = min_frames
        self.threshold = threshold

    @staticmethod
    def _segmentor(preds: List[int],
                   min_frames: int,
                   threshold: float) -> List[List[int]]:
        candidates = []

        n = len(preds)

        for idx_start in range(n):
            if preds[idx_start] == 1:
                if n - idx_start >= min_frames:
                    best_here = (-1, (-1, -1))
                    for idx_end in range(idx_start + min_frames - 1, len(preds)):
                        if preds[idx_end] == 1:
                            if np.mean(preds[idx_start:idx_end + 1]) >= threshold:
                                frames = idx_end - idx_start + 1
                                endpoints = (idx_start, idx_end)
                                if frames > best_here[0]:
                                    best_here = (frames, endpoints)
                    if best_here[0] > 0:
                        candidates.append(best_here[1])

        overlap = True
        while overlap:
            overlap = False
            for i in range(len(candidates)):
                ref_idx_start, ref_idx_end = candidates[i]

                for j in range(i + 1, len(candidates)):
                    comp_idx_start, comp_idx_end = candidates[j]
                    if ref_idx_start <= comp_idx_end <= ref_idx_end or ref_idx_start <= comp_idx_start <= ref_idx_end:
                        # overlapping, take the longer one
                        if comp_idx_end - comp_idx_end > ref_idx_end - ref_idx_start:
                            del candidates[i]
                        else:
                            del candidates[j]
                        overlap = True

                    if overlap:
                        break

                if overlap:
                    break

        return [list(range(idx_start, idx_end + 1)) for idx_start, idx_end in candidates]

    @staticmethod
    def _torch_img_to_pil(img: torch.Tensor) -> Image:
        return BuildDataset.transform_reverse(img)

    @staticmethod
    def _get_segment_len(indices: List[int]):
        return max(indices) - min(indices) + 1

    def segmentor(self, preds: List[int], images: List[torch.Tensor]) -> List[Segment]:
        segment_list = self._segmentor(preds, self.min_frames, self.threshold)
        return [
            ([self._torch_img_to_pil(images[idx])
              for idx in segment_idx], self._get_segment_len(segment_idx))
            for segment_idx in segment_list]

    def _predict(self, audio: torch.Tensor, image: torch.Tensor) -> int:
        return int(torch.max(self.model(audio.unsqueeze(0), image.unsqueeze(0)), 1)[1][0])

    def get_segments(self, path_video: str) -> List[Segment]:
        audio, images = BuildDataset.one_video_extract_audio_and_stills(path_video)
        preds = [self._predict(audio[idx], images[idx]) for idx in range(len(images))]
        return self.segmentor(preds, images)

    @staticmethod
    def show_images_horizontally(images: List[Image]) -> None:
        # https://stackoverflow.com/questions/36006136/how-to-display-images-in-a-row-with-ipython-display
        fig = figure(figsize=(20, 20))
        number_of_files = len(images)
        for i in range(number_of_files):
            a = fig.add_subplot(1, number_of_files, i + 1)
            image = images[i]
            imshow(image)
            axis('off')
        plt.show()

    def visualize_segments(self, path_video: str, n_to_show: int = 10) -> None:
        segments = self.get_segments(path_video)
        n_segments = len(segments)
        print(f'Found {len(segments)} segments')

        if n_segments > 0:
            for i, (segment_images, segment_len) in enumerate(segments):
                print(f'Segment {i + 1}, {segment_len} seconds')
                print(f'First {n_to_show}')
                self.show_images_horizontally(segment_images[:n_to_show])

                print(f'{n_to_show} random shots')
                self.show_images_horizontally(random.sample(segment_images, n_to_show))

                print('Last 10')
                self.show_images_horizontally(segment_images[-n_to_show:])
                print('=' * 10)

    def visualize_segments_youtube(self,
                                   youtube_id: str,
                                   n_to_show: int = 10,
                                   show_title: bool = True,
                                   remove_file: bool = True):
        yt = YouTube(f'http://youtube.com/watch?v={youtube_id}')
        if show_title:
            print(f'Title: {yt.title}')
        yt_stream = yt.streams.first()
        path = f'{yt_stream.default_filename}'
        yt_stream.download()
        self.visualize_segments(path, n_to_show)
        if remove_file:
            os.remove(path)
