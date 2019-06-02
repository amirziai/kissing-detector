import math
import os
import pickle
import shutil
from typing import List, Tuple

import cv2
import numpy as np
import torch
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

import vggish_input

VGGISH_FRAME_RATE = 0.96


def slice_clips(segments, root, fps=2):
    for path, classes in segments.items():

        for cls, ts in classes.items():
            for i, (t1, t2) in enumerate(ts):
                set_ = np.random.choice(['train', 'val'], p=[2 / 3, 1 / 3])
                # get all the still frames
                file_name, ext = path.split('.')
                target = f"{root}{file_name}_{cls}_{i + 1}.{ext}"
                print(f'target: {target}')
                ffmpeg_extract_subclip(f'{root}{path}', t1, t2, targetname=target)
                vidcap = cv2.VideoCapture(target)
                vidcap.set(cv2.CAP_PROP_FPS, fps)
                print(cv2.CAP_PROP_FPS)
                success, image = vidcap.read()
                count = 0
                while success:
                    frame_path = f'{root}casino/{set_}/{cls}/{file_name}_{i}_{count + 1}.jpg'
                    # print(frame_path)
                    cv2.imwrite(frame_path, image)  # save frame as JPEG file
                    success, image = vidcap.read()
                    # print('Read a new frame: ', success)
                    count += 1


class BuildDataset:
    def __init__(self,
                 base_path: str,
                 videos_and_labels: List[Tuple[str, str]],
                 output_path: str,
                 test_size: float = 1 / 3):
        assert 0 < test_size < 1
        self.videos_and_labels = videos_and_labels
        self.test_size = test_size
        self.output_path = output_path
        self.base_path = base_path

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
    def one_video_extract_audio_and_stills(path_video: str) -> Tuple[List[torch.Tensor],
                                                                     List[torch.Tensor]]:
        # return a list of image(s), audio tensors
        cap = cv2.VideoCapture(path_video)
        frame_rate = cap.get(5)
        images = []

        # process the image
        while cap.isOpened():
            frame_id = cap.get(1)
            success, frame = cap.read()

            if not success:
                print('Something went wrong!')
                break

            if frame_id % math.floor(frame_rate * VGGISH_FRAME_RATE) == 0:
                images.append(frame)

        cap.release()

        # process the audio
        # TODO: hack to get around OpenMP error
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

        tmp_audio_file = 'tmp.wav'
        VideoFileClip(path_video).audio.write_audiofile(tmp_audio_file)
        audio = vggish_input.wavfile_to_examples(tmp_audio_file)
        # audio = audio[:, None, :, :]  # add dummy dimension for "channel"
        # audio = torch.from_numpy(audio).float()  # Convert input example to float

        min_sizes = min(audio.shape[0], len(images))
        audio = [torch.from_numpy(audio[idx][None, :, :]).float() for idx in range(min_sizes)]
        images = [torch.from_numpy(img).permute((2, 1, 0)) for img in images[:min_sizes]]

        return audio, images
