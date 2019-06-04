import copy
import functools
import json
import os
import pickle
from glob import glob
from typing import Tuple, List

import torch
import torch.utils.data as data
from PIL import Image


class AV(data.Dataset):
    def __init__(self, path: str):
        self.path = path
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return self.data[idx]


class AudioVideo(AV):
    def __init__(self, path: str):
        # output format:
        # return (
        #     torch.rand((1, 96, 64)),
        #     torch.rand((3, 224, 224)),
        #     np.random.choice([0, 1])
        # )
        super().__init__(path)

        for file_path in glob(f'{path}/*.pkl'):
            audios, images, label = pickle.load(open(file_path, 'rb'))
            self.data += [(audios[i], images[i], label) for i in range(len(audios))]


class AudioVideo3D(AV):
    def __init__(self, path: str):
        # output format:
        # return (
        #     torch.rand((1, 96, 64)),
        #     torch.rand((3, 16, 224, 224)),
        #     np.random.choice([0, 1])
        # )
        super().__init__(path)
        frames = 16

        for file_path in glob(f'{path}/*.pkl'):
            audios, images, label = pickle.load(open(file_path, 'rb'))
            images_temporal = self._process_temporal_tensor(images, frames)
            self.data += [(audios[i], images_temporal[i], label) for i in range(len(audios))]

    @staticmethod
    def _process_temporal_tensor(images: List[torch.Tensor],
                                 frames: int) -> List[torch.Tensor]:
        out = []

        for i in range(len(images)):
            e = torch.zeros((frames, 3, 224, 224))
            e[-1] = images[0]
            for j in range(min(i, frames)):
                e[-1 - j] = images[j]
                # try:
                #     e[-1 - j] = images[j]
                # except:
                #     raise ValueError(f"trying to get {i} from images with len = {len(images)}")
            ee = e.permute((1, 0, 2, 3))
            out.append(ee)
        return out


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    # try:
    #     return accimage.Image(path)
    # except IOError:
    #     # Potentially a decoding problem, fall back to PIL.Image
    #     return pil_loader(path)
    return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            if subset == 'testing':
                video_names.append('test/{}'.format(key))
            else:
                label = value['annotations']['label']
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(video_path, sample_duration):
    dataset = []

    n_frames = len(os.listdir(video_path))

    begin_t = 1
    end_t = n_frames
    sample = {
        'video': video_path,
        'segment': [begin_t, end_t],
        'n_frames': n_frames,
    }

    step = sample_duration
    for i in range(1, (n_frames - sample_duration + 1), step):
        sample_i = copy.deepcopy(sample)
        sample_i['frame_indices'] = list(range(i, i + sample_duration))
        sample_i['segment'] = torch.IntTensor([i, i + sample_duration - 1])
        dataset.append(sample_i)

    return dataset


class Video(data.Dataset):
    def __init__(self, video_path,
                 spatial_transform=None, temporal_transform=None,
                 sample_duration=16, get_loader=get_default_video_loader):
        self.data = make_dataset(video_path, sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]['segment']

        return clip, target

    def __len__(self):
        return len(self.data)
