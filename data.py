import pickle
from glob import glob
from typing import Tuple, List

import torch
import torch.utils.data as data


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
